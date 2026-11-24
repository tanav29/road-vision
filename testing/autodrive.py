"""Conservative, opt-in controls for a Windows browser game only."""
import ctypes
from ctypes import wintypes
import os
import threading
import time

KEYS = {'up': 0x57, 'down': 0x53, 'left': 0x41, 'right': 0x44}  # W, S, A, D
CTRL, ALT, ARM, STOP = 0x11, 0x12, 0x47, 0x58


def front_limits(y, width, height, boundaries):
    """Horizontal limits of the vehicle's forward path at ground-image row y."""
    depth = min(1., max(.55, y/height))
    if boundaries is None:
        half_width = width * (.055 + .24*(depth-.55))
        return width/2-half_width, width/2+half_width
    # LaneTracker stores each marking's x at 90% and 62% of image height.
    t = (depth-.62)/.28
    left, right = (top + t*(bottom-top) for bottom, top in boundaries)
    center = (left+right)/2
    half_width = min(.18*width, max(0., (right-left)*.40))
    return center-half_width, center+half_width


def in_front(x1, x2, y2, width, height, boundaries):
    if y2 < .55*height:
        return False
    left, right = front_limits(y2, width, height, boundaries)
    # Use the middle half of the box base as the tire/foot contact area.
    inset = .25*(x2-x1)
    return x1+inset < right and x2-inset > left and right > left


class DrivePlanner:
    """Choose held game keys from consecutive road observations."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.steering = None
        self.side = None
        self.ahead_since = None
        self.started = self.retry_after = self.brake_until = 0.
        self.braking = False
        self.last_frame = None
        self.previous_lane = None

    def decide(self, lane, boxes, width, height, now, adjacent, boundaries=None):
        if self.last_frame is not None and now - self.last_frame > .6:
            self.reset()
        self.last_frame = now
        blocked = set()
        ahead = urgent = False
        for name, x1, y1, x2, y2 in boxes:
            name = name.lower()
            if name not in {'car', 'truck', 'bus', 'motorcycle', 'motorbike', 'bicycle', 'person'}:
                continue
            if y2 < .48*height:
                continue
            if x1 < .41*width and x2 > .03*width:
                blocked.add('left')
            if x2 > .59*width and x1 < .97*width:
                blocked.add('right')
            area = (x2-x1)*(y2-y1)/(width*height)
            if area > .004 and in_front(x1, x2, y2, width, height, boundaries):
                ahead = True
                urgent |= name == 'person' or (y2 > .72*height and area > .025)

        if urgent and not self.braking:
            self.brake_until = now + .18
        self.braking = urgent
        if urgent or lane is None:
            self.side = self.steering = self.ahead_since = self.previous_lane = None
            if urgent:
                self.retry_after = now + .6
                return ({'down'} if now < self.brake_until else set()), 'OBSTACLE: yielding'
            return set(), 'LANE LOST: controls released'

        if not ahead:
            self.ahead_since = None
        elif self.ahead_since is None:
            self.ahead_since = now

        if self.side:
            # When straddling the divider, the detected current lane switches
            # to the destination lane. Hand steering back to its centerline.
            sign = -1 if self.side == 'left' else 1
            crossed = (lane - self.previous_lane)*sign > .16
            if crossed or self.side in blocked or self.side not in adjacent or now-self.started > 1.6:
                self.side = self.steering = None
                self.retry_after = now + 1.5
        self.previous_lane = lane
        if (self.side is None and self.ahead_since is not None
                and now-self.ahead_since >= .15 and now >= self.retry_after):
            available = [side for side in adjacent if side not in blocked]
            if available:
                self.side = min(available, key=lambda side: abs(adjacent[side]))
                self.started = now

        target = adjacent[self.side] if self.side else lane
        # Separate engagement and release thresholds avoid key chatter near center.
        if target > .09:
            self.steering = 'right'
        elif target < -.09:
            self.steering = 'left'
        elif ((self.steering == 'right' and target < .035)
              or (self.steering == 'left' and target > -.035)):
            self.steering = None
        keys = {self.steering} if self.steering else set()
        if self.side or (not ahead and abs(target) < .42):
            keys.add('up')
        if self.side:
            return keys, f'CHANGING LANE: {self.side}'
        return keys, 'TRAFFIC: coasting' if ahead else 'DRIVING'


def decide(lane, boxes, width, height, now):
    """Single-frame decision for callers without a lane tracker."""
    return DrivePlanner().decide(lane, boxes, width, height, now, {})


class WindowsKeys:
    def __init__(self):
        if not hasattr(ctypes, 'windll'):
            raise RuntimeError('Autonomous mode requires Windows.')
        self.user32 = ctypes.windll.user32
        self.user32.GetForegroundWindow.restype = ctypes.c_void_p
        self.user32.WindowFromPhysicalPoint.argtypes = [wintypes.POINT]
        self.user32.WindowFromPhysicalPoint.restype = ctypes.c_void_p
        self.user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        self.user32.GetAncestor.restype = ctypes.c_void_p
        self.user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    def window_at(self, x, y):
        hit = self.user32.WindowFromPhysicalPoint(wintypes.POINT(x, y))
        return self.user32.GetAncestor(hit, 2)

    def foreground(self):
        return self.user32.GetAncestor(self.user32.GetForegroundWindow(), 2)

    def capture_target(self, point):
        window = self.window_at(*point)
        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
        if pid.value == os.getpid():
            return None  # Capture preview or destination map.
        return window

    def pressed(self, key):
        return bool(self.user32.GetAsyncKeyState(key) & 0x8000)

    def key(self, name, down):
        # Scan-code input is understood by more browser game engines than virtual keys.
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [('wVk', ctypes.c_ushort), ('wScan', ctypes.c_ushort),
                        ('dwFlags', ctypes.c_ulong), ('time', ctypes.c_ulong),
                        ('dwExtraInfo', ctypes.c_size_t)]
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [('dx', ctypes.c_long), ('dy', ctypes.c_long),
                        ('mouseData', ctypes.c_ulong), ('dwFlags', ctypes.c_ulong),
                        ('time', ctypes.c_ulong), ('dwExtraInfo', ctypes.c_size_t)]
        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [('uMsg', ctypes.c_ulong), ('wParamL', ctypes.c_ushort),
                        ('wParamH', ctypes.c_ushort)]
        class INPUTUNION(ctypes.Union):
            _fields_ = [('ki', KEYBDINPUT), ('mi', MOUSEINPUT), ('hi', HARDWAREINPUT)]
        class INPUT(ctypes.Structure):
            _anonymous_ = ('data',)
            _fields_ = [('type', ctypes.c_ulong), ('data', INPUTUNION)]
        scan = self.user32.MapVirtualKeyW(KEYS[name], 0)
        event = INPUT(type=1, ki=KEYBDINPUT(0, scan, 8 | (0 if down else 2), 0, 0))
        if self.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) != 1:
            raise ctypes.WinError()


class Controller:
    def __init__(self, backend, game_window=None, timeout=1.5, capture_point=None):
        self.capture_point = capture_point
        if not game_window and capture_point is None:
            raise ValueError('Could not identify the game window.')
        self.backend, self.game_window, self.timeout = backend, game_window, timeout
        self.lock = threading.Lock()
        self.held = set()
        self.armed = False
        self.reason = 'DISARMED: Ctrl+Alt+G to arm, Ctrl+Alt+X to stop'
        self.last_update = 0
        self.throttle_start = None
        self.throttle_until = 0
        self.held_until = 0
        self.closed = False
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _release(self):
        self.throttle_start = None
        for key in tuple(self.held):
            self.backend.key(key, False)
            self.held.remove(key)

    def _stop(self, reason):
        self.armed = False
        self._release()
        self.reason = reason

    def _watch(self):
        previous_arm = False
        while True:
            with self.lock:
                if self.closed:
                    return
                modifiers = self.backend.pressed(CTRL) and self.backend.pressed(ALT)
                arm = modifiers and self.backend.pressed(ARM)
                if modifiers and self.backend.pressed(STOP):
                    self._stop('STOPPED: Ctrl+Alt+X; Ctrl+Alt+G to re-arm')
                elif self.armed and self.backend.foreground() != self.game_window:
                    self._stop('STOPPED: click game, then Ctrl+Alt+G')
                elif self.armed and time.monotonic() - self.last_update > self.timeout:
                    self._stop('STOPPED: detection timed out')
                elif self.held and time.monotonic() >= self.held_until:
                    self._release()
                elif 'up' in self.held and time.monotonic() >= self.throttle_until:
                    self.backend.key('up', False)
                    self.held.remove('up')
                elif arm and not previous_arm:
                    target = (self.backend.capture_target(self.capture_point)
                              if self.capture_point is not None else self.game_window)
                    if target and self.backend.foreground() == target and self.game_window in (None, target):
                        self.game_window = target
                        self.armed = True
                        self.last_update = time.monotonic()
                        self.reason = 'ARMED: waiting for detection'
                    else:
                        self.reason = 'Move previews aside, click game, then Ctrl+Alt+G'
                previous_arm = arm
            time.sleep(.02)

    def update(self, keys, reason):
        with self.lock:
            if not self.armed:
                return self.reason
            if self.backend.foreground() != self.game_window:
                self._stop('STOPPED: click game, then Ctrl+Alt+G')
                return self.reason
            # Do not send driving keys while the arm shortcut modifiers are held.
            if self.backend.pressed(CTRL) or self.backend.pressed(ALT):
                self._release()
                self.last_update = time.monotonic()
                return 'ARMED: release Ctrl and Alt'
            # Cruise using short W taps. The watchdog ends each tap even
            # when the next detection takes longer than the pulse.
            now = time.monotonic()
            if 'up' in keys:
                if self.throttle_start is None:
                    self.throttle_start = now
                phase = (now - self.throttle_start) % .4
                self.throttle_until = now + .12 - phase
                if phase >= .12:
                    keys = keys - {'up'}
            else:
                self.throttle_start = None
            for key in self.held - keys:
                self.backend.key(key, False)
                self.held.remove(key)
            for key in keys - self.held:
                # Track first so cleanup releases the key even if Ctrl+C lands during SendInput.
                self.held.add(key)
                self.backend.key(key, True)
            self.held_until = time.monotonic() + .2
            self.last_update = time.monotonic()
            self.reason = reason
            return reason

    def close(self):
        with self.lock:
            self.closed = True
            self._stop('STOPPED')
        self.thread.join()
