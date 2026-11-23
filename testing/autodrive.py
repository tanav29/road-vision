"""Conservative, opt-in controls for a Windows browser game only."""
import ctypes
import threading
import time

KEYS = {'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27}
F8, F9 = 0x77, 0x78


def decide(lane, boxes, width, height, now):
    """Return keys for one control tick; bounding boxes are (class, x1, y1, x2, y2)."""
    for name, x1, y1, x2, y2 in boxes:
        if name.lower() not in {'car', 'truck', 'bus', 'motorcycle', 'motorbike', 'bicycle', 'person'}:
            continue
        # A lower, sufficiently large box intersecting the travel corridor is a hazard.
        if x2 > width * .35 and x1 < width * .65 and y2 > height * .55 and (x2-x1)*(y2-y1) > width*height*.012:
            return {'down'}, 'OBSTACLE: braking'
    if lane is None:
        return {'down'}, 'LANE LOST: braking'
    # Duty cycles keep both speed and steering low without relying on vehicle speed.
    keys = {'up'} if now % 1.0 < .25 else set()
    if now % .6 < .12:
        if lane < -.07:
            keys.add('left')
        elif lane > .07:
            keys.add('right')
    return keys, 'DRIVING'


class WindowsKeys:
    def __init__(self):
        if not hasattr(ctypes, 'windll'):
            raise RuntimeError('Autonomous mode requires Windows.')
        self.user32 = ctypes.windll.user32
        self.user32.GetForegroundWindow.restype = ctypes.c_void_p
        self.user32.WindowFromPoint.restype = ctypes.c_void_p
        self.user32.GetAncestor.restype = ctypes.c_void_p
        self.user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    def window_at(self, x, y):
        class POINT(ctypes.Structure):
            _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]
        return self.user32.GetAncestor(self.user32.WindowFromPoint(POINT(x, y)), 2)

    def foreground(self):
        return self.user32.GetForegroundWindow()

    def pressed(self, key):
        return bool(self.user32.GetAsyncKeyState(key) & 0x8000)

    def key(self, name, down):
        self.user32.keybd_event(KEYS[name], 0, 1 if down else 3, 0)


class Controller:
    def __init__(self, backend, game_window, timeout=1.5):
        if not game_window:
            raise ValueError('Could not identify the game window.')
        self.backend, self.game_window, self.timeout = backend, game_window, timeout
        self.lock = threading.Lock()
        self.held = set()
        self.armed = False
        self.reason = 'DISARMED: F8 to arm, F9 to stop'
        self.last_update = 0
        self.held_until = 0
        self.closed = False
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def _release(self):
        for key in tuple(self.held):
            self.backend.key(key, False)
            self.held.remove(key)

    def _stop(self, reason):
        self.armed = False
        self._release()
        self.reason = reason

    def _watch(self):
        previous_f8 = False
        while True:
            with self.lock:
                if self.closed:
                    return
                f8 = self.backend.pressed(F8)
                if self.backend.pressed(F9):
                    self._stop('STOPPED: F9; F8 to re-arm')
                elif self.armed and self.backend.foreground() != self.game_window:
                    self._stop('STOPPED: game lost focus')
                elif self.armed and time.monotonic() - self.last_update > self.timeout:
                    self._stop('STOPPED: detection timed out')
                elif self.held and time.monotonic() >= self.held_until:
                    self._release()
                elif f8 and not previous_f8 and self.backend.foreground() == self.game_window:
                    self.armed = True
                    self.last_update = time.monotonic()
                    self.reason = 'ARMED: waiting for detection'
                previous_f8 = f8
            time.sleep(.02)

    def update(self, keys, reason):
        with self.lock:
            if not self.armed:
                return self.reason
            if self.backend.foreground() != self.game_window:
                self._stop('STOPPED: game lost focus')
                return self.reason
            for key in self.held - keys:
                self.backend.key(key, False)
                self.held.remove(key)
            for key in keys - self.held:
                self.backend.key(key, True)
                self.held.add(key)
            self.held_until = time.monotonic() + .2
            self.last_update = time.monotonic()
            self.reason = reason
            return reason

    def close(self):
        with self.lock:
            self.closed = True
            self._stop('STOPPED')
        self.thread.join()
