"""Deterministic, one-lane crosswalk experiment. Positions refer to the car's nose."""
import argparse
from collections import deque
from dataclasses import dataclass
import math
import sys
import time


DT = 0.05
BRAKING = 4.0  # m/s^2
STOP_MARGIN = 2.0
CROSSING_WIDTH = 3.0
CAR_LENGTH = 4.5
FINISH = 14.0


@dataclass
class Event:
    name: str
    time: float
    position: float
    speed: float


def beyond_van(position, lateral):
    """Line of sight from car (position, 0) to walker (1.5, lateral).

    The van occupies x=-8..-2 m and y=1.8..3.8 m beside the approach.
    """
    if position >= -2 or lateral <= 0:
        return True
    near = max(position, -8)
    slope = lateral / (1.5 - position)
    return slope * (-2 - position) < 1.8 or slope * (near - position) > 3.8


def simulate(scenario, cruise, delay, animate=False):
    position, speed = -35.0, cruise
    samples = deque()
    observed = False
    brake = False
    events = {}
    delay_ticks = math.ceil(delay / DT - 1e-10)

    def record(name):
        if name not in events:
            events[name] = Event(name, now, position, speed)

    for tick in range(2401):
        now = tick * DT
        # The pedestrian appears on the sidewalk at 1 s, then walks across.
        lateral = 3.6 - 0.85 * (now - 1)
        crossing = scenario != "empty" and now >= 1 and lateral >= -2.0
        visible = crossing and (scenario != "van" or beyond_van(position, lateral))
        if visible:
            record("first visible")
        samples.append(visible)
        if len(samples) > delay_ticks:
            observed = samples.popleft()
        if observed:
            record("first detected")
        if scenario != "empty" and lateral < -2:
            record("crosswalk clear")

        # Use only the delayed observation and the car's own state to drive.
        # Begin a full stop when the stopping envelope reaches the yield line.
        stopping_distance = speed ** 2 / (2 * BRAKING)
        brake = observed and position < CROSSING_WIDTH and (
            brake or -position <= stopping_distance + STOP_MARGIN + speed * DT
        )
        if brake and speed > 0:
            record("brake commanded")
        if speed == 0:
            record("stopped")
            if not brake:
                record("resume commanded")

        if scenario != "empty" and now >= 1 and abs(lateral) <= 1.2 and (
            position >= 1.2 and position - CAR_LENGTH <= 1.8
        ):
            record("collision")
            return events, "FAIL: collision"
        if crossing and position >= 0:
            record("entered before clear")
        if animate and tick % 4 == 0:
            road = list("." * 58)
            road[35:38] = "|||"
            road[max(0, min(57, round(position + 35)))] = ">"
            print(f"\r{now:5.1f}s [{''.join(road)}] {speed:4.1f} m/s  "
                  f"walker {'visible' if visible else 'hidden' if crossing else 'absent':7} "
                  f"{'BRAKE' if brake else 'DRIVE'}", end="", flush=True)
            time.sleep(4 * DT)
        if position >= FINISH:
            record("route complete")
            if "entered before clear" in events:
                return events, "FAIL: entered occupied crossing"
            if scenario == "empty" and "brake commanded" in events:
                return events, "FAIL: unnecessary braking"
            return events, "PASS"

        next_speed = max(0.0, speed - BRAKING * DT) if brake else min(cruise, speed + 2 * DT)
        position += (speed + next_speed) * 0.5 * DT
        speed = next_speed
    return events, "FAIL: timed out"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("pedestrian", "empty", "van", "all"), default="all")
    parser.add_argument("--speed", type=float, default=8, help="approach speed in m/s (1 to 25)")
    parser.add_argument("--delay", type=float, default=0.2, help="sensor delay in seconds (0 to 10)")
    parser.add_argument("--animate", action="store_true", help="show a real-time terminal replay")
    args = parser.parse_args()
    if not 1 <= args.speed <= 25 or not 0 <= args.delay <= 10:
        parser.error("speed must be 1 to 25 m/s and delay must be 0 to 10 s")
    failed = False
    scenarios = ("pedestrian", "empty", "van") if args.scenario == "all" else (args.scenario,)
    for scenario in scenarios:
        print(f"\n{scenario} | {args.speed:g} m/s | sensor delay {args.delay:g} s")
        events, outcome = simulate(scenario, args.speed, args.delay, args.animate)
        print("\nevent                   time (s)   nose (m)  speed (m/s)")
        for event in events.values():
            print(f"{event.name:23} {event.time:8.2f} {event.position:10.2f} {event.speed:12.2f}")
        print(outcome)
        failed |= outcome != "PASS"
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
