"""Run Road Vision on a CARLA vehicle's forward-facing camera."""

import argparse
import queue
import random
from pathlib import Path

import carla
import cv2
import numpy as np
from ultralytics import YOLO

from detection import process_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic", type=int, default=20)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames; 0 means unlimited")
    parser.add_argument("--no-display", action="store_true", help="Process frames without opening an OpenCV window")
    args = parser.parse_args()
    if args.traffic < 0 or args.width <= 0 or args.height <= 0 or args.max_frames < 0:
        parser.error("Traffic and frame limit must be non-negative; image dimensions must be positive")

    model = YOLO(str(Path(__file__).with_name("model.pt")))
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    actors = []
    camera = None
    frames = queue.Queue(maxsize=1)

    def on_image(image):
        # Keep the newest frame if inference is slower than the simulator.
        try:
            frames.put_nowait(image)
        except queue.Full:
            try:
                frames.get_nowait()
            except queue.Empty:
                pass
            try:
                frames.put_nowait(image)
            except queue.Full:
                pass

    try:
        world = client.get_world()
        blueprints = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        vehicle_types = list(blueprints.filter("vehicle.*"))
        if not spawn_points or not vehicle_types:
            raise RuntimeError("The current CARLA map has no vehicle spawn points or blueprints")

        for point in spawn_points:
            vehicle = world.try_spawn_actor(random.choice(vehicle_types), point)
            if vehicle is not None:
                actors.append(vehicle)
                vehicle.set_autopilot(True)
                if len(actors) >= args.traffic + 1:
                    break
        if not actors:
            raise RuntimeError("Could not spawn a vehicle; try restarting CARLA or clearing traffic")
        print(f"Spawned ego vehicle and {len(actors) - 1} traffic vehicles.", flush=True)

        blueprint = blueprints.find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(args.width))
        blueprint.set_attribute("image_size_y", str(args.height))
        blueprint.set_attribute("fov", "90")
        blueprint.set_attribute("sensor_tick", "0.05")
        camera = world.spawn_actor(
            blueprint,
            carla.Transform(carla.Location(x=1.5, z=2.0)),
            attach_to=actors[0],
        )
        camera.listen(on_image)
        count = 0
        while True:
            try:
                image = frames.get(timeout=20)
            except queue.Empty as exc:
                raise RuntimeError("No camera frames received from CARLA for 20 seconds") from exc
            bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
            frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            result = process_frame(frame, model)
            count += 1
            if not args.no_display:
                cv2.imshow("Road Vision - CARLA (Q to quit)", result)
                if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty("Road Vision - CARLA (Q to quit)", cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_frames and count >= args.max_frames:
                break
        print(f"Processed {count} CARLA camera frames.", flush=True)
    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        for actor in reversed(actors):
            actor.destroy()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
