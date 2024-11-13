import time
import numpy as np
import keyboard
import argparse
import bettercam

from colorama import Fore
from ultralytics import YOLO
from mouse_instruct import MouseInstruct

SMOOTH_X = 0.8
SMOOTH_Y = 0.8

class Apex:
    def __init__(self, VID, PID, PING_CODE):
        self.cam = bettercam.create(output_color="RGB")
        self.mouse = MouseInstruct.get_mouse(VID, PID, PING_CODE)
        self.model = YOLO('./best_8s.pt').to('cuda')

    def get_xy(self):
        frame = self.cam.grab(region=(640, 220, 1280, 860))
        dx, dy = None, None
        if frame is None:
            return dx, dy

        results = self.model.predict(frame, verbose=False, conf=0.45)
        if results and len(results) > 0:
            enemy_boxes = results[0].boxes.xyxy.cpu().numpy()
            min_dist = float('inf')

            center_x, center_y = 320, 320

            # Finding the closest enemy
            for box in enemy_boxes:
                x1, y1, x2, y2 = box

                enemy_center_x = (x1 + x2) / 2
                enemy_center_y = (y1 + y2) / 2

                tx = enemy_center_x - center_x
                ty = enemy_center_y - center_y

                distance = np.hypot(tx, ty)

                if distance < min_dist:
                    min_dist = distance
                    dx, dy = tx, ty - 10

        return dx, dy

    def update(self):
        while True:
            if keyboard.is_pressed('o'):
                self.cam.stop()
                exit(0)

            if keyboard.is_pressed('alt'):
                dx, dy = self.get_xy()
                if dx is not None and dy is not None:
                    # x_mult = min(0.6, abs((dx - 320) / 100))
                    # y_mult = min(0.6, abs((dy - 320) / 100))
                    start = time.perf_counter()
                    self.mouse.move(int(dx * SMOOTH_X), int(dy * SMOOTH_Y))
                    print(f"Magnet {(time.perf_counter() - start):.4f}")

            if keyboard.is_pressed('v'):
                dx, dy = self.get_xy()
                if dx is not None and dy is not None:
                    start = time.perf_counter()
                    self.mouse.silent_flick(int(dx * 1.4), int(dy * 1.4))
                    print(f"Silent flick {(time.perf_counter() - start):.4f}")
            else:
                time.sleep(0.01)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid', help='Vendor ID of your mouse', default=None)
    parser.add_argument('--pid', help='Product ID of your mouse', default=None)
    parser.add_argument('--pcode', help='Pingcode of your mouse', default=None)

    args = parser.parse_args()

    print(f'{Fore.LIGHTWHITE_EX}Initializing YOLO...', end='')
    apex = Apex(int(args.vid, 16), int(args.pid, 16), int(args.pcode, 16))
    print(f'\r{Fore.LIGHTWHITE_EX}Initialized         ', end='\n', flush=True)
    print(f'{Fore.LIGHTWHITE_EX}"Alt" - Magnet')
    print(f'{Fore.LIGHTWHITE_EX} "V"  - Silent')
    print(f'{Fore.LIGHTWHITE_EX} "O"  - Exit  ')
    apex.update()
