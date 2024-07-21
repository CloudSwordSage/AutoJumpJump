import keyboard
from env import AutoJumpEnv
import numpy as np
import cv2
import os

env = AutoJumpEnv()
dpi = 1.5
hwnd = 0x00010672

train = 0
test = 400
root = './data'

def hotkey_pressed():
    global train, test
    img = env.get_screenshot(hwnd, dpi)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if train > 0:
        print(f'\rtrain: {3000 - train + 1} / 3000', end='')
        path = os.path.join(root, 'train')
        file_name = f'train_{3000 - train}.png'
        path = os.path.join(path, file_name)
        cv2.imwrite(path, img)
        train -= 1
    elif test > 0:
        print(f'\rtest: {400 - test + 1} / 1000', end='')
        path = os.path.join(root, 'test')
        file_name = f'test_{400 - test}.png'
        path = os.path.join(path, file_name)
        cv2.imwrite(path, img)
        test -= 1
    else:
        exit()

keyboard.add_hotkey('ctrl+shift+a', hotkey_pressed)

keyboard.wait()