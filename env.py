# -*- coding: utf-8 -*-
# @Time    : 2024/7/21 19:17
# @Author  : chenfeng
# @Email   : zlf100518@163.com
# @File    : env.py
# @LICENSE : MIT

import win32gui
import pyautogui
from PIL import Image, ImageGrab
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import time

class AutoJumpEnv():
    def __init__(self, hwnd: int=0, dpi: float=1.0, resource_pack: str='./resource', device = torch.device('cpu')) -> None:
        self.hwnd = hwnd
        self.dpi = dpi
        self.device = device
        self.model = torch.jit.load('./model/mnist_cnn.pt').to(self.device)
        self.resource_pack = resource_pack
        self.player = cv2.imread(f'{self.resource_pack}/player.png')
        self.end_img = cv2.imread(f'{self.resource_pack}/end.png')
        self.end_img = cv2.cvtColor(self.end_img, cv2.COLOR_BGR2GRAY)
        self.last_score = 0

    def __get_window_rect(self, hwnd: int) -> tuple[int, int, int, int]:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        top += 45
        return left, top, right - left, bottom - top

    def __get_screenshot(self, hwnd: int, dpi: float=1) -> Image.Image:
        left, top, width, height = self.__get_window_rect(hwnd)
        img = ImageGrab.grab(bbox=(left * dpi, top * dpi,
                                (left + width) * dpi, (top + height) * dpi))
        return img

    def __digital_divide(self, img, reverse=True) -> list:
        ans = []
        kernel = np.ones((2,2),np.uint8)
        img = cv2.bilateralFilter(img, 15, 20, 30)
        dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 19, 5)
        dst = cv2.dilate(dst, kernel, iterations=2)
        dst = cv2.erode(dst, kernel, iterations=3)
        img = 255 - dst if reverse else dst
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour, hier in zip(contours, hierarchy[0]) if hier[3] == -1]
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x -= 5
            y -= 5
            w += 10
            h += 10
            image = img[y: y + h, x: x + w]
            orig_height, orig_width = image.shape[:2]
            top, bottom = (80 - orig_height) // 2, (80 - orig_height) // 2 + (80 - orig_height) % 2  
            left, right = (80 - orig_width) // 2, (80 - orig_width) // 2 + (80 - orig_width) % 2  
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])  
            image = image.reshape((1, 80, 80))
            ans.append(image)
        return ans

    def __get_score(self, img) -> int:
        ans = 0
        width, height = img.size
        score = img.crop((50, 100, width - 10, 230))
        img = np.array(score)
        img_list = self.__digital_divide(img, reverse=True)
        for i in img_list:
            tensor_img = torch.tensor(i, dtype=torch.float32).to(self.device)
            y = self.model(tensor_img)
            _, y = torch.max(y, dim=1)
            ans *= 10
            ans += y.item()
        return ans
    
    def __multi_scale_search(self, pivot, screen, range=0.3, num=10):
        H, W = screen.shape[:2]
        h, w = pivot.shape[:2]

        found = None
        for scale in np.linspace(1-range, 1+range, num)[::-1]:
            resized = cv2.resize(screen, (int(W * scale), int(H * scale)))
            r = W / float(resized.shape[1])
            if resized.shape[0] < h or resized.shape[1] < w:
                break
            res = cv2.matchTemplate(resized, pivot, cv2.TM_CCOEFF_NORMED)

            loc = np.where(res >= res.max())
            pos_h, pos_w = list(zip(*loc))[0]

            if found is None or res.max() > found[-1]:
                found = (pos_h, pos_w, r, res.max())

        if found is None: return [0,0,0,0,0]
        pos_h, pos_w, r, score = found
        y, x = int(pos_h * r), int(pos_w * r)
        end_h, end_w = int((pos_h + h) * r), int((pos_w + w) * r)
        height = end_h - y
        width = end_w - x
        return [x, y, width, height, score]
    
    def score(self):
        img = self.__get_screenshot(self.hwnd, self.dpi)
        img = img.convert('L')
        score = self.__get_score(img)
        return score

    def state(self):
        img = self.__get_screenshot(self.hwnd, self.dpi)
        img = img.resize((600, 1280), Image.ANTIALIAS)
        img = np.array(img)
        start_y = 230
        y = start_y + 600
        img = img[start_y: y, :, :]
        # img = cv2.Canny(img, 40, 10)
        # img = np.expand_dims(img, axis=-1)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float().to(self.device)
        return img
    
    def end(self):
        img = self.__get_screenshot(self.hwnd, self.dpi)
        img = img.convert('L')
        img = np.array(img)
        pos = self.__multi_scale_search(self.end_img, img, 0.3, 10)
        if pos == [0, 0, 0, 0, 0] or pos[4] < 0.5:
            return False
        return True

    def step(self, action):
        action = action / 1000
        img = self.state()
        end_game = self.end()
        if end_game:
            score = self.last_score
            self.reset()
            return img, -2, end_game, score
        x, y = 2200, 1200
        pyautogui.moveTo(x, y)
        pyautogui.mouseDown(button='left')
        pyautogui.sleep(action)
        pyautogui.mouseUp(button='left')
        now_score = self.score()
        score = now_score - self.last_score
        self.last_score = now_score
        if score == 0:
            score = -1
        elif score > 2:
            score = 1
        return img, score, end_game, now_score

    def reset(self):
        self.last_score = 0
        pos = self.__get_window_rect(self.hwnd)
        pyautogui.moveTo(2304, 1097)
        pyautogui.click(button='left')

if __name__ == '__main__':
    env = AutoJumpEnv(hwnd=0x000E099C, dpi=1)
    img = env.state()
    print(img.shape)
    img = img.permute(1, 2, 0)
    img = img.numpy()
    # img = (img * 255).astype(np.uint8)
    plt.imshow(img, cmap='gray')
    plt.show()
    # img = img.permute(1, 2, 0)
    # img = img.numpy()
    # cv2.namedWindow('trackbar', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('trackbar', 600, 800)
    # def call_back(value):
    #     print(value)
    # cv2.createTrackbar('canny_min', 'trackbar', 40, 255, call_back)
    # cv2.createTrackbar('canny_max', 'trackbar', 10, 255, call_back)
    # while True:
    #     canny_min = cv2.getTrackbarPos('canny_min', 'trackbar')
    #     canny_max = cv2.getTrackbarPos('canny_max', 'trackbar')
    #     img1 = cv2.Canny(img, canny_min, canny_max)
    #     cv2.imshow('trackbar', img1)
    #     k = cv2.waitKey(1)
    #     if k == ord('q'):
    #         break
    # img = (img * 255).astype(np.uint8)
    # plt.imshow(img, cmap='gray')
    # plt.show()