# -*- coding: utf-8 -*-
# @Time    : 2024/7/21 19:17
# @Author  : chenfeng
# @Email   : zlf100518@163.com
# @File    : env.py
# @LICENSE : MIT

import win32gui
from PIL import Image, ImageGrab
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import os

class AutoJumpEnv():
    def __init__(self, hwnd: int=0, dpi: float=1.0, resource_pack: str='./resource') -> None:
        self.hwnd = hwnd
        self.dpi = dpi
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load('./model/mnist_cnn.pt').to(self.device)
        self.resource_pack = resource_pack
        self.player = cv2.imread(f'{self.resource_pack}/player.png')

    def get_window_rect(self, hwnd: int) -> tuple[int, int, int, int]:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        top += 45
        return left, top, right - left, bottom - top

    def get_screenshot(self, hwnd: int, dpi: float=1) -> tuple[Image.Image, Image.Image]:
        left, top, width, height = self.get_window_rect(hwnd)
        img = ImageGrab.grab(bbox=(left * dpi, top * dpi,
                                (left + width) * dpi, (top + height) * dpi))
        return img

    def digital_divide(self, img) -> list:
        ans = []
        kernel = np.ones((2,2),np.uint8)
        width, height = img.size
        score = img.crop((50, 100, width - 10, 200))
        img = np.array(score)
        img = cv2.bilateralFilter(img, 15, 20, 30)
        dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 19, 5)
        dst = cv2.dilate(dst, kernel, iterations=2)
        dst = cv2.erode(dst, kernel, iterations=3)
        img = 255 - dst
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

    def get_score(self, img) -> int:
        ans = 0
        img_list = self.digital_divide(img)
        for i in img_list:
            tensor_img = torch.tensor(i, dtype=torch.float32).to(self.device)
            y = self.model(tensor_img)
            _, y = torch.max(y, dim=1)
            ans *= 10
            ans += y.item()
        return ans

    def multi_scale_search(self, pivot, screen, range=0.3, num=10):
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
        h = end_h - y
        w = end_w - x
        return [x, y, w, h, score]
    
    def get_player_pos(self, img) -> list[int]:
        return self.multi_scale_search(self.player, img)

    def score(self, img):
        img = img.convert('L')
        score = self.get_score(img)
        return score
    
    def get_target_pos(self, img) -> list[int]:
        import time
        print(img.shape)
        s = time.time()
        start = os.path.join(self.resource_pack, 'target')
        file = os.listdir(start)
        ans = [0, 0, 0, 0, 0]
        for f in file:
            path = os.path.join(start, f)
            image = cv2.imread(path)
            resu = self.multi_scale_search(image, img)
            if resu[-1] > ans[-1]:
                ans = resu
        print(time.time() - s)
        return ans


    def test(self):
        image = self.get_screenshot(hwnd, dpi)
        img = np.array(image)
        print(img.shape)

        # x, y, w, h, score = self.get_player_pos(img)
        # print(self.score(image))
        # hi, wi, ci = img.shape
        # img1 = img[200:y + h, :, :]
        # xt, yt, wt, ht, score = self.get_target_pos(img1)
        # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # img = cv2.rectangle(img, (0, 200), (wi, y + h), (255, 0, 0), 2)
        # img = cv2.rectangle(img, (xt, 200 + yt), (xt + wt, 200 + yt + ht), (0, 0, 255), 2)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        
if __name__ == '__main__':
    dpi = 1.5
    hwnd = 0x00010672
    auto_jump = AutoJumpEnv(hwnd=hwnd, dpi=dpi, resource_pack='./resource')
    auto_jump.test()