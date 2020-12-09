from image import MyImage

import numpy as np
import cv2


class Shoulder(object):
    def __init__(self, img):
        self.color_image = img
        self.gray_image = None
        self.canny_image = None
        self.detect_area = [50, 450, 300, 300]
        self.hough_lines = []

    def get_gray_image(self):
        self.gray_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
    
    # canny変換
    def convert_canny_image(self):
        self.canny_image = cv2.Canny(self.gray_image, 50, 110)

    # 確率的ハフ変換
    def hough_lines_p(self):
        self.hough_lines = cv2.HoughLinesP(
            self.canny_image, rho=1, theta=np.pi/360, threshold=50, minLineLength=80, maxLineGap=10
        )
    
    def detect_area_line(self, line):
        print(line)
        x1, y1, x2, y2 = line[0]
        xa = (x2-x1)
        ya = (y2-y1)

        # 画面サイズを取得
        height, width, channels = MyImage.get_size(self.color_image)

        # 長すぎる直線を除く
        if xa > (width/2):
            return "false"
        # 上部の直線を除く
        if y1<self.detect_area[0] or y2<self.detect_area[0]:
            return "false"
        # 右部の直線を除く
        if x1>self.detect_area[1] or x2>self.detect_area[1]:
            return "false"
        # 下部の直線を除く
        if y1>self.detect_area[2] or y2>self.detect_area[2]:
            return "false"
        # 左部の直線を除く
        if x1<self.detect_area[3] or x2<self.detect_area[3]:
            return "false"
        # x方向に短すぎる直線を除く
        if xa < 50:
            return "false"
        return "true"

    # 結果
    def detect(self):
        self.gray_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        self.canny_image = cv2.Canny(self.gray_image, 50, 110)
        self.hough_lines = cv2.HoughLinesP(
            self.canny_image, rho=1, theta=np.pi/360, threshold=50, minLineLength=80, maxLineGap=10
        )
        xline = []
        yline = []
        for line in self.hough_lines:
            # 描画条件
            is_range = self.detect_area_line(line)
            if is_range=="true":
                cv2.line(self.color_image,(x1,y1),(x2,y2),(0,0,255),2) # 描画

                line = np.append(line, [xa,ya])
                # 負の数を正の数に変換
                if(xa < 0):
                    xa = -xa
                if(ya < 0):
                    ya = -ya
                xline =np.append(xline, xa)
                yline =np.append(yline, ya)
        # 描画後の画像保存
        save_path = MyImage.save(self.color_image)
        # ここかえたい
        if (len(xline) != 2 or len(yline) != 2):
            result = "検出できませんでした。"
        else:
            if (yline[0]-yline[1] > 10) or (yline[0]-yline[1] < -10):
                result = "傾むいてます。"
            elif (xline[0]-xline[1] > 10) or (xline[0]-xline[1] < -10):
                result = "回転してます。"
            else:
                result = "OK"
        return result, save_path