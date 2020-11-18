from image import MyImage

import numpy as np
import cv2


class Shoulder(object):
    def __init__(self, img):
        self.raw_image = img
        self.color_image = None
        self.gray_image = None
        self.canny_image = None
        self.hough_lines = []

    # 画像を読み込む
    def get_resize_rate(self, width, height):
        if width > 2000 or height > 2000:
            x_rate = 0.15
            y_rate = 0.15
        elif width > 1000 or height > 1000:
            x_rate = 0.5
            y_rate = 0.5
        else:
            x_rate = 1
            y_rate = 1
        return x_rate, y_rate

    def get_gray_image(self):
        height, width, channels = MyImage.get_size(self.raw_image)
        x_rate, y_rate = self.get_resize_rate(width, height)
        self.color_image = cv2.resize(self.raw_image, dsize=None, fx=x_rate, fy=y_rate)
        self.gray_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
    
    # canny変換
    def convert_canny_image(self):
        self.canny_image = cv2.Canny(self.gray_image, 50, 110)

    # 確率的ハフ変換
    def hough_lines_p(self):
        self.hough_lines = cv2.HoughLinesP(
            self.canny_image, rho=1, theta=np.pi/360, threshold=50, minLineLength=80, maxLineGap=10
        )

    # 結果
    def detect(self):
        self.get_gray_image()
        self.convert_canny_image()
        self.hough_lines_p()
        height, width, channels = MyImage.get_size(self.color_image)
        xline = []
        yline = []
        for line in self.hough_lines:
            x1, y1, x2, y2 = line[0]
            xa = (x2-x1)
            ya = (y2-y1)
            # 画面サイズ半分以上の直線を除く
            if xa < width/2:
                # 上部200pxまでの直線を除く
                if y1>200 or y2>200:
                    # 傾き2以上を除く
                    if xa > 25:
                        if ya>-50 and ya<50:
                            # 描画
                            cv2.line(self.color_image,(x1,y1),(x2,y2),(0,0,255),2)

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