from flask import Flask, render_template, request
from shoulder import Shoulder
from image import Image

import numpy as np
import cv2

SAVE_DIR = "./static/images"

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='flask test')

@app.route('/upload', methods=["POST"])
def upload():
    if request.files['image']:
        # 送信された画像を保存する
        # 画像読み込み
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        # 画像保存
        img_name = Image.save(img)

        # 肩検出
        shoulder = Shoulder(img_name)
        result, save_path = shoulder.detect()

        return render_template('upload.html', title='result', result=result, img=save_path)
    else:
        return "empty error"

if __name__ == "__main__":
    app.run(debug=True)