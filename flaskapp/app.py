from flask import Flask, render_template, request, jsonify
from shoulder import Shoulder

import numpy as np
import cv2


app = Flask(__name__)

app.config["JSON_AS_ASCII"] = False


@app.route('/')
def index():
    return render_template('index.html', title='ProtType')

@app.route('/upload', methods=["POST"])
def upload():
    if request.files['image']:
        # 画像読み込み
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        # 肩検出
        shoulder = Shoulder(img)
        result, save_path = shoulder.detect()

        return render_template('upload.html', title='result', result=result, img=save_path)
    else:
        return "empty error"

if __name__ == "__main__":
    app.run(debug=True)