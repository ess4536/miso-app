from flask import Flask, render_template, request, jsonify
from shoulder import Shoulder
from image import MyImage

import numpy as np
import cv2


app = Flask(__name__)

app.config["JSON_AS_ASCII"] = False


@app.route('/')
def index():
    return render_template('index.html', title='ProtType')


@app.route('/shoulder', methods=["POST"])
def shoulder():
    
    # return jsonify(return_json)
    stream = request.files['image'].stream
    img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)

    save_path = MyImage.save(img)
    result = "画像集めるためのアプリ"
    # shoulder = Shoulder(img)
    # result, save_path = shoulder.detect()

    # return "\'{\"result\":\"" + result + "\",\"img\":\"" + save_path + "\"}\'"
    return result + "," + save_path

if __name__ == "__main__":
    app.run(debug=True)