import os
import io
import pathlib

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask import Flask, request, jsonify

# ================== CẤU HÌNH ================== #
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

# File .tflite đặt trong thư mục anemia_app/model/
TFLITE_PATH = SCRIPT_DIR / "model" / "efficientnetb0_anemia_v2.tflite"

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Anemia", "Non_anemia"]     # 0 = Anemia, 1 = Non_anemia
THRESHOLD = 0.4                            # Ngưỡng quyết định

# ================== LOAD TFLITE MODEL ================== #
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)


def predict_pil_image(pil_img: Image.Image):
    """
    Nhận ảnh PIL, tiền xử lý và chạy suy luận bằng TFLite.
    Trả về: (label, p_anemia, p_non_anemia)
    """
    # Resize về đúng kích thước mô hình yêu cầu
    pil_img = pil_img.resize(IMG_SIZE)

    arr = np.array(pil_img).astype("float32")

    # Ảnh xám: nhân lên 3 kênh
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    # RGBA: bỏ kênh alpha
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]

    # (1, H, W, 3)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    # Chạy mô hình TFLite
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    # Giả sử output[0, 0] là xác suất Non_anemia
    p_non_anemia = float(output[0, 0])
    p_anemia = 1.0 - p_non_anemia

    pred_idx = 1 if p_non_anemia >= THRESHOLD else 0
    pred_label = CLASS_NAMES[pred_idx]

    return pred_label, p_anemia, p_non_anemia


@app.route("/predict", methods=["POST"])
def predict():
    """
    Nhận ảnh từ app (multipart/form-data, field name = 'file')
    Trả về JSON:
    {
      "success": true/false,
      "result": {...} hoặc "error": "..."
    }
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        label, p_anemia, p_non_anemia = predict_pil_image(pil_img)

        # Đổi sang tiếng Việt + độ tin cậy
        if label == "Anemia":
            label_vi = "Thiếu máu"
            confidence = p_anemia
        else:
            label_vi = "Bình thường"
            confidence = p_non_anemia

        return jsonify({
            "success": True,
            "result": {
                "label": label,              # 'Anemia' / 'Non_anemia'
                "label_vi": label_vi,        # 'Thiếu máu' / 'Bình thường'
                "prob_anemia": p_anemia,
                "prob_non_anemia": p_non_anemia,
                "confidence": confidence     # xác suất của nhãn dự đoán
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/", methods=["GET"])
def root():
    return "Anemia TFLite API is running", 200


if __name__ == "__main__":
    # Render sẽ truyền PORT qua biến môi trường
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
