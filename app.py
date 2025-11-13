# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from ultralytics import YOLO
# from PIL import Image
# import io, base64

# app = Flask(__name__)
# CORS(app)

# # Load YOLO model
# model = YOLO("bestModel2.pt")

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/detect", methods=["POST"])
# def detect():
#     file = request.files.get('image')
#     if not file:
#         print("❌ No file received")
#         return jsonify({"error": "No image provided"}), 400

#     print("✅ File received:", file.filename)
#     image = Image.open(file.stream)

#     # Run detection with 30% confidence
#     results = model.predict(image, conf=0.3)
#     count = len(results[0].boxes)
#     print("🍊 Fruits detected:", count)

#     # Convert result image to base64
#     plotted_img = results[0].plot()
#     img_pil = Image.fromarray(plotted_img)
#     buf = io.BytesIO()
#     img_pil.save(buf, format="PNG")
#     img_str = base64.b64encode(buf.getvalue()).decode()

#     return jsonify({
#         "count": count,
#         "confidence": 0.1,
#         "image": img_str
#     })

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io, base64
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO("bestModel2.pt")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get('image')
    if not file:
        print("❌ No file received")
        return jsonify({"error": "No image provided"}), 400

    print("✅ File received:", file.filename)
    image = Image.open(file.stream)

    # Run YOLO detection with 30% confidence
    results = model.predict(image, conf=0.3)
    count = len(results[0].boxes)
    print("🍊 Fruits detected:", count)

    # Plot results and fix color (BGR → RGB)
    plotted_img = results[0].plot()          # returns OpenCV BGR image
    plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(plotted_img_rgb)

    # Convert to base64 for frontend
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "count": count,
        "confidence": 0.3,      # 30% confidence threshold
        "image": img_str
    })

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=10000)
