from flask import Flask, render_template, request
import os

from detect_utils import load_model, run_detection

# Create Flask app FIRST
app = Flask(__name__)

# Load model once at startup
model = load_model("weights/best.pt")


@app.route("/", methods=["GET", "POST"])
def index():
    result_image_url = None
    detections = []

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            # Ensure folders exista
            os.makedirs("static/uploads", exist_ok=True)
            os.makedirs("static/outputs", exist_ok=True)

            # Save uploaded image
            upload_path = os.path.join("static/uploads", file.filename)
            file.save(upload_path)

            # Run detection
            result_path, detections = run_detection(model, upload_path)

            # Make path usable in HTML
            result_image_url = "/" + result_path.lstrip("./")

    return render_template(
        "index.html",
        result_image_url=result_image_url,
        detections=detections,
    )



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
