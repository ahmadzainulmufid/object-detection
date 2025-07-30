from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = Flask(__name__)
model = YOLO('best.pt') 

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Prediksi
            results = model.predict(image, conf=0.5)
            result = results[0]

            # Render bounding box ke gambar
            rendered_img = Image.fromarray(result.plot())

            # Konversi ke base64
            buffered = io.BytesIO()
            rendered_img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Ambil semua label deteksi
            class_names = [model.names[int(cls)] for cls in result.boxes.cls]
            class_summary = ', '.join(set(class_names)) if class_names else 'No detection'

            return render_template("index.html",
                                   uploaded=True,
                                   image_data=img_b64,
                                   filename=filename,
                                   prediction=class_summary)
    return render_template("index.html", uploaded=False)


if __name__ == '__main__':
    app.run(debug=True, port=5050)
