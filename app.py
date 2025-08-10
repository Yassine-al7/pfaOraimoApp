from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dictionnaire des modèles
models = {
    'YOLOv8': YOLO('models/yolov8.pt'),
    'YOLOv9': YOLO('models/best_spacebuds8.pt'),
    'YOLOv10': YOLO('models/yolov10.pt'),
    'YOLOv11': YOLO('models/yolov11.pt'),
}

# Tu peux remplacer ces valeurs par les vraies plus tard
model_metrics = {
    'YOLOv8': {'mAP': '98,3%', 'Precision': '98,3%', 'Recall': '97,9%'},
    'YOLOv9': {'mAP': '93.2%', 'Precision': '90%', 'Recall': '87%'},
    'YOLOv10': {'mAP': '94%', 'Precision': '86%', 'Recall': '86,8%'},
    'YOLOv11': {'mAP': '98.2%', 'Precision': '98,2%', 'Recall': '97,8%'},
}

@app.route('/')
def index():
    return render_template('index.html', metrics=model_metrics)

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        image = request.files['image']
        model_name = request.form['model']
        if image.filename == '':
            return 'No image selected'

        # Save uploaded image
        image_id = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_id)
        image.save(image_path)

        # Run detection
        results = models[model_name](image_path)
        results[0].save(filename=image_path)

        return render_template('detect.html', image_path=image_path, model=model_name)
    return render_template('detect.html', image_path=None)

if __name__ == '__main__':
    print("Lancement de Flask...")
    app.run(debug=True)
