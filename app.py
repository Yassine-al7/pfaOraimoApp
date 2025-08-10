from flask import Flask, render_template, request, flash, redirect, url_for
from ultralytics import YOLO
import os
import uuid
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Add template function for cache busting
@app.template_global()
def moment():
    return datetime.now()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """Clean up old uploaded files (older than 1 hour)"""
    import time
    try:
        current_time = time.time()
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                if file_age > 3600:  # 1 hour
                    os.remove(file_path)
                    logger.info(f"Cleaned up old file: {filename}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

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
    """Home page with model metrics"""
    cleanup_old_files()  # Clean up old files on each visit
    return render_template('index.html', metrics=model_metrics)

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    """Image detection page"""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'image' not in request.files:
                flash('Aucun fichier sélectionné', 'error')
                return redirect(request.url)
            
            image = request.files['image']
            model_name = request.form.get('model')
            
            # Validate inputs
            if image.filename == '':
                flash('Aucun fichier sélectionné', 'error')
                return redirect(request.url)
            
            if model_name not in models:
                flash('Modèle non valide', 'error')
                return redirect(request.url)
            
            if not allowed_file(image.filename):
                flash('Type de fichier non autorisé. Utilisez: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP', 'error')
                return redirect(request.url)
            
            # Secure filename and save
            original_filename = secure_filename(image.filename)
            file_extension = original_filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            image.save(image_path)
            logger.info(f"Saved uploaded image: {unique_filename}")
            
            # Run detection
            logger.info(f"Running {model_name} detection on {unique_filename}")
            results = models[model_name](image_path)
            
            # Save results with detection boxes
            results[0].save(filename=image_path)
            
            # Create web-accessible path
            web_image_path = f"/static/uploads/{unique_filename}"
            
            flash(f'Détection réussie avec {model_name}!', 'success')
            return render_template('detect.html', 
                                 image_path=web_image_path, 
                                 model=model_name,
                                 filename=unique_filename)
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            flash(f'Erreur lors de la détection: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('detect.html', image_path=None)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash("Fichier trop volumineux. Taille maximale: 16MB", 'error')
    return redirect(url_for('detect'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("🚀 Lancement de l'application YOLO Detection...")
    print(f"📁 Dossier d'upload: {UPLOAD_FOLDER}")
    print(f"🤖 Modèles chargés: {list(models.keys())}")
    print("🌐 Application disponible sur: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
