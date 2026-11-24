import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from src.AnimalRecognization import logger
from src.AnimalRecognization.entity.config_entity import PredictionConfig


class PredictionService:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model = None
        self.class_names = []

    def load_model(self):
        """Lädt das trainierte MobileNetV2 .h5 Modell."""
        if self.model is None:
            logger.info(f"Lade Modell von {self.config.trained_model_path}")
            self.model = tf.keras.models.load_model(str(self.config.trained_model_path))
        return self.model

    def load_class_names(self):
        """Lädt Klassennamen aus labels.txt (oder JSON / Verzeichnis-Fallback)."""
        class_file = str(self.config.class_names_file)

        if os.path.exists(class_file):
            if class_file.endswith(".json"):
                with open(class_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.class_names = [k for k, _ in sorted(data.items(), key=lambda x: x[1])]
                    else:
                        self.class_names = data
            else:
                with open(class_file, "r", encoding="utf-8") as f:
                    self.class_names = [line.strip() for line in f if line.strip()]

            logger.info(f"{len(self.class_names)} Klassennamen geladen aus {class_file}")
            return self.class_names

        base_ingestion = Path(self.config.root_dir).parent / "data_ingestion"
        nested_paths = [
            base_ingestion / "animal-image-dataset-90-different-animals" / "animals" / "animals",
            base_ingestion / "animal-image-dataset-90-different-animals" / "animals",
            base_ingestion / "train",
            base_ingestion,
        ]

        class_path = None
        for path in nested_paths:
            if path.exists():
                subdirs = [d.name for d in path.iterdir() if d.is_dir() and d.name not in {"train", "validate"}]
                if subdirs:
                    class_path = path
                    break

        if class_path:
            self.class_names = sorted([d.name for d in class_path.iterdir() if d.is_dir()])
            os.makedirs(os.path.dirname(class_file), exist_ok=True)
            if class_file.endswith(".json"):
                with open(class_file, "w", encoding="utf-8") as f:
                    json.dump(self.class_names, f, indent=4)
            else:
                with open(class_file, "w", encoding="utf-8") as f:
                    for name in self.class_names:
                        f.write(f"{name}\n")
            logger.info(f"{len(self.class_names)} Klassennamen gespeichert unter {class_file}")
        else:
            logger.warning("Keine Klassenverzeichnisse gefunden. Nutze Platzhalter-Namen.")
            num_classes = getattr(self.config, "params_num_classes", 90)
            self.class_names = [f"class_{i}" for i in range(num_classes)]

        return self.class_names

    def preprocess_image(self, image_path):
        """Bereitet ein Bild für das Modell vor (RGB + [0, 1] Skalierung)."""
        img = Image.open(image_path).convert("RGB")
        img = img.resize(tuple(self.config.params_image_size))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path):
        """Führt eine Einzelbild-Vorhersage durch."""
        model = self.load_model()
        if not self.class_names:
            self.load_class_names()

        img_array = self.preprocess_image(image_path)
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_idx])

        if predicted_class_idx < len(self.class_names):
            class_name = self.class_names[predicted_class_idx]
        else:
            class_name = f"class_{predicted_class_idx}"

        return {
            "class_name": class_name,
            "confidence": confidence,
            "class_index": predicted_class_idx,
        }

    def setup_flask_app(self):
        """Erstellt die Flask-Webapp unter webapp_dir."""
        os.makedirs(self.config.webapp_dir, exist_ok=True)

        app_py_content = """from flask import Flask, request, render_template, jsonify
import os
import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.AnimalRecognization.pipeline.stage_05_prediction import PredictionPipeline

app = Flask(__name__)

predictor = PredictionPipeline()
prediction_service = predictor.get_prediction_service()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
    file.save(temp_path)

    try:
        result = prediction_service.predict(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
"""

        with open(os.path.join(self.config.webapp_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write(app_py_content)

        templates_dir = os.path.join(self.config.webapp_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)

        index_html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Animal Recognition</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; text-align: center; }
        .upload-container { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
        #result { margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; display: none; }
        #preview { max-width: 300px; margin: 15px auto; display: none; }
    </style>
</head>
<body>
    <h1>Animal Recognition</h1>
    <div class="upload-container">
        <h2>Upload an animal image for classification</h2>
        <input type="file" id="imageUpload" accept="image/*">
        <img id="preview" src="#" alt="Preview">
    </div>
    <div id="result"></div>

    <script>
        document.getElementById('imageUpload').addEventListener('change', function(e) {
            if (!e.target.files || !e.target.files[0]) return;

            const preview = document.getElementById('preview');
            preview.src = URL.createObjectURL(e.target.files[0]);
            preview.style.display = 'block';

            const formData = new FormData();
            formData.append('file', e.target.files[0]);

            fetch('/predict', { method: 'POST', body: formData })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('result');
                if (data.error) {
                    resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
                } else {
                    resultDiv.innerHTML = `
                        <h3>Prediction Result:</h3>
                        <p><strong>Animal:</strong> ${data.class_name}</p>
                        <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                    `;
                }
                resultDiv.style.display = 'block';
            })
            .catch(error => {
                document.getElementById('result').innerHTML = `<p>Error: ${error}</p>`;
                document.getElementById('result').style.display = 'block';
            });
        });
    </script>
</body>
</html>
"""

        with open(os.path.join(templates_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(index_html_content)

        logger.info(f"Flask-App eingerichtet in {self.config.webapp_dir}")
