from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
import numpy as np
import cv2
import tensorflow as tf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Charger le modèle
model = tf.keras.models.load_model("static/cat_dog_model.keras")

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Fonction de prétraitement
def preprocess_image(image_bytes):
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img.reshape(1, 150, 150, 1).astype("float32") / 255.0
    return img

# Route pour afficher les images uploadées
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route principale
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="Aucun fichier envoyé")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="Fichier non sélectionné")

        try:
            # Écraser l'image précédente
            filename = "last_uploaded.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Prédiction
            with open(filepath, "rb") as f:
                img = preprocess_image(f.read())
            prediction_result = model.predict(img)
            is_cat = prediction_result[0][0] >= 0.5
            prediction = "😺 Chat" if is_cat else "🐶 Chien"
            confidence = float(prediction_result[0][0])

            # URL pour afficher l'image dans le HTML
            image_url = url_for('uploaded_file', filename=filename)

        except Exception as e:
            return render_template('index.html', error=f"Erreur lors de la prédiction : {str(e)}")

    return render_template('index.html', prediction=prediction, confidence=confidence, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
