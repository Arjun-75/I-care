import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.layers import Layer
import os
from flask import url_for


# ─────────────── Custom Cast Layer ───────────────
class Cast(Layer):
    def __init__(self, dtype='float32', **kwargs):
        super().__init__(**kwargs)
        self.target_dtype = tf.as_dtype(dtype)

    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'dtype': self.target_dtype.name})
        return config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load both models
model_oct = load_model('best_oct_model.h5')
model_fundus = load_model(
    'best_fundus_model.h5',
    custom_objects={'Cast': Cast},
    compile=False
)

# Define class names for each model
class_names_oct = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'normal']
class_names_fundus = ['DR', 'MH', 'ARMD', 'CSR']

disease_explanations = {
    'AMD': "Age-related macular degeneration (AMD) is a leading cause of vision loss in older adults. It affects the macula, the central part of the retina responsible for sharp, detailed vision. AMD can cause blurred or lost central vision, making it difficult to see faces, read, or perform other everyday tasks. Common in people over 50.",
    'CNV': "Choroidal neovascularization (CNV) is a condition where abnormal blood vessels grow in the choroid, a layer of tissue behind the retina. This can lead to fluid and blood leaking into the retina, damaging it and causing vision loss. CNV is often associated with age-related macular degeneration (AMD).",
    'CSR': "Central Serous Retinopathy (CSR), also known as Central Serous Chorioretinopathy (CSC), is a condition where fluid accumulates under the retina, causing blurred or distorted vision. It primarily affects the macula, the central part of the retina responsible for sharp, detailed vision. CSR is often linked to stress, high cortisol levels, or steroid use.",
    'DME': "Diabetic Macular Edema (DME) is a serious eye condition that occurs in people with diabetes, causing fluid buildup in the macula, the central part of the retina responsible for sharp, central vision. This fluid buildup leads to swelling and thickening of the macula, resulting in blurred or distorted vision. DME is a leading cause of vision loss and blindness in working-age adults with diabetes.",
    'DR': "Diabetic retinopathy (DR) is a serious eye disease caused by damage to the blood vessels in the retina due to diabetes. This damage can lead to vision loss and even blindness if left untreated. DR is the leading cause of vision loss in working-age adults.",
    'DRUSEN': "Drusen are small, yellow deposits that form under the retina, the light-sensitive tissue at the back of the eye. They are typically made of lipids and proteins and can be a sign of aging, with many people over 40 having some. While small, hard drusen are often harmless, larger, soft drusen, particularly in the macula (the central part of the retina), can be an early sign of age-related macular degeneration (AMD).",
    'MH': "Macular hole (MH) is a full-thickness defect in the retina at the center of the macula, causing central vision loss. This condition can lead to distortions in vision, like seeing wavy lines or a blind spot in the center. Macular holes are often idiopathic, meaning they develop without a clear cause, and they can be treated surgically to attempt to close the hole and improve vision.",
    'normal': "No abnormalities detected: Healthy retinal scan."
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(img_path, model_type):
    try:
        target_size = (224, 224)
        if model_type == 'oct':
            model = model_oct
            class_names = class_names_oct
        else:
            model = model_fundus
            class_names = class_names_fundus

        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict(img_array)
        return class_names[np.argmax(predictions)], float(np.max(predictions))
    except Exception as e:
        print(f"Error in {model_type} prediction: {str(e)}")
        return None, 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure both images are provided
    if 'oct_file' not in request.files or 'fundus_file' not in request.files:
        return jsonify({'error': 'Please upload both OCT and Fundus images.'}), 400

    oct_file = request.files['oct_file']
    fundus_file = request.files['fundus_file']

    if oct_file.filename == '' or fundus_file.filename == '':
        return jsonify({'error': 'Both OCT and Fundus files are required.'}), 400

    if not (allowed_file(oct_file.filename) and allowed_file(fundus_file.filename)):
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg.'}), 400

    uploaded_files = {'oct': None, 'fundus': None}
    confidence_scores = {}

    try:
        # Save and predict OCT
        oct_path = os.path.join(app.config['UPLOAD_FOLDER'], oct_file.filename)
        oct_file.save(oct_path)
        uploaded_files['oct'] = oct_path
        oct_pred, oct_conf = predict_image(oct_path, 'oct')
        if oct_pred:
            confidence_scores[oct_pred] = oct_conf

        # Save and predict Fundus
        fundus_path = os.path.join(app.config['UPLOAD_FOLDER'], fundus_file.filename)
        fundus_file.save(fundus_path)
        uploaded_files['fundus'] = fundus_path
        fundus_pred, fundus_conf = predict_image(fundus_path, 'fundus')
        if fundus_pred:
            # Map ARMD to AMD
            if fundus_pred == 'ARMD':
                fundus_pred = 'AMD'
            confidence_scores[fundus_pred] = confidence_scores.get(fundus_pred, 0) + fundus_conf

        # Combine and average confidences
        combined = {}
        for disease, conf in confidence_scores.items():
            # Average confidence if both models contributed
            if disease in ['AMD', 'DR', 'MH', 'CSR'] and conf > 1:
                combined[disease] = conf / 2
            else:
                combined[disease] = conf

        if not combined:
            return jsonify({'error': 'No valid predictions from uploaded images.'}), 500

        final_prediction = max(combined, key=combined.get)
        final_confidence = combined[final_prediction]
        used_models = ['OCT', 'Fundus']

        response = {
            'prediction': final_prediction,
            'confidence': f"{final_confidence*100:.2f}%",
            'used_models': used_models,
            'explanation': disease_explanations.get(final_prediction, "No explanation available"),
            'image_urls': {
                'oct': uploaded_files['oct'],
                'fundus': uploaded_files['fundus']
            }
        }

    except Exception as e:
        response = {'error': str(e)}
    finally:
        # Cleanup uploaded files
        for path in uploaded_files.values():
            if path and os.path.exists(path):
                os.remove(path)

    return jsonify(response)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
