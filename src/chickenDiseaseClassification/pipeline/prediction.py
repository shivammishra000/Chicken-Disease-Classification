import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model_path = os.path.join("artifacts", "training", "model.h5")
        self.class_names = ["Coccidiosis", "Healthy"]  

    def predict(self):
        model = load_model(self.model_path)

        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        label = self.class_names[predicted_class]

        return {
            "prediction": label,
            "confidence": confidence,
        }
