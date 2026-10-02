import base64
import numpy as np
import cv2
import pymongo
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime

app = Flask(__name__)
CORS(app)

# 1. MongoDB Setup
client = pymongo.MongoClient("mongodb://localhost:27017/") 
db = client["FaceRecognition_Project"]
emotion_collection = db["face_emotion"]

# 2. Model & Labels Setup
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

try:
    # Jo model humne notebook se save kiya tha, use yahan load kar rahe hain
    model = load_model('emotion_model.h5')
    print("✅ Trained Model successfully loaded in app.py!")
except Exception as e:
    print(f"❌ Model load karne mein error: {e}")

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    try:
        data = request.json
        image_b64 = data['image']
        
        header, encoded = image_b64.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
        else:
            roi_gray = gray

        gray_resized = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(gray_resized)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        # AI Prediction
        predictions = model.predict(img_pixels, verbose=0)
        max_index = int(np.argmax(predictions.flatten()))
        predicted_emotion = emotion_labels[max_index]

        # MongoDB mein save karna
        live_entry = {
            "detected_emotion": predicted_emotion,
            "time": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source": "React Frontend WebApp"
        }
        emotion_collection.insert_one(live_entry)

        return jsonify({
            "status": "success",
            "emotion": predicted_emotion.upper()
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)
