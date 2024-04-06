from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
import os
import numpy as np
import joblib
import package
import package.model_training
import package.preprocess_data
from typing import List
from tensorflow.keras.models import load_model

app = FastAPI()

class_labels = joblib.load(r'data\class_labels.joblib')
model = load_model(r'models/trained_model_110.h5')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    
    # Read image as a stream of bytes
    image_data = await file.read()
    face_array = package.preprocess_data.preprocess_image(image_data)

    # Predict the class
    predictions = model.predict(face_array)
    predicted_class_index = np.argmax(predictions[0])
    probability = np.max(predictions[0])

    # Return the predicted class label
    if (probability*100)>85.0:
        return {"predicted_class": class_labels[predicted_class_index], "Probability": str(probability)}
    else:
        return {"predicted_class": 0, "Probability": str(probability)}


@app.post("/upload/")
async def upload_images(label: str = Form(...), files: List[UploadFile] = File(...)):

    # Create a directory for the label if it doesn't exist
    label_dir = os.path.join('data/temp', label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Save each file in the directory
    for file in files:
        contents = await file.read()
        file_path = os.path.join(label_dir, file.filename)
        with open(file_path, 'wb') as f:
            f.write(contents)

    package.model_training.add_user()
    return {"label": label, "message": f"Saved {len(files)} images to folder {label}"}


@app.get("/train/")
async def train(background_tasks: BackgroundTasks):
       
    background_tasks.add_task(package.model_training.create_model())
    return {"Training": "Started"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

