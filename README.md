## Note:
#### Create virtual environment with python 3.7 to ensure compatibility with pretrained models
#### Packages required for this project are specified in __requirements.txt__

## How to execute?
#### In the terminal:
#### >> uvicorn app:app --reload

## Endpoints:

### POST --> 127.0.0.1:8000/predict/
#### (Gives the label of the image uploaded)
#### no parameters required
#### Request body (multipart/form-data)
#### { 'key': 'file', 'value': choose binary file (image) }

### POST --> 127.0.0.1:8000/upload/
#### (Adds the images of the person uploaded to the dataset along with label)
#### no parameters required
#### Request body (multipart/form-data)
#### { 'key': 'label', 'value': name of the person(string) }
#### { 'key': 'files', 'value': choose binary files (images) of the person }

### GET --> 127.0.0.1:8000/train/
#### (Initiates the training process of the model)
#### no parameters required
#### no Request body required

## Sources:
#### Dataset source: https://www.kaggle.com/datasets/hereisburak/pins-face-recognition
#### Trained Model Link: https://drive.google.com/file/d/12CVJ9y9EYq3I8pcAb3jg3_KmCX6k6q6w/view?usp=sharing
#### Base Model source: https://www.kaggle.com/datasets/utkarshsaxenadn/facenet-keras
