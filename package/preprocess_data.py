import splitfolders
import cv2
import numpy as np
from tqdm import tqdm
import os
import glob
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing import image


def preprocess_image(image_data, target_size=(160, 160)):

    # Load Face Detector
    face_cascade = cv2.CascadeClassifier(r'models\haarcascade_frontalface_default.xml')
    # Load Image
    img = Image.open(BytesIO(image_data))
    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(img) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 

    # Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, return a message
    if len(faces) == 0:
        return {"message": "No faces detected"}

    # For simplicity, let's use the first detected face
    x, y, w, h = faces[0]
    face = open_cv_image[y:y+h, x:x+w]
    # Convert face to PIL Image for preprocessing
    face_img = Image.fromarray(face, 'RGB')
    face_img = face_img.resize((160, 160))
    face_array = image.img_to_array(face_img)
    face_array = np.expand_dims(face_array, axis=0)
    face_array /= 255.0  # Scale image values
    return face_array


def train_val_split():
    
    input_folder = r'data/buffer'
    # Path to the output folder where the train/test subfolders will be created
    output_folder = r"data/splitted_dataset"

    # Split with a ratio of 0.8 for training and 0.2 for testing
    splitfolders.ratio(input_folder, output=output_folder, seed=1, ratio=(.8, .2))


def check_images():
    problematic_files = []
    for root, dirs, files in os.walk(r'data\splitted_dataset'):
        for file in tqdm(files, desc="Checking Images"):
            try:
                image = Image.open(os.path.join(root, file))
                image.verify()  # Attempt to open and verify the image
            except (IOError, SyntaxError) as e:
                print(f"Problematic Image: {os.path.join(root, file)}")
                problematic_files.append(os.path.join(root, file))

    print("Problematic files:", problematic_files)


def extract_faces(source_dir, target_dir):

    # Initialize the face detector
    face_cascade = cv2.CascadeClassifier(r'models\haarcascade_frontalface_default.xml')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over all subdirectories in the source directory
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        target_subdir_path = os.path.join(target_dir, subdir)

        # Create a corresponding subdirectory in the target directory
        if not os.path.exists(target_subdir_path):
            os.makedirs(target_subdir_path)

        # Process all image files in the current subdirectory
        for file in glob.glob(subdir_path + '/*.jpg'):  # Assuming the images are in JPG format
            img = cv2.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Save each face as a separate file
            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y+h, x:x+w]
                face_filename = f'face_{i}_{os.path.basename(file)}'
                cv2.imwrite(os.path.join(target_subdir_path, face_filename), face)

