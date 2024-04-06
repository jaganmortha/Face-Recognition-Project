import os
import package
import package.model_trainer
import package.preprocess_data
import joblib
import shutil

temp_path = r'data/temp'
path = r'data/buffer'
train_path = r'data/splitted_dataset/train'
test_path = r'data/splitted_dataset/val'


def clean_buffer_and_temp():
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")

    for folder_name in os.listdir(temp_path):
        folder_path = os.path.join(temp_path, folder_name)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")

            
def add_user():

    package.preprocess_data.extract_faces(r'data/temp', path)

    package.preprocess_data.train_val_split()

    return


def create_model():


    class_labels = package.model_trainer.train_model(train_path, test_path)

    joblib.dump(class_labels, r"data\new_class_labels.joblib")

    clean_buffer_and_temp()

    return