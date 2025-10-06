from facial_recognition.face_recognition_lib import find_face_duplicates
import os

if __name__ == "__main__":
    base_folder = "/Users/manfredatokwamenaamanfu/Desktop/vision_ai/data_set/images"
    print("Running Vision AI from:", os.getcwd())
    print("Checking for images in:", base_folder)
    find_face_duplicates(base_folder)
