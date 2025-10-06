import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import os
import numpy as np
import face_recognition  # type: ignore
from PIL import Image
from multiprocessing import Pool, cpu_count
import shutil

def encode_face(args):
    img_array, filename = args
    face_locations = face_recognition.face_locations(img_array)
    face_encs = face_recognition.face_encodings(img_array, face_locations)
    if face_encs:
        return (filename, face_encs[0])
    else:
        print(f"No face found in {filename}")
        return None

def find_face_duplicates(base_folder):
    print("Welcome to the Vision AI project!")

    output_root = os.path.dirname(os.path.dirname(base_folder))
    location = os.path.join(output_root, "output")
    os.makedirs(location, exist_ok=True)
    print("Output directory created at:", location)

    out_dirs = {
        "no_human_face": os.path.join(location, "no_human_face"),
        "same_person": os.path.join(location, "same_person"),
        "similar_faces": os.path.join(location, "similar_faces"),
        "not_the_same_person": os.path.join(location, "not_the_same_person"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    images = []
    filenames = []
    for filename in os.listdir(base_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(base_folder, filename)
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            images.append((img_array, filename))
            filenames.append(filename)

    print(f"\nImages processed: {filenames}")

    with Pool(cpu_count()) as pool:
        encodings_raw = pool.map(encode_face, images)

    encodings = [e for e in encodings_raw if e is not None]
    no_human_face_files = [filename for (img_array, filename), enc in zip(images, encodings_raw) if enc is None]

    same_person = set()
    similar_faces = set()
    not_same_person = set()
    for i in range(len(encodings)):
        for j in range(i + 1, len(encodings)):
            file1, enc1 = encodings[i]
            file2, enc2 = encodings[j]
            result_strict = face_recognition.compare_faces([enc1], enc2, tolerance=0.5)
            result_similar = face_recognition.compare_faces([enc1], enc2, tolerance=0.6)
            pair = frozenset([file1, file2])
            if result_strict[0]:
                same_person.add(pair)
            elif result_similar[0]:
                similar_faces.add(pair)
            else:
                not_same_person.add(pair)

    print(f"\nGROUP: Images with no human face detected ({len(no_human_face_files)}):")
    for f in no_human_face_files:
        print(f)

    print(f"\nGROUP: Images of the same person ({len(same_person)}):")
    for pair in same_person:
        print(" <--> ".join(pair))

    print(f"\nGROUP: Images with similar faces ({len(similar_faces)}):")
    for pair in similar_faces:
        print(" <--> ".join(pair))

    print(f"\nGROUP: Images of different people ({len(not_same_person)}):")
    for pair in not_same_person:
        print(" <--> ".join(pair))

    print("\nGrouping and sorting complete.")

    for f in no_human_face_files:
        src = os.path.join(base_folder, f)
        dst = os.path.join(out_dirs["no_human_face"], f)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    for pair in same_person:
        for f in pair:
            src = os.path.join(base_folder, f)
            dst = os.path.join(out_dirs["same_person"], f)
            if os.path.exists(src):
                shutil.copy2(src, dst)

    for pair in similar_faces:
        for f in pair:
            src = os.path.join(base_folder, f)
            dst = os.path.join(out_dirs["similar_faces"], f)
            if os.path.exists(src):
                shutil.copy2(src, dst)

    for pair in not_same_person:
        for f in pair:
            src = os.path.join(base_folder, f)
            dst = os.path.join(out_dirs["not_the_same_person"], f)
            if os.path.exists(src):
                shutil.copy2(src, dst)


def main():
    base_folder = "/Users/manfredatokwamenaamanfu/Desktop/vision_ai/data_set/images"
    find_face_duplicates(base_folder)


if __name__ == "__main__":
    main()