import os
import numpy as np
from PIL import Image
from difPy import dif

images_np = []
filenames = []

temp_folder = "/Users/manfredatokwamenaamanfu/Desktop/VisionAI_temp"
os.makedirs(temp_folder, exist_ok = True)

for i, arr in enumerate(images_np):
    img = Image.fromarray(arr.astype("uint8"))
    fname = filenames[i]
    if not (fname.lower().endswith(".jpeg") or fname.lower().endswith(".png")):
        fname = fname + ".jpeg"
    img.save(os.path.join(temp_folder, fname))

search = dif(temp_folder)

print("Duplicate results:")
print(search.result)
