import os
from PIL import Image
import numpy as np

folder = "/Users/manfredatokwamenaamanfu/Desktop/vision_ai/images"

images = []

for filename in os.listdir(folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        images.append(np.array(img))

        
print(f"Loaded {len(images)} images into the array.")

import imagehash
from PIL import Image
import numpy as np

hashes = []
duplicates = []

for i, arr in enumerate(images):
    pil_img = Image.fromarray(arr)
    hash_val = imagehash.average_hash(pil_img)
    hashes.append((i, hash_val))

    
for i in range(len(hashes)):
    for j in range(i + 1, len(hashes)):
        idx1, hash1 = hashes[i]
        idx2, hash2 = hashes[j]

        diff = hash1 - hash2

        if diff == 0:
          print(f"Duplicate found: Image {idx1} and Image {idx2}")
          duplicates.append((idx1, idx2))
        elif diff <= 5:
            print(f"Near-duplicate found: Image {idx1} and Image {idx2} (diff = {diff})")

print("\n Duplicate search finished")
print(f"Total duplicates found: {len(duplicates)}")
print("imagehash is very fast but only good fot duplicate images")