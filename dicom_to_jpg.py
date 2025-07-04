import os
import pydicom
from PIL import Image
from tqdm import tqdm

input_dir = 'stage_2_train_images'
output_dir = 'train_preprocess'

os.makedirs(output_dir, exist_ok=True)

for filename in tqdm(os.listdir(input_dir)):
    if filename.lower().endswith('.dcm'):
        dcm_path = os.path.join(input_dir, filename)
        jpg_filename = os.path.splitext(filename)[0] + '.jpg'
        jpg_path = os.path.join(output_dir, jpg_filename)
        try:
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array
            # Normalize to 0-255 and convert to uint8
            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')
            im = Image.fromarray(img)
            im = im.convert('L')  # Convert to grayscale
            im.save(jpg_path)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")