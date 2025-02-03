import json
import os
from distutils.dir_util import copy_tree
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps, ImageEnhance
import random

base_path = r'D:\DeepFake Detection\dfdc_train_part_48'
dataset_path = './prepared_dataset/'
print('Creating Directory: ' + dataset_path)
os.makedirs(dataset_path, exist_ok=True)

tmp_fake_path = './tmp_fake_faces'
print('Creating Directory: ' + tmp_fake_path)
os.makedirs(tmp_fake_path, exist_ok=True)

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

def augment_image(image_path, augmentations=5):
    img = Image.open(image_path)
    augmented_images = []
    for _ in range(augmentations):
        augmented_img = img
        # Apply random flip
        if random.choice([True, False]):
            augmented_img = ImageOps.mirror(augmented_img)
        # Apply random rotation
        angle = random.randint(-30, 30)
        augmented_img = augmented_img.rotate(angle)
        # Apply random color enhancement
        enhancer = ImageEnhance.Color(augmented_img)
        augmented_img = enhancer.enhance(random.uniform(0.8, 1.2))
        augmented_images.append(augmented_img)
    return augmented_images

# Add error handling for metadata file access
metadata_path = os.path.join(base_path, 'metadata.json')
print(f"Attempting to load metadata from: {metadata_path}")
try:
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at: {metadata_path}")
        print(f"Directory exists: {os.path.exists(base_path)}")
        print("Directory contents:")
        for item in os.listdir(base_path):
            print(f"- {item}")
        raise FileNotFoundError(f"metadata.json not found in {base_path}")
    
    with open(metadata_path, 'r') as metadata_json:
        metadata = json.load(metadata_json)
        print(f"Successfully loaded metadata with {len(metadata)} entries")

except Exception as e:
    print(f"Error accessing metadata file: {str(e)}")
    raise

real_path = os.path.join(dataset_path, 'real')
print('Creating Directory: ' + real_path)
os.makedirs(real_path, exist_ok=True)

fake_path = os.path.join(dataset_path, 'fake')
print('Creating Directory: ' + fake_path)
os.makedirs(fake_path, exist_ok=True)

for filename in metadata.keys():
    print(filename)
    print(metadata[filename]['label'])
    tmp_path = os.path.join(os.path.join(base_path, get_filename_only(filename)), 'faces')
    print(tmp_path)
    if os.path.exists(tmp_path):
        if metadata[filename]['label'] == 'REAL':    
            print('Copying to :' + real_path)
            copy_tree(tmp_path, real_path)
        elif metadata[filename]['label'] == 'FAKE':
            print('Copying to :' + tmp_fake_path)
            copy_tree(tmp_path, tmp_fake_path)
        else:
            print('Ignored..')

all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
print('Total Number of Real faces: ', len(all_real_faces))

all_fake_faces = [f for f in os.listdir(tmp_fake_path) if os.path.isfile(os.path.join(tmp_fake_path, f))]
print('Total Number of Fake faces: ', len(all_fake_faces))

# Augment data to balance real and fake faces
if len(all_real_faces) < len(all_fake_faces):
    print("\nStarting Real Face Augmentation:")
    print(f"Original Real Faces Count: {len(all_real_faces)}")
    print(f"Target Count (Fake Faces): {len(all_fake_faces)}")
    augment_count = len(all_fake_faces) - len(all_real_faces)
    print(f'Augmenting real faces to match fake faces count...')
    
    current_real_count = len(all_real_faces)
    real_faces_to_augment = np.random.choice(all_real_faces, augment_count, replace=True)
    
    for i, fname in enumerate(real_faces_to_augment):
        if current_real_count >= len(all_fake_faces):
            print(f"Reached balanced count. Stopping augmentation.")
            break
            
        src = os.path.join(real_path, fname)
        augmented_images = augment_image(src, augmentations=1)  # Create only 1 augmented version at a time
        
        for aug_img in augmented_images:
            if current_real_count >= len(all_fake_faces):
                break
            aug_img.save(os.path.join(real_path, f'aug_{i}_{fname}'))
            current_real_count += 1
            
        if i % 1000 == 0:  # Print progress every 1000 images
            print(f"Current real faces count: {current_real_count}")
    
    final_real_count = len([f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))])
    print(f"Final Real Faces Count after Augmentation: {final_real_count}")

elif len(all_fake_faces) < len(all_real_faces):
    print("\nStarting Fake Face Augmentation:")
    print(f"Original Fake Faces Count: {len(all_fake_faces)}")
    print(f"Target Count (Real Faces): {len(all_real_faces)}")
    augment_count = len(all_real_faces) - len(all_fake_faces)
    print(f'Augmenting fake faces to match real faces count...')
    
    current_fake_count = len(all_fake_faces)
    fake_faces_to_augment = np.random.choice(all_fake_faces, augment_count, replace=True)
    
    for i, fname in enumerate(fake_faces_to_augment):
        if current_fake_count >= len(all_real_faces):
            print(f"Reached balanced count. Stopping augmentation.")
            break
            
        src = os.path.join(tmp_fake_path, fname)
        augmented_images = augment_image(src, augmentations=1)  # Create only 1 augmented version at a time
        
        for aug_img in augmented_images:
            if current_fake_count >= len(all_real_faces):
                break
            aug_img.save(os.path.join(tmp_fake_path, f'aug_{i}_{fname}'))
            current_fake_count += 1
            
        if i % 1000 == 0:  # Print progress every 1000 images
            print(f"Current fake faces count: {current_fake_count}")
    
    final_fake_count = len([f for f in os.listdir(tmp_fake_path) if os.path.isfile(os.path.join(tmp_fake_path, f))])
    print(f"Final Fake Faces Count after Augmentation: {final_fake_count}")

all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
all_fake_faces = [f for f in os.listdir(tmp_fake_path) if os.path.isfile(os.path.join(tmp_fake_path, f))]

# Match the number of fake faces to the number of real faces
num_faces_to_sample = min(len(all_real_faces), len(all_fake_faces))
random_faces = np.random.choice(all_fake_faces, num_faces_to_sample, replace=False)

for fname in random_faces:
    src = os.path.join(tmp_fake_path, fname)
    dst = os.path.join(fake_path, fname)
    shutil.copyfile(src, dst)

print('Down-sampling Done!')

# Split into Train/ Val/ Test folders
def split_and_copy(files, source_path, train_path, val_path, test_path):
    train_files, temp_files = train_test_split(files, test_size=0.2, random_state=1377)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=1377)

    for file in train_files:
        shutil.copy(os.path.join(source_path, file), os.path.join(train_path, file))
    for file in val_files:
        shutil.copy(os.path.join(source_path, file), os.path.join(val_path, file))
    for file in test_files:
        shutil.copy(os.path.join(source_path, file), os.path.join(test_path, file))

split_base_path = 'split_dataset'
os.makedirs(split_base_path, exist_ok=True)

# Create directories for real and fake faces
for category in ['real', 'fake']:
    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(split_base_path, folder, category), exist_ok=True)

split_and_copy(all_real_faces[:num_faces_to_sample], real_path, os.path.join(split_base_path, 'train', 'real'), os.path.join(split_base_path, 'val', 'real'), os.path.join(split_base_path, 'test', 'real'))
split_and_copy(random_faces, fake_path, os.path.join(split_base_path, 'train', 'fake'), os.path.join(split_base_path, 'val', 'fake'), os.path.join(split_base_path, 'test', 'fake'))

print('Train/ Val/ Test Split Done!')