import cv2
from mtcnn import MTCNN
import sys, os.path
import json
from keras import backend as K
import tensorflow as tf

print(tf.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

base_path = 'dfdc_train_part_48'

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    metadata = json.load(metadata_json)
    print(f'Total videos to process: {len(metadata)}')

# Initialize counters for progress tracking
videos_processed = 0
frames_processed = 0
total_videos = len(metadata)
total_frames = total_videos * 10  # Since you are processing 10 frames per video

for filename in metadata.keys():
    tmp_path = os.path.join(base_path, get_filename_only(filename))
    faces_path = os.path.join(tmp_path, 'faces')
    
    # Check if the video has already been processed by counting processed frames
    processed_frames = len([f for f in os.listdir(faces_path) if os.path.isfile(os.path.join(faces_path, f))]) if os.path.exists(faces_path) else 0
    
    if processed_frames >= 10:
        # If all 10 frames have been processed, count them and skip to the next video
        videos_processed += 1
        frames_processed += 10
        print(f'Skipping already processed video: {filename}')
        print(f'Videos processed: {videos_processed}/{total_videos} ({(videos_processed / total_videos) * 100:.2f}%)')
        print(f'Frames processed: {frames_processed}/{total_frames} ({(frames_processed / total_frames) * 100:.2f}%)')
        continue
    
    print(f'Processing Directory: {tmp_path}')
    
    if not os.path.exists(tmp_path):
        print(f"Directory does not exist: {tmp_path}")
        continue
    
    frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))][:10]  # Only consider first 10 frames
    print(f'Creating Directory: {faces_path}')
    os.makedirs(faces_path, exist_ok=True)
    print('Cropping Faces from Images...')
    
    for frame in frame_images:
        face_filename = os.path.join(faces_path, get_filename_only(frame) + '-00.png')
        if os.path.exists(face_filename):
            processed_frames += 1
            print(f"Skipping already processed image: {frame}")
            continue
        
        print(f'Processing {frame}')
        detector = MTCNN()
        image_path = os.path.join(tmp_path, frame)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image_rgb)
        faces_found = len(results)  # Count the number of faces found in the frame
        
        print(f'Faces found in {frame}: {faces_found}')
        
        count = 0
        
        for result in results:
            bounding_box = result['box']
            confidence = result['confidence']
            
            if len(results) < 2 or confidence > 0.95:
                # Define the margins based on the bounding box size
                margin_x = bounding_box[2] * 0.3  # 30% as the margin
                margin_y = bounding_box[3] * 0.3  # 30% as the margin
                    
                # Calculate the coordinates for cropping, ensuring they stay within image bounds
                x1 = max(0, int(bounding_box[0] - margin_x))
                x2 = min(image.shape[1], int(bounding_box[0] + bounding_box[2] + margin_x))
                y1 = max(0, int(bounding_box[1] - margin_y))
                y2 = min(image.shape[0], int(bounding_box[1] + bounding_box[3] + margin_y))

                # Crop the image using the calculated coordinates
                crop_image = image[y1:y2, x1:x2]

                # Check if the cropped image is valid
                if crop_image is None or crop_image.size == 0:
                    print("Warning: Cropped image is empty, skipping this face.")
                    continue

                print(f"Cropped Image Dimensions: {crop_image.shape}")

                # Save the cropped face image
                new_filename = '{}-{:02d}.png'.format(os.path.join(faces_path, get_filename_only(frame)), count)
                count += 1
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))

            else:
                print('Skipped a face..')


        
        # Increment frames processed and show progress
        frames_processed += 1
        print(f'Frames processed: {frames_processed}/{total_frames} ({(frames_processed / total_frames) * 100:.2f}%)')
        print(f'Total faces found in this frame: {faces_found}')
    
    # Increment videos processed and show progress
    videos_processed += 1
    print(f'Videos processed: {videos_processed}/{total_videos} ({(videos_processed / total_videos) * 100:.2f}%)')
