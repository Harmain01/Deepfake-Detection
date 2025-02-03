import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Parameters
img_height, img_width = 224, 224

def preprocess_frame(frame):
    # Resize frame to match model's expected sizing
    frame = cv2.resize(frame, (img_height, img_width))
    # Convert BGR to RGB (since OpenCV uses BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to array and normalize
    frame = img_to_array(frame)
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

# Find the latest model checkpoint
def find_latest_checkpoint(checkpoint_dir):
    best_model_dir = os.path.join(checkpoint_dir, 'best_model')
    if os.path.exists(best_model_dir):
        checkpoints = [os.path.join(best_model_dir, f) for f in os.listdir(best_model_dir) if f.endswith('.keras')]
        if checkpoints:
            return max(checkpoints, key=os.path.getctime)
    return None

# Load the trained model
checkpoint_dir = 'checkpoint'
model_path = find_latest_checkpoint(checkpoint_dir)

if model_path is None:
    print("No trained model found in checkpoints directory!")
    exit()

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# Open the video file
video_path = r'D:\DeepFake Detection\eg.mp4'
print(f"Attempting to open video file: {video_path}")

if not os.path.exists(video_path):
    print(f"Video file not found at: {video_path}")
else:
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

frame_count = 0
fake_count = 0
real_count = 0

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Preprocess the frame
    processed_frame = preprocess_frame(frame)
    
    # Make prediction
    prediction = model.predict(processed_frame, verbose=0)[0][0]
    
    # Classify as fake or real (threshold = 0.45)
    threshold = 0.5
    is_fake = prediction > threshold
    if is_fake:
        fake_count += 1
        label = f"FAKE ({prediction:.2f})"
        color = (0, 0, 255)  # Red for fake
    else:
        real_count += 1
        label = f"REAL ({prediction:.2f})"
        color = (0, 255, 0)  # Green for real
    
    # Draw the prediction on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Display the frame
    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate statistics
total_frames = frame_count
fake_percentage = (fake_count / total_frames) * 100
real_percentage = (real_count / total_frames) * 100

print("\nVideo Analysis Results:")
print(f"Total Frames Processed: {total_frames}")
print(f"Frames Classified as Fake: {fake_count} ({fake_percentage:.1f}%)")
print(f"Frames Classified as Real: {real_count} ({real_percentage:.1f}%)")
print(f"\nFinal Verdict: Video is {'FAKE' if fake_percentage > 50 else 'REAL'}")

# Release resources
cap.release()
cv2.destroyAllWindows()
