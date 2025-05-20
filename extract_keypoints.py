import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize MediaPipe Pose (once, globally)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoint_sequences(video_path, annotation_csv, output_dir, sequence_length=30):
    """
    Extract 30-frame keypoint sequences around annotated frames.
    """
    # Check if annotation CSV exists
    if not os.path.exists(annotation_csv):
        print(f"Error: Annotation file {annotation_csv} not found")
        return [], []

    # Read annotations
    try:
        df = pd.read_csv(annotation_csv)
    except Exception as e:
        print(f"Error reading {annotation_csv}: {e}")
        return [], []

    actions = ['forehand-gs', 'backhand-gs', 'forehand-volley', 'backhand-volley']
    label_map = {label: idx for idx, label in enumerate(actions)}

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        cap.release()
        return [], []

    # Get video dimensions to address NORM_RECT warning
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    X = []  # Sequences
    y = []  # Labels
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    half_sequence = sequence_length // 2

    for _, row in df.iterrows():
        shot = row['Shot']
        if shot not in actions:
            print(f"Warning: Unknown shot {shot} at frame {row['FrameId']}")
            continue

        # Center sequence around annotated frame
        start_frame = max(0, row['FrameId'] - half_sequence)
        end_frame = start_frame + sequence_length
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        keypoints_sequence = []
        frame_count = start_frame

        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Video ended early at frame {frame_count}")
                break

            # Resize frame to provide IMAGE_DIMENSIONS for NORM_RECT
            frame = cv2.resize(frame, (frame_width, frame_height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with MediaPipe
            try:
                results = pose.process(frame_rgb)
                keypoints = []
                if results.pose_landmarks:
                    for landmark in results.pose_landmarks.landmark:
                        keypoints.extend([landmark.x, landmark.y, landmark.visibility])
                else:
                    keypoints = [0] * 99  # Placeholder for missing keypoints
                keypoints_sequence.append(keypoints)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                keypoints_sequence.append([0] * 99)  # Fallback
                continue

            frame_count += 1

        # Ensure sequence is complete
        if len(keypoints_sequence) == sequence_length:
            X.append(np.array(keypoints_sequence))
            y.append(label_map[shot])
        else:
            print(f"Warning: Incomplete sequence at frame {row['FrameId']} (length {len(keypoints_sequence)})")

    cap.release()

    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = Path(video_path).stem
    np.save(os.path.join(output_dir, f'X_{output_prefix}.npy'), np.array(X))
    np.save(os.path.join(output_dir, f'y_{output_prefix}.npy'), np.array(y))
    print(f"Saved {len(X)} sequences to {output_dir}/X_{output_prefix}.npy")

    return X, y

# Process multiple videos
video_dir = 'videos'
annotation_dir = 'annotations'
output_dir = 'dataset'
os.makedirs(output_dir, exist_ok=True)

all_X = []
all_y = []

for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_dir, video_file)
        annotation_file = os.path.join(annotation_dir, f"annotation_{Path(video_file).stem}.csv")
        if os.path.exists(annotation_file):
            X, y = extract_keypoint_sequences(video_path, annotation_file, output_dir)
            all_X.extend(X)
            all_y.extend(y)
        else:
            print(f"Warning: Annotation file {annotation_file} not found")

# Combine all sequences
if all_X:
    np.save(os.path.join(output_dir, 'X.npy'), np.array(all_X))
    np.save(os.path.join(output_dir, 'y.npy'), np.array(all_y))
    print(f"Combined dataset: {len(all_X)} sequences saved to {output_dir}/X.npy")
else:
    print("No sequences generated.")

# Close MediaPipe Pose
pose.close()