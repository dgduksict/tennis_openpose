import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from pathlib import Path
import os

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = None
try:
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5)
    print("MediaPipe Pose initialized successfully with model_complexity=0.")
except Exception as e:
    print(f"Failed to initialize MediaPipe Pose: {e}")
    raise

# Class names
class_names = ['Forehand-GS', 'Backhand-GS', 'Forehand-Volley', 'Backhand-Volley']

def preprocess_sequence(sequence):
    """
    Preprocess a single 30-frame keypoint sequence to match training data.
    """
    # Normalize (mean=0, std=1)
    sequence = (sequence - np.mean(sequence, axis=(0, 1), keepdims=True)) / (np.std(sequence, axis=(0, 1), keepdims=True) + 1e-8)
    
    # Handle missing keypoints (replace zeros with previous frame)
    for t in range(1, sequence.shape[0]):
        if np.all(sequence[t] == 0):
            sequence[t] = sequence[t-1]
    
    return sequence

def extract_keypoints_from_video(video_path, frame_indices=None, sequence_length=30, stride=10):
    """
    Extract 30-frame keypoint sequences from a video.
    If frame_indices is provided, extract sequences around those frames.
    Otherwise, use a sliding window with given stride.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None, None, None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")

        sequences = []
        sequence_centers = []

        if frame_indices is None:
            # Sliding window approach
            frame_indices = range(sequence_length // 2, total_frames - sequence_length // 2, stride)
        else:
            # Filter valid frame indices
            frame_indices = [f for f in frame_indices if sequence_length // 2 <= f < total_frames - sequence_length // 2]
            print(f"Using {len(frame_indices)} annotated frame indices")

        for center_frame in frame_indices:
            start_frame = max(0, center_frame - sequence_length // 2)
            end_frame = start_frame + sequence_length
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            keypoints_sequence = []
            frame_count = start_frame

            while frame_count < end_frame:
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Failed to read frame {frame_count}")
                    break

                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                keypoints = []
                if results.pose_landmarks:
                    for landmark in results.pose_landmarks.landmark:
                        keypoints.extend([landmark.x, landmark.y, landmark.visibility])
                else:
                    keypoints = [0] * 99  # 33 landmarks * 3 (x, y, visibility)

                keypoints_sequence.append(keypoints)
                frame_count += 1

            if len(keypoints_sequence) == sequence_length:
                sequences.append(preprocess_sequence(np.array(keypoints_sequence)))
                sequence_centers.append(center_frame)
            else:
                print(f"Warning: Incomplete sequence at center frame {center_frame} (got {len(keypoints_sequence)} frames)")

        cap.release()
        return np.array(sequences), sequence_centers, fps

    except Exception as e:
        print(f"Error in extract_keypoints_from_video: {e}")
        cap.release()
        return None, None, None

def test_model_on_video(video_path, model_path, annotation_csv=None, output_video=None, display_duration=1.0):
    """
    Test the LSTM model on a video and optionally save output with predictions.
    Text labels are displayed for display_duration seconds.
    """
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")

        # Load annotations if provided
        frame_indices = None
        if annotation_csv and os.path.exists(annotation_csv):
            import pandas as pd
            df = pd.read_csv(annotation_csv)
            frame_indices = df['FrameId'].values
            print(f"Loaded {len(frame_indices)} annotated frames from {annotation_csv}")

        # Extract sequences
        sequences, sequence_centers, fps = extract_keypoints_from_video(video_path, frame_indices)
        if sequences is None or len(sequences) == 0:
            print("No valid sequences extracted")
            return

        # Predict
        print("Running model predictions...")
        predictions = model.predict(sequences, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        # Print predictions
        for i, (center, pred, conf) in enumerate(zip(sequence_centers, predicted_classes, confidences)):
            print(f"Frame {center}: {class_names[pred]} (Confidence: {conf:.2f})")

        # Visualize predictions on video (if output_video is specified)
        if output_video:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path} for visualization")
                return

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                print(f"Error: Could not create output video {output_video}")
                cap.release()
                return

            # Calculate display frames
            display_frames = int(fps * display_duration)
            display_ranges = [(max(0, c - display_frames // 2), c + display_frames // 2) for c in sequence_centers]

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Find the most recent prediction within display range
                label = None
                for i, (start, end) in enumerate(display_ranges):
                    if start <= frame_idx <= end:
                        label = f"{class_names[predicted_classes[i]]} ({confidences[i]:.2f})"
                        break

                # Draw keypoints
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Draw label if available
                if label:
                    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                out.write(frame)
                frame_idx += 1

            cap.release()
            out.release()
            print(f"Output video saved to {output_video}")

    except Exception as e:
        print(f"Error in test_model_on_video: {e}")
    finally:
        if pose is not None:
            pose.close()
            print("MediaPipe Pose closed.")