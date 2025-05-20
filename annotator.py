from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import cv2
import os
import mediapipe as mp

#Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Key bindings for your shot classes
W_KEY = ord('w')  # Forehand-Volley
A_KEY = ord('a')  # Backhand-GS
S_KEY = ord('s')  # Backhand-Volley
D_KEY = ord('d')  # Forehand-GS
ESC_KEY = 27      # Exit

def annotate_video(video_path, output_dir):
    """
    Annotate a tennis video with shot types and save to a CSV in the specified output directory.
    Press:
    - W: Forehand-Volley
    - A: Backhand-GS
    - S: Backhand-Volley
    - D: Forehand-GS
    - Esc: Exit
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize MediaPipe Pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Initialize annotation list
    annotations = []
    frame_id = 0

    print("Annotating video. Press W, A, S, D for shots. Esc to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        # Display instructions
        cv2.putText(frame, "W: Forehand-Volley  A: Backhand-GS  S: Backhand-Volley  D: Forehand-GS",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Tennis Shot Annotation", frame)

        # Wait for key press
        k = cv2.waitKey(30)

        # Annotate based on key press
        if k == D_KEY:
            annotations.append({"Shot": "forehand-gs", "FrameId": frame_id})
            print(f"Annotated Forehand-GS at frame {frame_id}")
        elif k == A_KEY:
            annotations.append({"Shot": "backhand-gs", "FrameId": frame_id})
            print(f"Annotated Backhand-GS at frame {frame_id}")
        elif k == W_KEY:
            annotations.append({"Shot": "forehand-volley", "FrameId": frame_id})
            print(f"Annotated Forehand-Volley at frame {frame_id}")
        elif k == S_KEY:
            annotations.append({"Shot": "backhand-volley", "FrameId": frame_id})
            print(f"Annotated Backhand-Volley at frame {frame_id}")
        elif k == ESC_KEY:
            print("Annotation completed.")
            break

        frame_id += 1

    # Save annotations to CSV
    if annotations:
        df = pd.DataFrame.from_records(annotations)
        out_file = os.path.join(output_dir, f"annotation_{Path(video_path).stem}.csv")
        df.to_csv(out_file, index=False)
        print(f"Annotation file saved to {out_file}")
    else:
        print("No annotations recorded.")

    # Cleanup
    cap.release()
    pose.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser(description="Annotate a tennis video with shot types.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--output-dir", default="annotations", help="Directory to save annotation CSV")
    args = parser.parse_args()

    annotate_video(args.video, args.output_dir)
