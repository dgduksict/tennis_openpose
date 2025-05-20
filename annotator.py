from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import cv2
import os

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
    - RIGHT_ARROW: Forehand-GS
    - LEFT_ARROW: Backhand-GS
    - UP_ARROW: Forehand-Volley
    - DOWN_ARROW: Backhand-Volley
    - Esc: Exit
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Initialize DataFrame
    df = pd.DataFrame(columns=["Shot", "FrameId"])
    annotations = []
    frame_id = 0

    print("Annotating video. Press F, B, V, N for shots, Esc to exit.")

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Display frame with instructions
        cv2.putText(frame, "W: Forehand-Volley  A: Backhand-GS  S: Backhand-Volley  D: Forehand-GS",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Tennis Shot Annotation", frame)

        # Wait for key press
        k = cv2.waitKey(30)

        # Record annotations
        if k == D_KEY:
            annotations.append({"Shot": "forehand-gs", "FrameId": frame_id})
            print(f"Annotated Forehand-GS at frame {frame_id}")
            cv2.putText(frame, "Forehand-GS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Tennis Shot Annotation", frame)
            cv2.waitKey(100)
        elif k == A_KEY:
            annotations.append({"Shot": "backhand-gs", "FrameId": frame_id})
            print(f"Annotated Backhand-GS at frame {frame_id}")
            cv2.putText(frame, "Backhand-GS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Tennis Shot Annotation", frame)
            cv2.waitKey(100)
        elif k == W_KEY:
            annotations.append({"Shot": "forehand-volley", "FrameId": frame_id})
            print(f"Annotated Forehand-Volley at frame {frame_id}")
            cv2.putText(frame, "Forehand-Volley", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Tennis Shot Annotation", frame)
            cv2.waitKey(100)
        elif k == S_KEY:
            annotations.append({"Shot": "backhand-volley", "FrameId": frame_id})
            print(f"Annotated Backhand-Volley at frame {frame_id}")
            cv2.putText(frame, "Backhand-Volley", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Tennis Shot Annotation", frame)
            cv2.waitKey(100)
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
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser(description="Annotate a tennis video with shot types.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--output-dir", default="annotations", help="Directory to save annotation CSV")
    args = parser.parse_args()

    annotate_video(args.video, args.output_dir)