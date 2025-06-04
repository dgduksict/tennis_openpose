import time
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2
import os

# Import from extract_human_pose.py
from extract_human_pose import HumanPoseExtractor, RoI, draw_keypoints, draw_edges, draw_roi

physical_devices = tf.config.experimental.list_physical_devices("GPU")
print(tf.config.experimental.list_physical_devices("GPU"))

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU memory growth enabled")
else:
    print("No GPU detected, running on CPU")

print("Num GPUs Available: ", len(physical_devices))

class ShotCounter:
    """
    Tracks and counts tennis shots based on probabilities from the neural network.
    """
    MIN_FRAMES_BETWEEN_SHOTS = 60

    def __init__(self):
        self.nb_history = 30
        self.probs = np.zeros(4)
        self.nb_forehands = 0
        self.nb_backhands = 0
        self.nb_serves = 0
        self.last_shot = "neutral"
        self.frames_since_last_shot = self.MIN_FRAMES_BETWEEN_SHOTS
        self.results = []

    def update(self, probs, frame_id):
        """Update current state with shot probabilities"""
        if len(probs) == 4:
            self.probs = probs
        else:
            self.probs[0:3] = probs

        if (
            probs[0] > 0.98
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_backhands += 1
            self.last_shot = "backhand"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif (
            probs[1] > 0.98
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_forehands += 1
            self.last_shot = "forehand"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif (
            len(probs) > 3
            and probs[3] > 0.98
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_serves += 1
            self.last_shot = "serve"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})

        self.frames_since_last_shot += 1

    def display(self, frame):
        """Display shot counters on the frame"""
        cv2.putText(
            frame,
            f"Backhands = {self.nb_backhands}",
            (20, frame.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0)
            if (self.last_shot == "backhand" and self.frames_since_last_shot < 30)
            else (0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Forehands = {self.nb_forehands}",
            (20, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0)
            if (self.last_shot == "forehand" and self.frames_since_last_shot < 30)
            else (0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Serves = {self.nb_serves}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0)
            if (self.last_shot == "serve" and self.frames_since_last_shot < 30)
            else (0, 0, 255),
            thickness=2,
        )

# Define draw_probs, draw_fps, and draw_frame_id here since they are not in extract_human_pose.py
BAR_WIDTH = 30
BAR_HEIGHT = 170
MARGIN_ABOVE_BAR = 30
SPACE_BETWEEN_BARS = 55
TEXT_ORIGIN_X = 75
BAR_ORIGIN_X = 70

def draw_probs(frame, probs):
    """Draw vertical bars representing probabilities"""
    cv2.putText(
        frame,
        "S",
        (TEXT_ORIGIN_X, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.putText(
        frame,
        "B",
        (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.putText(
        frame,
        "N",
        (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS * 2, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.putText(
        frame,
        "F",
        (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS * 3, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=3,
    )
    cv2.rectangle(
        frame,
        (
            BAR_ORIGIN_X,
            int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[3]),
        ),
        (BAR_ORIGIN_X + BAR_WIDTH, BAR_HEIGHT + MARGIN_ABOVE_BAR),
        color=(0, 0, 255),
        thickness=-1,
    )
    cv2.rectangle(
        frame,
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS,
            int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[0]),
        ),
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS + BAR_WIDTH,
            BAR_HEIGHT + MARGIN_ABOVE_BAR,
        ),
        color=(0, 0, 255),
        thickness=-1,
    )
    cv2.rectangle(
        frame,
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 2,
            int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[2]),
        ),
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 2 + BAR_WIDTH,
            BAR_HEIGHT + MARGIN_ABOVE_BAR,
        ),
        color=(0, 0, 255),
        thickness=-1,
    )
    cv2.rectangle(
        frame,
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 3,
            int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[1]),
        ),
        (
            BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 3 + BAR_WIDTH,
            BAR_HEIGHT + MARGIN_ABOVE_BAR,
        ),
        color=(0, 0, 255),
        thickness=-1,
    )
    for i in range(4):
        cv2.rectangle(
            frame,
            (
                BAR_ORIGIN_X + SPACE_BETWEEN_BARS * i,
                int(MARGIN_ABOVE_BAR),
            ),
            (
                BAR_ORIGIN_X + SPACE_BETWEEN_BARS * i + BAR_WIDTH,
                BAR_HEIGHT + MARGIN_ABOVE_BAR,
            ),
            color=(255, 255, 255),
            thickness=1,
        )

    return frame

def draw_fps(frame, fps):
    """Draw fps to demonstrate performance"""
    cv2.putText(
        frame,
        f"{int(fps)} fps",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )

def draw_frame_id(frame, frame_id):
    """Used for debugging purpose"""
    cv2.putText(
        frame,
        f"Frame {frame_id}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )

class GT:
    """Ground truth to optionally assess results"""
    def __init__(self, path_to_annotation):
        self.shots = pd.read_csv(path_to_annotation)
        self.current_row_in_shots = 0
        self.nb_backhands = 0
        self.nb_forehands = 0
        self.nb_serves = 0
        self.last_shot = "neutral"

    def display(self, frame, frame_id):
        """Display ground truth shot counters"""
        if self.current_row_in_shots < len(self.shots):
            if frame_id == self.shots.iloc[self.current_row_in_shots]["FrameId"]:
                if self.shots.iloc[self.current_row_in_shots]["Shot"] == "backhand":
                    self.nb_backhands += 1
                elif self.shots.iloc[self.current_row_in_shots]["Shot"] == "forehand":
                    self.nb_forehands += 1
                elif self.shots.iloc[self.current_row_in_shots]["Shot"] == "serve":
                    self.nb_serves += 1
                self.last_shot = self.shots.iloc[self.current_row_in_shots]["Shot"]
                self.current_row_in_shots += 1

        cv2.putText(
            frame,
            f"Backhands = {self.nb_backhands}",
            (frame.shape[1] - 300, frame.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if self.last_shot != "backhand" else (0, 255, 0),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Forehands = {self.nb_forehands}",
            (frame.shape[1] - 300, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if self.last_shot != "forehand" else (0, 255, 0),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Serves = {self.nb_serves}",
            (frame.shape[1] - 300, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if self.last_shot != "serve" else (0, 255, 0),
            thickness=2,
        )

def compute_recall_precision(gt, shots):
    """
    Assess results against ground truth for recall and precision
    """
    gt_numpy = gt.to_numpy()
    nb_match = 0
    nb_misses = 0
    nb_fp = 0
    fp_backhands = 0
    fp_forehands = 0
    fp_serves = 0
    for gt_shot in gt_numpy:
        found_match = False
        for shot in shots:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if found_match:
            nb_match += 1
        else:
            nb_misses += 1

    for shot in shots:
        found_match = False
        for gt_shot in gt_numpy:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if not found_match:
            nb_fp += 1
            if shot["Shot"] == "backhand":
                fp_backhands += 1
            elif shot["Shot"] == "forehand":
                fp_forehands += 1
            elif shot["Shot"] == "serve":
                fp_serves += 1

    precision = nb_match / (nb_match + nb_fp) if (nb_match + nb_fp) > 0 else 0
    recall = nb_match / (nb_match + nb_misses) if (nb_match + nb_misses) > 0 else 0

    print(f"Recall {recall*100:.1f}%")
    print(f"Precision {precision*100:.1f}%")
    print(f"FP: backhands = {fp_backhands}, forehands = {fp_forehands}, serves = {fp_serves}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Track tennis player and display shot probabilities using webcam")
    parser.add_argument("model", help="Path to the RNN model file (e.g., tennis_rnn.h5)")
    parser.add_argument("--evaluate", help="Path to annotation file for ground truth evaluation")
    parser.add_argument("--output", default="output_webcam.mp4", help="Output video path")
    parser.add_argument(
        "--left-handed",
        action="store_const",
        const=True,
        default=False,
        help="If player is left-handed",
    )
    parser.add_argument(
        "--debug",
        action="store_const",
        const=True,
        default=False,
        help="Show subframe (RoI)",
    )
    args = parser.parse_args()

    shot_counter = ShotCounter()

    if args.evaluate is not None:
        gt = GT(args.evaluate)

    m1 = keras.models.load_model(args.model)

    # Use webcam (index 0) instead of video file
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Get video properties
    fps = 30  # Default for most webcams; adjust if needed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam feed: {width}x{height} @ {fps} FPS")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        cap.release()
        exit()

    human_pose_extractor = HumanPoseExtractor(frame.shape)

    NB_IMAGES = 30
    FRAME_ID = 0
    features_pool = []
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        FRAME_ID += 1

        human_pose_extractor.extract(frame)

        # Discard non-significant points/edges
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)

        if args.left_handed:
            features[:, 1] = 1 - features[:, 1]

        features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)
        features_pool.append(features)

        if len(features_pool) == NB_IMAGES:
            features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26)
            assert features_seq.shape == (1, 30, 26)
            probs = m1.__call__(features_seq)[0]
            shot_counter.update(probs, FRAME_ID)
            features_pool = features_pool[1:]

        draw_probs(frame, shot_counter.probs)
        shot_counter.display(frame)

        if args.evaluate is not None:
            gt.display(frame, FRAME_ID)

        fps_current = 1 / (time.time() - prev_time)
        prev_time = time.time()
        draw_fps(frame, fps_current)
        draw_frame_id(frame, FRAME_ID)

        # Display results on original frame
        human_pose_extractor.draw_results_frame(frame)
        if (
            shot_counter.frames_since_last_shot < 30
            and shot_counter.last_shot != "neutral"
        ):
            human_pose_extractor.roi.draw_shot(frame, shot_counter.last_shot)

        # Draw subframe (RoI) if debug mode is enabled
        if args.debug:
            subframe = human_pose_extractor.draw_results_subframe()
            cv2.imshow("Subframe", subframe)

        # Write frame to output video
        out.write(frame)

        # Display the frame in a window
        cv2.imshow("Webcam Feed", frame)

        human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

        # Optional: Display progress
        if FRAME_ID % 100 == 0:
            print(f"Processed {FRAME_ID} frames...")

        # Break the loop if 'Esc' key is pressed
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved to: {args.output}")
    print(shot_counter.results)

    if args.evaluate is not None:
        compute_recall_precision(gt.shots, shot_counter.results)