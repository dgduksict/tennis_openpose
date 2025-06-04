import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load the GRU model
model = tf.keras.models.load_model("models/tennismodel_gru_legacy.h5")

class ShotCounter:
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

    def get_state(self):
        return {
            "probs": self.probs.tolist(),
            "backhands": self.nb_backhands,
            "forehands": self.nb_forehands,
            "serves": self.nb_serves,
            "last_shot": self.last_shot,
            "results": self.results
        }

# In-memory shot counter instance (for simplicity; use a database for production)
shot_counter = ShotCounter()

# Request model for keypoints
class KeypointsRequest(BaseModel):
    keypoints: List[List[float]]  # List of 26-element keypoint arrays (30 frames)
    frame_id: int
    left_handed: bool = False

@app.post("/predict")
async def predict_shot(request: KeypointsRequest):
    try:
        # Process keypoints
        keypoints = np.array(request.keypoints)
        if request.left_handed:
            keypoints[:, :, 1] = 1 - keypoints[:, :, 1]  # Flip y-coordinates
        features_seq = keypoints.reshape(1, 30, 26)
        
        # Run model inference
        probs = model.predict(features_seq)[0]
        
        # Update shot counter
        shot_counter.update(probs, request.frame_id)
        
        # Return response
        return shot_counter.get_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reset")
async def reset_counter():
    global shot_counter
    shot_counter = ShotCounter()
    return {"message": "Shot counter reset"}

@app.get("/state")
async def get_state():
    return shot_counter.get_state()

@app.get("/")
async def root():
    return {"message": "Welcome to the Tennis Shot Prediction API. Use /predict to send keypoints."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)