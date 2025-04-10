from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from main import Main

app = FastAPI()

class SubtitleRequest(BaseModel):
    frame_base64: str
    # text: str
    start: float
    end: float

@app.post("/assign-subtitle-region")
def assign_subtitle_region(data: SubtitleRequest):
    frame_data = base64.b64decode(data.frame_base64)
    np_img = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    result = Main.run_subtitle_pipeline(frame, data.start, data.end)
    return result
