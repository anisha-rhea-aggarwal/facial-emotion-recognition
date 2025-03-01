import json
import base64
import numpy as np
import cv2
from fer import FER
from fastapi import FastAPI, WebSocket

app = FastAPI()
detector = FER()

# ✅ Add a root HTTP GET route (to prevent 404 Not Found)
@app.get("/")
async def root():
    return {"message": "FastAPI WebSocket server is running!"}

# ✅ WebSocket Route
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_text()
            payload = json.loads(payload)

            image_base64 = payload['data']['image'].split(',')[1]
            image_data = base64.b64decode(image_base64)

            # ✅ Use np.frombuffer (instead of deprecated np.fromstring)
            image_np = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            # Detect Emotion
            prediction = detector.detect_emotions(image)
            
            if prediction:
                response = {
                    "predictions": prediction[0]['emotions'],
                    "emotion": max(prediction[0]['emotions'], key=prediction[0]['emotions'].get)
                }
            else:
                response = {"error": "No face detected"}

            await websocket.send_json(response)

    except Exception as e:
        print(f"WebSocket error: {e}")  # ✅ Debugging
        await websocket.close()
