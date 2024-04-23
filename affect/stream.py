import asyncio
import cv2
import base64
import requests
import websockets
import json
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


async def connect_to_websocket():
    uri = "wss://api.hume.ai/v0/stream/models"
    extra_headers = {"X-Hume-Api-Key": os.getenv("HUME_API_KEY")}

    async with websockets.connect(uri, extra_headers=extra_headers) as websocket:
        await read_frames_and_call_api(websocket, "/")


def encode_data(filepath: Path) -> str:
    with Path(filepath).open("rb") as fp:
        bytes_data = base64.b64encode(fp.read())
        encoded_data = bytes_data.decode("utf-8")
        return encoded_data


def encode_frame(frame) -> str:
    # Convert the frame to bytes
    frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()

    # Encode the bytes in base64 format
    encoded_data = base64.b64encode(frame_bytes).decode("utf-8")

    return encoded_data


async def read_frames_and_call_api(websocket, path):
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        payload = {
            "models": {"face": {"identify_faces": True}},
            "data": encode_frame(frame),
        }

        await websocket.send(json.dumps(payload))

        message = await websocket.recv()
        pred = json.loads(message)

        # Display the frame
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam
    cap.release()


asyncio.get_event_loop().run_until_complete(connect_to_websocket())
