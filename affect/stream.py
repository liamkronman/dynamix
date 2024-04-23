import asyncio
import cv2
import base64
import requests
import websockets
import json
from pathlib import Path
from dotenv import load_dotenv
import os
import numpy as np

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

        if "predictions" in pred["face"]:
            annotated_frame = draw_on_frame(frame, pred["face"]["predictions"])
        else:
            annotated_frame = frame

        # Display the frame
        cv2.imshow("frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam
    cap.release()


def draw_on_frame(frame, pred):
    # Convert frame to numpy array

    frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Add bounding boxes
    for p in pred:
        x, y, w, h = (
            int(p["bbox"]["x"]),
            int(p["bbox"]["y"]),
            int(p["bbox"]["w"]),
            int(p["bbox"]["h"]),
        )

        # Draw bounding box
        cv2.rectangle(frame_np, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Add top three emotions
        emotions = p["emotions"]
        top_three = sorted(emotions, key=lambda x: x["score"], reverse=True)[:3]
        labels = [f"{e['name']}: {round(e['score'], 2)}" for e in top_three]
        # label = "\n".join(labels)
        i = 0
        for l in labels:
            cv2.putText(
                frame_np,
                l,
                (x, y - 20 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            i += 1

    # Convert the modified numpy array back to a frame
    modified_frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    return modified_frame


asyncio.get_event_loop().run_until_complete(connect_to_websocket())
