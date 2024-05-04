import cv2
import threading
import queue

# Define the expensive process function
def expensive_process(frame):
    # Simulate an expensive process (e.g., blurring the frame)
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    return blurred_frame


# Function to read frames from the video stream
def read_frames(cap, frame_queue, frame_skip):
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame_queue.put(frame)
        frame_count += 1


# Function to process frames
def process_frames(frame_queue):
    while True:
        frame = frame_queue.get()
        processed_frame = expensive_process(frame)

        cv2.imshow("Processed Frame", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    cap = cv2.VideoCapture(0)  # or 0 for webcam
    if not cap.isOpened():
        print("Error: Unable to open video source")
        return

    frame_queue = queue.Queue()
    frame_skip = 5  # Process every 5th frame

    # Start threads for reading and processing frames
    read_thread = threading.Thread(
        target=read_frames, args=(cap, frame_queue, frame_skip)
    )
    process_thread = threading.Thread(target=process_frames, args=(frame_queue))

    read_thread.start()
    process_thread.start()

    read_thread.join()  # Wait for the read thread to finish (which it never does in this case)
    process_thread.join()  # Wait for the processing thread to finish

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
