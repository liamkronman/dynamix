// Frontend.jsx
import React, { useState, useEffect } from "react";
import Webcam from "react-webcam";
import "../styles/Stream.css";

const Stream = () => {
  const [boundingBoxes, setBoundingBoxes] = useState([]);
  const webcamRef = React.useRef(null);

  useEffect(() => {
    const interval = setInterval(() => {
      captureFrame();
    }, 2000); // Adjust the interval as needed

    return () => clearInterval(interval);
  }, []);

  const captureFrame = async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      try {
        const response = await fetch("http://localhost:8000/process_frame", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ imageSrc }),
        });
        const data = await response.json();
        setBoundingBoxes(data.boundingBoxes);
      } catch (error) {
        console.error("Error sending frame to backend:", error);
      }
    }
  };

  return (
    <div className="webcam-container">
      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        className="webcam"
      />
      {boundingBoxes.map((box, index) => (
        <>
          <div
            key={index}
            className="bounding-box"
            style={{
              top: `${box.top}px`,
              left: `${box.left}px`,
              width: `${box.width}px`,
              height: `${box.height}px`,
            }}
          />
          {box.top_emotions &&
            box.top_emotions.map((e, i) => {
              return (
                <div
                  className="bounding-box-label"
                  style={{
                    top: `${box.top - i * 20}px`,
                    left: `${box.left}px`,
                    width: `${box.width * 2}px`,
                    height: `${box.height}px`,
                  }}
                >
                  <p>
                    {e.name}: {e.score}
                  </p>
                </div>
              );
            })}
        </>
      ))}
    </div>
  );
};

export default Stream;
