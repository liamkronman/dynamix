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
      ))}
    </div>
  );
};

export default Stream;
