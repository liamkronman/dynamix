import React, { useEffect, useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPause, faPlaneUp } from "@fortawesome/free-solid-svg-icons";
import { faPlay } from "@fortawesome/free-solid-svg-icons";
import "../styles/Queue.css";
import axios from "axios";

function AudioPlayer() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioSrc, setAudioSrc] = useState("");

  const handlePlay = async () => {
    setIsPlaying(true);
    const response = await axios.get("http://localhost:8000/toggle_stream");
  };

  const handlePause = async () => {
    setIsPlaying(false);
    const response = await axios.get("http://localhost:8000/toggle_stream");
  };

  return (
    <div className="audio-controls-container">
      <h3>Controls</h3>
      <div className="Controls">
        {isPlaying ? (
          <button className="control-button" onClick={() => handlePause()}>
            <FontAwesomeIcon icon={faPause} />
          </button>
        ) : (
          <button className="control-button" onClick={() => handlePlay(true)}>
            <FontAwesomeIcon icon={faPlay} />
          </button>
        )}
        <audio autoPlay controls>
          <source src="http://localhost:8000/stream_audio" type="audio/mpeg" />
        </audio>
      </div>
    </div>
  );
}

export default AudioPlayer;
