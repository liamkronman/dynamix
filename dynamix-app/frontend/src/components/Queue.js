import React, { useEffect, useState } from "react";
import "../styles/Queue.css";

function SongCard() {
  return (
    <div class="song-card">
      <div class="song-details">
        <h2 class="song-title">Song Title</h2>
        <p class="artist">Artist Name</p>
        <p class="album">Album Name</p>
      </div>
      <div class="shine"></div>
    </div>
  );
}

function Queue() {
  return (
    <div className="queue-container">
      <h3>Queue</h3>
      <div className="Queue">
        <SongCard />
        {/* <SongCard />
        <SongCard /> */}
      </div>
    </div>
  );
}

export default Queue;
