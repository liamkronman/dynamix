import logo from "./logo.svg";
import "./styles/App.css";

import Stream from "./components/Stream";
import Queue from "./components/Queue";
import AudioPlayer from "./components/AudioPlayer";

function App() {
  return (
    <>
      <div className="App">
        <Stream />
        <div className="settings-container">
          <h1 className="title">DYNAMIX</h1>
          <Queue />
          <AudioPlayer />
        </div>
      </div>
    </>
  );
}

export default App;
