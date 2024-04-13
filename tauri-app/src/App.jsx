import { useState } from "react";
import reactLogo from "./assets/react.svg";
import axios from "axios";
import "./App.css";

function App() {
  const [promptMsg, setPromptMsg] = useState("");
  const [name, setName] = useState("");

  // async function prompt() {
  //   // Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
  //   // setPromptMsg(await invoke("prompt", { text }));
  // }

  const prompt = (e) => {
    axios
      .post("http://127.0.0.1:5000/prompt", { name })
      .then((response) => {
        setPromptMsg(response.data.result);
      })
      .catch((error) => {
        console.error(error);
      });
  };

  return (
    <div className="container">
      <h1>mastermind</h1>

      {/* <div className="row">
        <a href="https://vitejs.dev" target="_blank">
          <img src="/vite.svg" className="logo vite" alt="Vite logo" />
        </a>
        <a href="https://tauri.app" target="_blank">
          <img src="/tauri.svg" className="logo tauri" alt="Tauri logo" />
        </a>
        <a href="https://reactjs.org" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>

      <p>Click on the Tauri, Vite, and React logos to learn more.</p> */}

      {/* <form className="row"> */}
      <input
        id="greet-input"
        onChange={(e) => setName(e.currentTarget.value)}
        placeholder="Enter a prompt..."
      />
      <button onClick={prompt}>Prompt!</button>
      {/* </form> */}

      {promptMsg && <p className="output">{promptMsg}</p>}
    </div>
  );
}

export default App;
