import { useState, useRef, useEffect } from "react";
import HomePage from "./components/HomePage";
import Header from "./components/Header";
import FileDisplay from "./components/FileDisplay";
import Information from "./components/Information";
import Transcribing from "./components/Transcribing";
import { MessageTypes } from "./utils/presets";

function App() {
  const [file, setFile] = useState(null);
  const [audioStream, setAudioStream] = useState(null);
  const [output, setOutput] = useState(null);
  const [downloading, setDownloading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [finished, setFinished] = useState(false);
  const [language, setLanguage] = useState("en");

  const isAudioAvailable = file || audioStream;

  function handleAudioReset() {
    setFile(null);
    setAudioStream(null);
    setOutput(null);
    setFinished(false);
    setLoading(false);
    setDownloading(false);
  }

  const worker = useRef(null);

  useEffect(() => {
    if (!worker.current) {
      worker.current = new Worker(
        new URL("./utils/whisper.worker.js", import.meta.url),
        {
          type: "module",
        }
      );
    }

    const onMessageReceived = async (e) => {
      switch (e.data.type) {
        case MessageTypes.DOWNLOADING:
          setDownloading(true);
          console.log("DOWNLOADING");
          break;
        case MessageTypes.LOADING:
          setLoading(true);
          setDownloading(false);
          console.log("LOADING");
          break;
        case MessageTypes.RESULT:
          setOutput(e.data.results);
          console.log("Results:", e.data.results);
          break;
        case MessageTypes.INFERENCE_DONE:
          setFinished(true);
          setLoading(false);
          console.log("TRANSCRIPTION DONE");
          break;
        case MessageTypes.ERROR:
          console.error("Worker error:", e.data.error);
          setLoading(false);
          setDownloading(false);
          // might show this error to the user
          alert(`Transcription error: ${e.data.error}`);
          break;
      }
    };

    worker.current.addEventListener("message", onMessageReceived);

    return () => {
      if (worker.current) {
        worker.current.removeEventListener("message", onMessageReceived);
      }
    };
  }, []);

  async function readAudioFrom(file) {
    const sampling_rate = 16000;
    const audioCTX = new AudioContext({ sampleRate: sampling_rate });
    const response = await file.arrayBuffer();
    const decoded = await audioCTX.decodeAudioData(response);
    const audio = decoded.getChannelData(0);
    return audio;
  }

  async function handleFormSubmission() {
    if (!file && !audioStream) {
      return;
    }

    try {
      setLoading(true);
      setOutput(null);
      setFinished(false);

      let audio = await readAudioFrom(file ? file : audioStream);

      console.log("Sending transcription request with language:", language);

      worker.current.postMessage({
        type: MessageTypes.INFERENCE_REQUEST,
        audio,
        language, // This is the key - passing the language
      });
    } catch (error) {
      console.error("Error processing audio:", error);
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col max-w-[1000px] mx-auto w-full">
      <section className="min-h-screen flex flex-col">
        <Header />
        {output ? (
          <Information output={output} finished={finished} />
        ) : loading ? (
          <Transcribing />
        ) : isAudioAvailable ? (
          <FileDisplay
            handleFormSubmission={handleFormSubmission}
            handleAudioReset={handleAudioReset}
            file={file}
            audioStream={audioStream}
          />
        ) : (
          <HomePage
            setFile={setFile}
            setAudioStream={setAudioStream}
            language={language}
            setLanguage={setLanguage}
          />
        )}
      </section>
      <footer></footer>
    </div>
  );
}

export default App;
