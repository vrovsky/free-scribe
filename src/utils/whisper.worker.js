import { pipeline } from "@xenova/transformers";
import { MessageTypes } from "./presets";

class MyTranscriptionPipeline {
  static task = "automatic-speech-recognition";
  static model = null;
  static instance = null;

  static async getInstance(language = "en", progress_callback = null) {
    const selectedModel =
      language === "en" ? "openai/whisper-tiny.en" : "openai/whisper-tiny";

    // Force reload if model changed
    if (this.instance === null || this.model !== selectedModel) {
      this.model = selectedModel;
      this.instance = await pipeline(this.task, this.model, {
        progress_callback,
      });
    }

    return this.instance;
  }
}

self.addEventListener("message", async (event) => {
  const { type, audio, language = "en" } = event.data;
  if (type === MessageTypes.INFERENCE_REQUEST) {
    await transcribe(audio, language);
  }
});

async function transcribe(audio, language = "en") {
  sendLoadingMessage("loading");
  console.log("Transcribing with language:", language);

  let pipeline;

  try {
    pipeline = await MyTranscriptionPipeline.getInstance(
      language,
      load_model_callback
    );
  } catch (err) {
    console.log("Pipeline error:", err.message);
    return;
  }

  sendLoadingMessage("success");

  const stride_length_s = 5;

  const generationTracker = new GenerationTracker(pipeline, stride_length_s);

  const pipelineOptions = {
    top_k: 0,
    do_sample: false,
    chunk_length: 30,
    stride_length_s,
    return_timestamps: true,
    callback_function:
      generationTracker.callbackFunction.bind(generationTracker),
    chunk_callback: generationTracker.chunkCallback.bind(generationTracker),
  };

  if (language !== "en") {
    pipelineOptions.language = language;
  }

  await pipeline(audio, pipelineOptions);
  generationTracker.sendFinalResult();
}

async function load_model_callback(data) {
  const { status } = data;
  if (status === "progress") {
    const { file, progress, loaded, total } = data;
    sendDownloadingMessage(file, progress, loaded, total);
  }
}

function sendLoadingMessage(status) {
  self.postMessage({
    type: MessageTypes.LOADING,
    status,
  });
}

async function sendDownloadingMessage(file, progress, loaded, total) {
  self.postMessage({
    type: MessageTypes.DOWNLOADING,
    file,
    progress,
    loaded,
    total,
  });
}

class GenerationTracker {
  constructor(pipeline, stride_length_s) {
    this.pipeline = pipeline;
    this.stride_length_s = stride_length_s;
    this.chunks = [];
    this.time_precision = 0.01;
    this.processed_chunks = [];
    this.callbackFunctionCounter = 0;
  }

  sendFinalResult() {
    self.postMessage({ type: MessageTypes.INFERENCE_DONE });
  }

  callbackFunction(beams) {
    this.callbackFunctionCounter += 1;
    if (this.callbackFunctionCounter % 10 !== 0) {
      return;
    }

    const bestBeam = beams[0];
    let text = this.pipeline.tokenizer.decode(bestBeam.output_token_ids, {
      skip_special_tokens: true,
    });

    const result = {
      text,
      start: this.getLastChunkTimestamp(),
      end: undefined,
    };

    createPartialResultMessage(result);
  }

  chunkCallback(data) {
    this.chunks.push(data);
    const [text, { chunks }] = this.pipeline.tokenizer._decode_asr(
      this.chunks,
      {
        time_precision: this.time_precision,
        return_timestamps: true,
        force_full_sequence: false,
      }
    );

    this.processed_chunks = chunks.map((chunk, index) => {
      return this.processChunk(chunk, index);
    });

    createResultMessage(
      this.processed_chunks,
      false,
      this.getLastChunkTimestamp()
    );
  }

  getLastChunkTimestamp() {
    if (this.processed_chunks.length === 0) {
      return 0;
    }
    return this.processed_chunks[this.processed_chunks.length - 1].end || 0;
  }

  processChunk(chunk, index) {
    const { text, timestamp } = chunk;
    const [start, end] = timestamp;

    return {
      index,
      text: `${text.trim()}`,
      start: Math.round(start),
      end: Math.round(end) || Math.round(start + 0.9 * this.stride_length_s),
    };
  }
}

function createResultMessage(results, isDone, completedUntilTimestamp) {
  self.postMessage({
    type: MessageTypes.RESULT,
    results,
    isDone,
    completedUntilTimestamp,
  });
}

function createPartialResultMessage(result) {
  self.postMessage({
    type: MessageTypes.RESULT_PARTIAL,
    result,
  });
}
