import { pipeline } from "@xenova/transformers";
import { MessageTypes } from "./presets";

class MyTranscriptionPipeline {
  static task = "automatic-speech-recognition";
  static model = null;
  static instance = null;

  static async getInstance(language = "en", progress_callback = null) {
    const selectedModel =
      language === "en" ? "Xenova/whisper-tiny.en" : "Xenova/whisper-tiny";

    if (this.instance === null || this.model !== selectedModel) {
      this.model = selectedModel;
      console.log(`Loading model: ${selectedModel}`);

      try {
        this.instance = await pipeline(this.task, this.model, {
          progress_callback,
          dtype: "fp32",
        });
        console.log(`Model ${selectedModel} loaded successfully`);
      } catch (error) {
        console.error(`Failed to load model ${selectedModel}:`, error);
        throw error;
      }
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

  let transcriptionPipeline;

  try {
    transcriptionPipeline = await MyTranscriptionPipeline.getInstance(
      language,
      load_model_callback
    );
  } catch (err) {
    console.error("Pipeline error:", err.message);
    self.postMessage({
      type: MessageTypes.ERROR,
      error: `Failed to load transcription model: ${err.message}`,
    });
    return;
  }

  sendLoadingMessage("success");

  const stride_length_s = 5;

  try {
    const generationTracker = new GenerationTracker(
      transcriptionPipeline,
      stride_length_s
    );

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

    if (language === "ru") {
      pipelineOptions.language = "russian";
      pipelineOptions.forced_decoder_ids = null;
    }

    console.log("Starting transcription with options:", pipelineOptions);
    await transcriptionPipeline(audio, pipelineOptions);
    generationTracker.sendFinalResult();
  } catch (error) {
    console.error("Transcription error:", error);
    self.postMessage({
      type: MessageTypes.ERROR,
      error: `Transcription failed: ${error.message}`,
    });
  }
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
