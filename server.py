import io
import time
import wave
import numpy as np
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import pipeline

# --- Config ---
MODEL_NAME = "openai/whisper-large-v3"
DEVICE = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32
SAMPLE_RATE = 16000
# Buffer duration in seconds: accumulate this much audio before transcribing
BUFFER_SECONDS = 5

pipe = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    print(f"Loading model {MODEL_NAME} on {DEVICE}...", flush=True)
    start = time.time()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        torch_dtype=DTYPE,
        device=DEVICE,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    print(f"Model loaded in {time.time() - start:.1f}s", flush=True)
    yield
    pipe = None


app = FastAPI(title="Whisper Realtime API", lifespan=lifespan)


def transcribe_audio(audio_array: np.ndarray, language: str | None = None, initial_prompt: str | None = None) -> dict:
    """Transcribe a numpy audio array (16kHz mono float32)."""
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    if initial_prompt:
        prompt_ids = pipe.tokenizer.get_prompt_ids(initial_prompt, return_tensors="pt")
        generate_kwargs["prompt_ids"] = prompt_ids

    result = pipe(
        audio_array,
        chunk_length_s=30,
        batch_size=4,
        generate_kwargs=generate_kwargs,
        return_timestamps=True,
    )
    return result


# --- REST endpoint: file upload ---
@app.post("/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    initial_prompt: str = Form(default=None),
):
    audio_bytes = await file.read()

    # Use librosa to load any audio format
    import librosa
    audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)

    start = time.time()
    result = transcribe_audio(audio_array, language=language, initial_prompt=initial_prompt)
    elapsed = time.time() - start

    return JSONResponse({
        "text": result["text"],
        "chunks": result.get("chunks", []),
        "processing_time_s": round(elapsed, 2),
        "audio_duration_s": round(len(audio_array) / SAMPLE_RATE, 2),
    })


# --- WebSocket endpoint: realtime streaming ---
@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    await ws.accept()

    # Receive config message
    config = await ws.receive_json()
    language = config.get("language")
    buffer_seconds = config.get("buffer_seconds", BUFFER_SECONDS)
    input_sample_rate = config.get("sample_rate", SAMPLE_RATE)
    initial_prompt = config.get("initial_prompt")

    await ws.send_json({"type": "ready", "message": "Send audio chunks as binary"})

    audio_buffer = np.array([], dtype=np.float32)
    buffer_threshold = int(SAMPLE_RATE * buffer_seconds)

    try:
        while True:
            data = await ws.receive_bytes()

            # Decode raw PCM int16 mono -> float32
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # Resample if needed
            if input_sample_rate != SAMPLE_RATE:
                import librosa
                chunk = librosa.resample(chunk, orig_sr=input_sample_rate, target_sr=SAMPLE_RATE)

            audio_buffer = np.concatenate([audio_buffer, chunk])

            # Transcribe when buffer is full
            if len(audio_buffer) >= buffer_threshold:
                start = time.time()
                result = transcribe_audio(audio_buffer, language=language, initial_prompt=initial_prompt)
                elapsed = time.time() - start

                await ws.send_json({
                    "type": "transcription",
                    "text": result["text"],
                    "chunks": result.get("chunks", []),
                    "processing_time_s": round(elapsed, 2),
                    "audio_duration_s": round(len(audio_buffer) / SAMPLE_RATE, 2),
                })

                audio_buffer = np.array([], dtype=np.float32)

    except WebSocketDisconnect:
        # Transcribe remaining buffer on disconnect
        if len(audio_buffer) > SAMPLE_RATE:  # at least 1 second
            result = transcribe_audio(audio_buffer, language=language)
            # Client already disconnected, just log
            print(f"Final transcription: {result['text']}", flush=True)


# --- Health check ---
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
