import modal

app = modal.App("whisper-realtime-api")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate",
        "fastapi",
        "librosa",
        "soundfile",
        "numpy",
        "python-multipart",
    )
)

model_volume = modal.Volume.from_name("whisper-model-cache", create_if_missing=True)

GPU = "A10G"
MODEL_NAME = "openai/whisper-large-v3"


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/model-cache": model_volume},
    scaledown_window=300,
    timeout=600,
)
class WhisperAPI:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import pipeline
        import os

        os.environ["HF_HOME"] = "/model-cache"

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            torch_dtype=torch.float16,
            device="cuda:0",
            model_kwargs={"attn_implementation": "sdpa"},
        )
        self.processor = self.pipe.tokenizer
        print(f"Model {MODEL_NAME} loaded on {GPU}", flush=True)

    @modal.asgi_app()
    def web(self):
        import io
        import time
        import numpy as np
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
        from fastapi.responses import JSONResponse

        web_app = FastAPI(title="Whisper Realtime API")

        @web_app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_NAME, "gpu": GPU}

        @web_app.post("/transcribe")
        async def transcribe_file(
            file: UploadFile = File(...),
            language: str = Form(default=None),
            initial_prompt: str = Form(default=None),
        ):
            import librosa

            audio_bytes = await file.read()
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            duration = len(audio_array) / 16000

            generate_kwargs = {"task": "transcribe"}
            if language:
                generate_kwargs["language"] = language
            if initial_prompt:
                prompt_ids = self.pipe.tokenizer.get_prompt_ids(initial_prompt, return_tensors="pt")
                generate_kwargs["prompt_ids"] = prompt_ids

            start = time.time()
            result = self.pipe(
                audio_array,
                chunk_length_s=30,
                batch_size=24,
                generate_kwargs=generate_kwargs,
                return_timestamps=True,
            )
            elapsed = time.time() - start

            return JSONResponse({
                "text": result["text"],
                "chunks": result.get("chunks", []),
                "processing_time_s": round(elapsed, 2),
                "audio_duration_s": round(duration, 2),
            })

        @web_app.websocket("/ws/transcribe")
        async def websocket_transcribe(ws: WebSocket):
            await ws.accept()

            config = await ws.receive_json()
            language = config.get("language")
            buffer_seconds = config.get("buffer_seconds", 5)
            input_sample_rate = config.get("sample_rate", 16000)
            initial_prompt = config.get("initial_prompt")

            await ws.send_json({"type": "ready", "message": "Send audio chunks as binary"})

            audio_buffer = np.array([], dtype=np.float32)
            buffer_threshold = int(16000 * buffer_seconds)

            def build_generate_kwargs():
                kwargs = {"task": "transcribe"}
                if language:
                    kwargs["language"] = language
                if initial_prompt:
                    prompt_ids = self.pipe.tokenizer.get_prompt_ids(initial_prompt, return_tensors="pt")
                    kwargs["prompt_ids"] = prompt_ids.to(self.pipe.device)
                return kwargs

            try:
                while True:
                    data = await ws.receive_bytes()
                    chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                    if input_sample_rate != 16000:
                        import librosa
                        chunk = librosa.resample(chunk, orig_sr=input_sample_rate, target_sr=16000)

                    audio_buffer = np.concatenate([audio_buffer, chunk])

                    if len(audio_buffer) >= buffer_threshold:
                        start = time.time()
                        result = self.pipe(
                            audio_buffer,
                            chunk_length_s=30,
                            batch_size=24,
                            generate_kwargs=build_generate_kwargs(),
                            return_timestamps=True,
                        )
                        elapsed = time.time() - start

                        await ws.send_json({
                            "type": "transcription",
                            "text": result["text"],
                            "chunks": result.get("chunks", []),
                            "processing_time_s": round(elapsed, 2),
                            "audio_duration_s": round(len(audio_buffer) / 16000, 2),
                        })

                        audio_buffer = np.array([], dtype=np.float32)

            except WebSocketDisconnect:
                if len(audio_buffer) > 16000:
                    result = self.pipe(
                        audio_buffer,
                        chunk_length_s=30,
                        batch_size=24,
                        generate_kwargs=build_generate_kwargs(),
                        return_timestamps=True,
                    )
                    print(f"Final transcription: {result['text']}", flush=True)

        return web_app
