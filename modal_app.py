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
        "anthropic",
    )
)

model_volume = modal.Volume.from_name("whisper-model-cache", create_if_missing=True)

GPU = "A10G"
MODEL_NAME = "openai/whisper-large-v3"

# Hallucination filter: common phrases Whisper generates on silence/noise
HALLUCINATION_PATTERNS = [
    "ご視聴ありがとうございました",
    "ご協力ありがとうございました",
    "ありがとうございました",
    "チャンネル登録",
    "お疲れ様でした",
    "よろしくお願いします",
    "おやすみなさい",
    "では、また",
    "字幕",
    "Subtitles",
    "Thank you",
    "Thanks for watching",
]

LLM_CORRECTION_PROMPT = """\
あなたは医療文書の校正者です。音声認識（Whisper）で文字起こしされたテキストを受け取り、誤認識を修正してください。

## ルール
- 医療用語の誤認識を正しい用語に修正してください（例: 「代替歯頭筋」→「大腿四頭筋」、「知覚どんま」→「知覚鈍麻」）
- 文の意味が通るように句読点や区切りを整えてください
- 元の発言の意味を変えないでください。言い回しの変更や要約はしないでください
- 明らかな誤認識だけを修正し、正しく認識されている部分はそのまま残してください
- 修正後のテキストのみを出力してください。説明や注釈は不要です"""


def clean_transcription(text, initial_prompt=None):
    """Remove hallucinated phrases and prompt leakage from transcription."""
    if not text or not text.strip():
        return ""

    cleaned = text.strip()

    # Remove prompt text that leaked into the output
    if initial_prompt:
        for sentence in initial_prompt.replace("。", "\n").replace("、", "\n").split("\n"):
            sentence = sentence.strip()
            if len(sentence) > 5 and sentence in cleaned:
                cleaned = cleaned.replace(sentence, "")

    # Remove known hallucination patterns
    for pattern in HALLUCINATION_PATTERNS:
        cleaned = cleaned.replace(pattern, "")

    return cleaned.strip()


def correct_with_llm(text, context=None):
    """Use Claude to correct medical terminology in transcription."""
    import anthropic
    import os

    if not text or not text.strip():
        return text

    client = anthropic.Anthropic()

    user_message = ""
    if context:
        user_message += f"## 録音の前提\n{context}\n\n"
    user_message += f"## 文字起こしテキスト\n{text}"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=LLM_CORRECTION_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text.strip()


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/model-cache": model_volume},
    secrets=[modal.Secret.from_name("anthropic-api-key")],
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
        print(f"Model {MODEL_NAME} loaded on {GPU}", flush=True)

    def _build_generate_kwargs(self, language=None, initial_prompt=None):
        kwargs = {"task": "transcribe"}
        if language:
            kwargs["language"] = language
        if initial_prompt:
            prompt_ids = self.pipe.tokenizer.get_prompt_ids(initial_prompt, return_tensors="pt")
            kwargs["prompt_ids"] = prompt_ids.to(self.pipe.device)
        return kwargs

    def _transcribe(self, audio, generate_kwargs, initial_prompt=None, use_llm=False, context=None):
        import time

        start = time.time()
        result = self.pipe(
            audio,
            chunk_length_s=30,
            batch_size=24,
            generate_kwargs=generate_kwargs,
            return_timestamps=True,
        )
        whisper_elapsed = time.time() - start

        # Clean up the transcription
        cleaned_text = clean_transcription(result["text"], initial_prompt)
        cleaned_chunks = []
        for chunk in result.get("chunks", []):
            cleaned = clean_transcription(chunk["text"], initial_prompt)
            if cleaned:
                cleaned_chunks.append({"timestamp": chunk["timestamp"], "text": cleaned})

        # LLM correction
        corrected_text = cleaned_text
        llm_elapsed = 0
        if use_llm and cleaned_text:
            start = time.time()
            corrected_text = correct_with_llm(cleaned_text, context)
            llm_elapsed = time.time() - start

        return {
            "raw_text": cleaned_text,
            "text": corrected_text,
            "chunks": cleaned_chunks,
            "whisper_time_s": round(whisper_elapsed, 2),
            "llm_time_s": round(llm_elapsed, 2),
            "processing_time_s": round(whisper_elapsed + llm_elapsed, 2),
        }

    @modal.asgi_app()
    def web(self):
        import io
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
            use_llm: bool = Form(default=False),
            context: str = Form(default=None),
        ):
            import librosa

            audio_bytes = await file.read()
            audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            duration = len(audio_array) / 16000

            generate_kwargs = self._build_generate_kwargs(language, initial_prompt)
            result = self._transcribe(
                audio_array, generate_kwargs, initial_prompt,
                use_llm=use_llm, context=context,
            )

            return JSONResponse({
                "text": result["text"],
                "raw_text": result["raw_text"],
                "chunks": result["chunks"],
                "processing_time_s": result["processing_time_s"],
                "whisper_time_s": result["whisper_time_s"],
                "llm_time_s": result["llm_time_s"],
                "audio_duration_s": round(duration, 2),
            })

        @web_app.websocket("/ws/transcribe")
        async def websocket_transcribe(ws: WebSocket):
            await ws.accept()

            config = await ws.receive_json()
            language = config.get("language")
            buffer_seconds = config.get("buffer_seconds", 10)
            input_sample_rate = config.get("sample_rate", 16000)
            initial_prompt = config.get("initial_prompt")
            use_llm = config.get("use_llm", False)
            context = config.get("context")

            await ws.send_json({"type": "ready", "message": "Send audio chunks as binary"})

            audio_buffer = np.array([], dtype=np.float32)
            buffer_threshold = int(16000 * buffer_seconds)
            generate_kwargs = self._build_generate_kwargs(language, initial_prompt)

            try:
                while True:
                    data = await ws.receive_bytes()
                    chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                    if input_sample_rate != 16000:
                        import librosa
                        chunk = librosa.resample(chunk, orig_sr=input_sample_rate, target_sr=16000)

                    audio_buffer = np.concatenate([audio_buffer, chunk])

                    if len(audio_buffer) >= buffer_threshold:
                        result = self._transcribe(
                            audio_buffer, generate_kwargs, initial_prompt,
                            use_llm=use_llm, context=context,
                        )

                        if result["text"]:
                            await ws.send_json({
                                "type": "transcription",
                                "text": result["text"],
                                "raw_text": result["raw_text"],
                                "chunks": result["chunks"],
                                "processing_time_s": result["processing_time_s"],
                                "whisper_time_s": result["whisper_time_s"],
                                "llm_time_s": result["llm_time_s"],
                                "audio_duration_s": round(len(audio_buffer) / 16000, 2),
                            })

                        audio_buffer = np.array([], dtype=np.float32)

            except WebSocketDisconnect:
                if len(audio_buffer) > 16000:
                    result = self._transcribe(
                        audio_buffer, generate_kwargs, initial_prompt,
                        use_llm=use_llm, context=context,
                    )
                    if result["text"]:
                        print(f"Final transcription: {result['text']}", flush=True)

        return web_app
