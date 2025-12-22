#!/usr/bin/env python3
import os, asyncio, json, base64, time, tempfile, io
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import torch, numpy as np, uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import librosa
from pydantic import BaseModel
from asyncio import Queue

# --- Moshi Streaming Imports ---
from moshi.models import loaders, MimiModel, LMModel, LMGen
import sentencepiece

# --- OpenAI Whisper API Compatible Response Models ---
class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float

class TranscriptionSegment(BaseModel):
    id: int
    seek: float
    start: float
    end: float
    text: str
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    words: Optional[list[TranscriptionWord]] = None

class TranscriptionResponse(BaseModel):
    text: str
    task: str = "transcribe"
    language: str = "en"
    duration: float
    segments: Optional[list[TranscriptionSegment]] = None

# --- Core Streaming Engine (same as before) ---
class StreamingKyutaiEngine:
    def __init__(self, device: str):
        self.device = device
        logger.info("🚀 Loading Moshi streaming model components...")
        
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
        self.mimi: MimiModel = checkpoint_info.get_mimi(device=device)
        self.text_tokenizer: sentencepiece.SentencePieceProcessor = checkpoint_info.get_text_tokenizer()
        self.lm_model: LMModel = checkpoint_info.get_moshi(device=device)
        
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.sample_rate = self.mimi.sample_rate
        self._model_loaded = True
        
        # --- Lock to protect the stateful model ---
        self.lock = asyncio.Lock()
        
        logger.info(f"🎉 Moshi streaming engine components loaded on {self.device}")

    async def transcribe_audio_file(self, audio_data: np.ndarray, sample_rate: int = None) -> tuple[str, float]:
        """Transcribe audio file and return (text, duration)"""
        async with self.lock:
            try:
                # Resample if necessary
                if sample_rate and sample_rate != self.sample_rate:
                    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
                
                duration = len(audio_data) / self.sample_rate
                
                # Create a new generator and set up the streaming context
                lm_gen = LMGen(self.lm_model, temp=0, temp_text=0, use_sampling=False)
                transcription_text = ""
                
                with self.mimi.streaming(batch_size=1), lm_gen.streaming(batch_size=1):
                    first_frame = True
                    
                    # Process audio in chunks
                    for i in range(0, len(audio_data), self.frame_size):
                        chunk = audio_data[i:i + self.frame_size]
                        if len(chunk) == self.frame_size:
                            writable_chunk = chunk.copy()
                            in_pcms = torch.from_numpy(writable_chunk).to(self.device).unsqueeze(0).unsqueeze(0)
                            
                            codes = self.mimi.encode(in_pcms)
                            
                            if first_frame:
                                lm_gen.step(codes)
                                first_frame = False

                            tokens = lm_gen.step(codes)
                            if tokens is None: 
                                continue

                            text_id = tokens[0, 0].cpu().item()
                            if text_id not in [0, 3]:
                                text_fragment = self.text_tokenizer.id_to_piece(text_id)
                                clean_fragment = text_fragment.replace("▁", " ")
                                transcription_text += clean_fragment

                return transcription_text.strip(), duration
                
            except Exception as e:
                logger.error(f"Error transcribing audio: {e}")
                return "", 0.0
            
class RealtimeStreamingSession:
    def __init__(self, engine: StreamingKyutaiEngine):
        self.engine = engine
        self.audio_buffer = []
        self.closed = False

        self.text_queue = Queue()

        self.lm_gen = LMGen(
            self.engine.lm_model,
            temp=0,
            temp_text=0,
            use_sampling=False
        )

        self.first_frame = True

    def append_pcm16(self, pcm_bytes: bytes):
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer.append(audio)

    def consume_audio(self):
        if not self.audio_buffer:
            return None

        audio = np.concatenate(self.audio_buffer)

        # DO NOT clear everything
        if len(audio) < self.engine.frame_size:
            return None

        # keep remainder
        usable = audio[: (len(audio) // self.engine.frame_size) * self.engine.frame_size]
        remainder = audio[len(usable):]

        self.audio_buffer = [remainder] if len(remainder) else []
        return usable

# Global engine instance
stt_engine: Optional[StreamingKyutaiEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan management"""
    # Startup
    global stt_engine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stt_engine = StreamingKyutaiEngine(device=device)
    logger.info("✅ Kyutai OpenAI Whisper API Compatible service is ready.")
    
    yield
    
    # Shutdown (if needed)
    logger.info("🔄 Shutting down Kyutai service...")

# --- FastAPI App Setup with modern lifespan ---
app = FastAPI(
    title="Kyutai OpenAI Whisper API Compatible STT", 
    version="3.0.0",
    lifespan=lifespan
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health_check():
    is_ready = stt_engine and stt_engine._model_loaded
    if is_ready:
        return {"status": "healthy", "model_loaded": True, "api_format": "openai_whisper_compatible"}
    else:
        return {"status": "unhealthy", "model_loaded": False}, 503

# --- OpenAI Whisper API Compatible Endpoints ---

@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[str] = Form(None)
):
    """
    OpenAI Whisper API Compatible transcription endpoint
    
    Compatible with:
    - OpenAI official clients
    - Groq API clients  
    - Any Whisper API client
    
    Just change the base_url to point here!
    """
    
    if not stt_engine:
        raise HTTPException(status_code=503, detail="STT engine not ready")
    
    try:
        # Read the uploaded file
        audio_content = await file.read()
        
        # Load audio using librosa (supports many formats)
        audio_data, original_sr = librosa.load(io.BytesIO(audio_content), sr=None, mono=True)
        
        logger.info(f"Processing audio file: {file.filename}, duration: {len(audio_data)/original_sr:.2f}s")
        
        # Transcribe using Kyutai engine
        transcription_text, duration = await stt_engine.transcribe_audio_file(audio_data, original_sr)
        
        # Create OpenAI-compatible response
        if response_format == "text":
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=transcription_text, media_type="text/plain")
        elif response_format == "srt":
            # Simple SRT format
            srt_content = f"1\n00:00:00,000 --> {int(duration//60):02d}:{int(duration%60):02d},{int((duration%1)*1000):03d}\n{transcription_text}\n"
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=srt_content, media_type="text/plain")
        elif response_format == "vtt":
            # Simple VTT format
            vtt_content = f"WEBVTT\n\n00:00:00.000 --> {int(duration//60):02d}:{int(duration%60):02d}.{int((duration%1)*1000):03d}\n{transcription_text}\n"
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=vtt_content, media_type="text/plain")
        else:
            # Default JSON response (OpenAI format)
            segments = []
            if transcription_text:
                segments = [
                    TranscriptionSegment(
                        id=0,
                        seek=0.0,
                        start=0.0,
                        end=duration,
                        text=transcription_text,
                        tokens=[],
                        temperature=temperature,
                        avg_logprob=0.0,
                        compression_ratio=1.0,
                        no_speech_prob=0.0
                    )
                ]
            
            return TranscriptionResponse(
                text=transcription_text,
                task="transcribe",
                language=language or "en",
                duration=duration,
                segments=segments if timestamp_granularities else None
            )
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/v1/audio/translations", response_model=TranscriptionResponse)
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0)
):
    """
    OpenAI Whisper API Compatible translation endpoint
    Note: Kyutai model outputs English, so this behaves the same as transcription
    """
    # For now, treat translation the same as transcription since Kyutai outputs English
    return await create_transcription(
        file=file,
        model=model,
        language="en",  # Force English for translation
        prompt=prompt,
        response_format=response_format,
        temperature=temperature
    )

# --- Keep existing endpoints for backward compatibility ---

@app.websocket("/v1/realtime")
async def openai_realtime_websocket(
    websocket: WebSocket,
    model: str = Query(default="kyutai/stt-1b-en_fr")
):
    await websocket.accept()    

    session_id = f"sess_{int(time.time())}"
    logger.info(f"🟢 [WS:{session_id}] Connected")

    if not stt_engine:
        logger.error("❌ STT engine not ready")
        await websocket.close()
        return

    # session = RealtimeStreamingSession(
    #                 StreamingKyutaiEngine(device=stt_engine.device)
    #             )

    session = RealtimeStreamingSession(stt_engine)

    await websocket.send_json({
        "type": "session.created",
        "session": {
            "id": session_id,
            "model": model,
            "sample_rate": stt_engine.sample_rate
        }
    })
    await websocket.send_json({
                                "type": "response.output_text.delta",
                                "delta": "version 1 "
                            })
    async def decoder_loop():
        logger.info(f"🎧 [WS:{session_id}] Decoder loop started")
        try:
            # async with stt_engine.lock:
                with stt_engine.mimi.streaming(batch_size=1), session.lm_gen.streaming(batch_size=1):
                    while not session.closed:
                        await asyncio.sleep(0.04)  # ⭐ latency control

                        audio = session.consume_audio()
                        if audio is None or len(audio) < stt_engine.frame_size:
                            continue

                        # IMPORTANT: process STRICT Mimi frames only
                        num_frames = len(audio) // stt_engine.frame_size

                        for f in range(num_frames):
                            start = f * stt_engine.frame_size
                            end = start + stt_engine.frame_size
                            chunk = audio[start:end]

                            pcm = torch.from_numpy(chunk).to(stt_engine.device)
                            pcm = pcm.unsqueeze(0).unsqueeze(0)

                            codes = stt_engine.mimi.encode(pcm)

                            if session.first_frame:
                                session.lm_gen.step(codes)
                                session.first_frame = False
                                continue

                            tokens = session.lm_gen.step(codes)
                            if tokens is None:
                                continue

                            text_id = tokens[0, 0].cpu().item()
                            if text_id in (0, 3):
                                continue

                            piece = stt_engine.text_tokenizer.id_to_piece(text_id)
                            text = piece.replace("▁", " ")

                            logger.info(f"📝 [WS:{session_id}] Δ {text}")

                            await session.text_queue.put(text)

        except asyncio.CancelledError:
            logger.warning(f"⛔ [WS:{session_id}] Decoder cancelled")
        except Exception as e:
            logger.exception(f"❌ [WS:{session_id}] Decoder error: {e}")

    decoder_task = asyncio.create_task(decoder_loop())

    async def sender_loop():
        try:
            while not session.closed:
                text = await session.text_queue.get()
                await websocket.send_json({
                    "type": "response.output_text.delta",
                    "delta": text
                })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(f"Sender error: {e}")
    sender_task = asyncio.create_task(sender_loop())
    

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            logger.debug(f"📩 [WS:{session_id}] RX {data.get('type')}")

            if data["type"] == "input_audio_buffer.append":
                pcm = base64.b64decode(data["audio"])
                session.append_pcm16(pcm)

            elif data["type"] == "input_audio_buffer.commit":
                # if not session.audio_buffer:
                #     logger.debug(f"⚠️ Commit ignored (no audio)")
                #     continue
                logger.debug(f"✅ Commit accepted")

    except WebSocketDisconnect:
        logger.warning(f"🔌 [WS:{session_id}] Disconnected")
    except Exception as e:
        logger.exception(f"❌ [WS:{session_id}] WS error: {e}")
    finally:
        session.closed = True
        decoder_task.cancel()
        sender_task.cancel()
        logger.info(f"🛑 [WS:{session_id}] Session closed")

@app.websocket("/v1/audio/stream")
async def legacy_websocket_endpoint(websocket: WebSocket):
    """Legacy endpoint for backward compatibility (existing)"""
    # ... existing streaming code ...
    await websocket.accept()
    # Simplified for space - use your existing streaming implementation

# --- Models endpoint (OpenAI compatible) ---

@app.get("/v1/models")
async def list_models():
    """OpenAI compatible models endpoint"""
    return {
        "object": "list",
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "created": 1677532384,
                "owned_by": "kyutai",
                "permission": [],
                "root": "whisper-1",
                "parent": None
            },
            {
                "id": "kyutai/stt-1b-en_fr",
                "object": "model", 
                "created": 1677532384,
                "owned_by": "kyutai",
                "permission": [],
                "root": "kyutai/stt-1b-en_fr",
                "parent": None
            }
        ]
    }

# --- Main Execution ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"🚀 Starting Kyutai OpenAI Whisper API Compatible service on {host}:{port}")
    logger.info(f"📋 OpenAI Whisper API endpoint: http://{host}:{port}/v1/audio/transcriptions")
    uvicorn.run(app, host=host, port=port, log_level="info")