# Kyutai STT OpenAI API

A FastAPI-based Speech-to-Text service that provides OpenAI Whisper API compatibility using Kyutai's powerful STT models. This allows you to use any OpenAI Whisper client with Kyutai's models as a drop-in replacement.

---

## Features

- **OpenAI Whisper API Compatible** – Works with any OpenAI Whisper client  
- **Multiple Response Formats** – JSON, text, SRT, VTT  
- **Real-time Streaming** – WebSocket support for live transcription  
- **Docker Ready** – Easy deployment with Docker Compose  
- **GPU Accelerated** – CUDA support for faster processing  
- **Multiple Audio Formats** – Supports MP3, WAV, FLAC, and more via `librosa`

---

## Quick Start

### Using Docker Compose (Recommended)

```bash
git clone https://github.com/dwain-barnes/kyutai-stt-openai-api.git
cd kyutai-stt-openai-api
docker-compose up -d
```

The service will be available at [http://localhost:8080](http://localhost:8080)

---

## Manual Installation

```bash
# Clone the repository
git clone https://github.com/dwain-barnes/kyutai-stt-openai-api.git
cd kyutai-stt-openai-api

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### With OpenAI Python Client

```python
from openai import OpenAI

# Point to your Kyutai service instead of OpenAI
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy_key"  # Not validated
)

# Use exactly like OpenAI Whisper
with open("audio.mp3", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )

print(transcription.text)
```

---

### Run the Service

```bash
python app.py
```

---

### With cURL

```bash
curl -X POST "http://localhost:8080/v1/audio/transcriptions"   -H "Content-Type: multipart/form-data"   -F "file=@audio.mp3"   -F "model=whisper-1"   -F "response_format=json"
```

---

## Response Formats

- **JSON** – Standard OpenAI format with segments and metadata  
- **Text** – Plain text transcription  
- **SRT** – Subtitle format  
- **VTT** – WebVTT format

---

## API Endpoints

- `POST /v1/audio/transcriptions` – Transcribe audio (OpenAI compatible)  
- `POST /v1/audio/translations` – Translate audio to English  
- `GET /v1/models` – List available models  
- `GET /health` – Health check  
- `WS /v1/realtime` – Real-time streaming transcription

---

## Configuration

Set environment variables to customize the service:

- `PORT` – Service port (default: `8080`)  
- `HOST` – Service host (default: `0.0.0.0`)  
- `MODEL_NAME` – Kyutai model to use (default: `kyutai/stt-1b-en_fr`)  
- `CUDA_VISIBLE_DEVICES` – GPU selection

---

## Requirements

- Python 3.12+  
- PyTorch with CUDA support  
- 4GB+ GPU memory (recommended)  
- FFmpeg for audio processing  

---

## Model Information

This service uses Kyutai's STT models:

- `kyutai/stt-1b-en_fr`: 1B parameter model supporting English and French  
- Optimized for streaming and real-time applications  
- High accuracy with low latency

---

## Docker Deployment

The included `docker-compose.yml` provides:

- GPU acceleration  
- Health checks  
- Volume mounts for model caching  
- Automatic restart policies

---

## Testing

Run the test script to verify your installation:

```bash
# Example testing script (replace with actual test command if applicable)
python test.py
```
