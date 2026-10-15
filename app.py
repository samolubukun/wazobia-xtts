"""WaZoBia XTTS-v2 Speech Synthesis Engine for Hugging Face ZeroGPU Spaces.

Features all 6 Sovereign XTTS-v2 Models:
  1. samuelolubukun/xtts-v2-hausa-openbible-conversational
  2. samuelolubukun/xtts-v2-hausa-openbible
  3. samuelolubukun/xtts-v2-igbo-openbible-conversational
  4. samuelolubukun/xtts-v2-igbo-openbible
  5. samuelolubukun/xtts-v2-yoruba-openbible-conversational
  6. samuelolubukun/xtts-v2-yoruba-openbible

Bundled Platform Voices:
  - male_victor (voices/male_victor.wav)
  - female_blessing (voices/female_blessing.wav)

Endpoints:
  - Interactive Gradio Web UI
  - OpenAI-compatible REST API (/v1/audio/speech, /v1/models, /healthz)
  - ZeroGPU compatible via @spaces.GPU
"""

import hmac
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import gradio as gr
import gradio_client.utils
import spaces

# Fix Gradio schema bug
_orig_get_type = gradio_client.utils.get_type
def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return _orig_get_type(schema)
gradio_client.utils.get_type = _patched_get_type

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

# Monkeypatch transformers for XTTS compatibility with modern transformers
import transformers.utils.import_utils
transformers.utils.import_utils.is_torchcodec_available = lambda: True

from transformers import pytorch_utils
if not hasattr(pytorch_utils, "isin_mps_friendly"):
    def patched_isin_mps_friendly(elements, test_elements):
        if test_elements.ndim == 0:
            test_elements = test_elements.unsqueeze(0)
        return elements.tile(test_elements.shape[0], 1).eq(test_elements.unsqueeze(1)).sum(dim=0).bool().squeeze()
    pytorch_utils.isin_mps_friendly = patched_isin_mps_friendly

# Monkeypatch torchaudio.load & TTS.tts.models.xtts.load_audio to use soundfile/librosa
# This prevents: "ImportError: TorchCodec is required for load_with_torchcodec"
def _safe_load_audio(audiopath, sr=22050):
    try:
        wav, original_sr = sf.read(audiopath)
    except Exception:
        wav, original_sr = librosa.load(audiopath, sr=None)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if original_sr != sr:
        wav = librosa.resample(wav, orig_sr=original_sr, target_sr=sr)
    tensor = torch.from_numpy(wav).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

def _safe_torchaudio_load(filepath, *args, **kwargs):
    wav, sr = sf.read(filepath)
    if wav.ndim == 1:
        tensor = torch.from_numpy(wav).unsqueeze(0).float()
    else:
        tensor = torch.from_numpy(wav.T).float()
    return tensor, sr

torchaudio.load = _safe_torchaudio_load

# Monkeypatch TTS check for torchcodec
try:
    import TTS.utils.import_utils
    TTS.utils.import_utils.is_torchcodec_available = lambda: True
except Exception:
    pass

import TTS.tts.models.xtts
TTS.tts.models.xtts.load_audio = _safe_load_audio

# Monkeypatch GPT2InferenceModel.forward for transformers>=4.48 compatibility
import TTS.tts.layers.xtts.gpt_inference
_orig_gpt_forward = TTS.tts.layers.xtts.gpt_inference.GPT2InferenceModel.forward

def _safe_gpt_forward(self, input_ids=None, past_key_values=None, attention_mask=None, **kwargs):
    if attention_mask is None and input_ids is not None:
        device = input_ids.device
        prefix_len = self.cached_prefix_emb.shape[1] if hasattr(self, 'cached_prefix_emb') and self.cached_prefix_emb is not None else 0
        past_len = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_len = past_key_values.get_seq_length()
            elif hasattr(past_key_values, "seen_tokens"):
                past_len = past_key_values.seen_tokens
            elif hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > 0:
                past_len = past_key_values.key_cache[0].shape[-2]
            else:
                try:
                    past_len = past_key_values[0][0].shape[-2]
                except Exception:
                    past_len = 0
        seq_len = prefix_len + past_len + input_ids.shape[1]
        attention_mask = torch.ones((input_ids.shape[0], seq_len), dtype=torch.long, device=device)
    return _orig_gpt_forward(self, input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs)

TTS.tts.layers.xtts.gpt_inference.GPT2InferenceModel.forward = _safe_gpt_forward

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from starlette.routing import BaseRoute, Match

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("XTTS-WaZoBia")

MODELS_CATALOG = {
    "hausa-conversational": "samuelolubukun/xtts-v2-hausa-openbible-conversational",
    "hausa-foundation": "samuelolubukun/xtts-v2-hausa-openbible",
    "yoruba-conversational": "samuelolubukun/xtts-v2-yoruba-openbible-conversational",
    "yoruba-foundation": "samuelolubukun/xtts-v2-yoruba-openbible",
    "igbo-conversational": "samuelolubukun/xtts-v2-igbo-openbible-conversational",
    "igbo-foundation": "samuelolubukun/xtts-v2-igbo-openbible",
}

LANG_CODE_MAP = {
    "hausa-conversational": "ha",
    "hausa-foundation": "ha",
    "yoruba-conversational": "yo",
    "yoruba-foundation": "yo",
    "igbo-conversational": "ig",
    "igbo-foundation": "ig",
}

PLATFORM_VOICES = {
    "female_blessing": os.path.join(os.path.dirname(__file__), "voices", "female_blessing.wav"),
    "male_victor": os.path.join(os.path.dirname(__file__), "voices", "male_victor.wav"),
}

DEFAULT_MODEL_KEY = "hausa-conversational"

loaded_xtts_models: Dict[str, Xtts] = {}
loaded_configs: Dict[str, XttsConfig] = {}

def get_xtts_model(model_key: str):
    if model_key not in MODELS_CATALOG:
        model_key = DEFAULT_MODEL_KEY
    repo_id = MODELS_CATALOG[model_key]

    if model_key not in loaded_xtts_models:
        logger.info(f"Downloading/Loading XTTS model checkpoint: {repo_id}")
        model_dir = snapshot_download(repo_id=repo_id)

        config = XttsConfig()
        config.load_json(os.path.join(model_dir, "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=os.path.join(model_dir, "model.pth"),
            vocab_path=os.path.join(model_dir, "vocab.json"),
            use_deepspeed=False,
        )

        orig_preprocess = model.tokenizer.preprocess_text
        def safe_preprocess_text(txt, lang):
            try:
                return orig_preprocess(txt, lang)
            except NotImplementedError:
                txt = txt.lower()
                return re.sub(r"\s+", " ", txt).strip()
        model.tokenizer.preprocess_text = safe_preprocess_text

        model.eval()
        loaded_xtts_models[model_key] = model
        loaded_configs[model_key] = config
        logger.info(f"Loaded XTTS model: {repo_id}")

    return loaded_xtts_models[model_key], loaded_configs[model_key], repo_id

@spaces.GPU(duration=90)
def gpu_synthesize(
    text: str,
    model_key: str,
    reference_wav_path: str,
    temperature: float = 0.75,
    repetition_penalty: float = 2.0,
) -> np.ndarray:
    model, config, repo_id = get_xtts_model(model_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    lang_code = LANG_CODE_MAP.get(model_key, "yo")

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=reference_wav_path,
        gpt_cond_len=30,
        max_ref_length=30,
        sound_norm_refs=False,
    )

    with torch.inference_mode():
        out = model.inference(
            text=text.strip(),
            language=lang_code,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
        )

    wav = out["wav"]
    return wav

# ---------------------------------------------------------------------------
# FastAPI Endpoints (OpenAI audio/speech compatible)
# ---------------------------------------------------------------------------
class OpenAISpeechRequest(BaseModel):
    model: str = Field(default=MODELS_CATALOG[DEFAULT_MODEL_KEY])
    input: str
    voice: Optional[str] = Field(default="female_blessing")
    response_format: Optional[Literal["wav", "mp3"]] = Field(default="wav")
    speed: Optional[float] = Field(default=1.0)

fastapi_app = FastAPI(
    title="WaZoBia XTTS-v2 Speech API",
    description="OpenAI-compatible /v1/audio/speech endpoint hosting all 6 XTTS models for Hausa, Igbo, Yoruba.",
    version="1.0.0",
)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_key_scheme = APIKeyHeader(name="Authorization", scheme_name="BearerAuth", auto_error=False)

def verify_api_key(auth_header: Optional[str] = Security(_api_key_scheme)):
    expected = os.environ.get("XTTS_API_KEY")
    if not expected:
        return True
    token = ""
    if auth_header:
        token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else auth_header.strip()
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return True

@fastapi_app.get("/healthz")
def health_check():
    return {
        "status": "healthy",
        "service": "wazobia-xtts-tts",
        "models": list(MODELS_CATALOG.values()),
        "platform_voices": list(PLATFORM_VOICES.keys()),
        "time": datetime.utcnow().isoformat(),
    }

@fastapi_app.get("/v1/models")
def list_models(auth=Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": repo,
                "object": "model",
                "owned_by": "samuelolubukun",
                "type": "text-to-speech",
            }
            for repo in MODELS_CATALOG.values()
        ],
    }

@fastapi_app.post("/v1/audio/speech")
async def create_speech(req: OpenAISpeechRequest, auth=Depends(verify_api_key)):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required.")

    # Resolve model
    model_key = DEFAULT_MODEL_KEY
    for k, v in MODELS_CATALOG.items():
        if req.model.strip().lower() in [k.lower(), v.lower()]:
            model_key = k
            break

    # Resolve voice
    ref_path = PLATFORM_VOICES.get(req.voice.lower(), PLATFORM_VOICES["female_blessing"])

    try:
        wav = await run_in_threadpool(gpu_synthesize, req.input, model_key, ref_path)
    except Exception as e:
        logger.error(f"XTTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    buffer = io.BytesIO()
    sf.write(buffer, wav, 24000, format="WAV")
    buffer.seek(0)

    return Response(content=buffer.read(), media_type="audio/wav")

# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
def gradio_tts(text_input, model_choice, voice_choice, custom_audio, temp, rep_pen):
    if not text_input or not text_input.strip():
        return None, "⚠️ Please provide text to synthesize."

    # Determine reference voice
    if custom_audio:
        ref_path = custom_audio
    else:
        ref_path = PLATFORM_VOICES.get(voice_choice, PLATFORM_VOICES["female_blessing"])

    model_key = DEFAULT_MODEL_KEY
    for k, v in MODELS_CATALOG.items():
        if model_choice == v or model_choice == k:
            model_key = k
            break

    wav = gpu_synthesize(
        text=text_input,
        model_key=model_key,
        reference_wav_path=ref_path,
        temperature=temp,
        repetition_penalty=rep_pen,
    )

    out_file = "output_xtts.wav"
    sf.write(out_file, wav, 24000)
    info = f"✅ Synthesized {len(text_input.split())} words using `{MODELS_CATALOG[model_key]}` ({'Custom Audio' if custom_audio else voice_choice})."
    return out_file, info

with gr.Blocks(title="WaZoBia XTTS Demo") as demo:
    gr.Markdown("""# 🗣️ WaZoBia XTTS Demo
### Sovereign Nigerian Voice Synthesis & Zero-Shot Cloning (All 6 Models)
Supports Yoruba, Hausa, and Igbo with platform voices (**Victor** & **Blessing**) and custom zero-shot voice cloning.
""")

    with gr.Row():
        with gr.Column(scale=3):
            text_prompt = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter text in Hausa, Yoruba, or Igbo...",
                lines=4,
                value="Ina kwana, fatan kowa yana cikin ƙoshin lafiya a yau?",
            )
            synth_btn = gr.Button("🎙️ Generate Speech", variant="primary")
            audio_out = gr.Audio(label="Synthesized Speech", type="filepath")
            status_text = gr.Markdown("")

            gr.Examples(
                examples=[
                    ["Ina kwana, fatan kowa yana cikin ƙoshin lafiya a yau?", MODELS_CATALOG["hausa-conversational"], "male_victor"],
                    ["Ẹ kú àárọ̀ gbogbo ilé, ṣé àlàáfíà ni ẹ wà lónìí?", MODELS_CATALOG["yoruba-conversational"], "female_blessing"],
                    ["Nnọọ, kedu ka ị mere taa?", MODELS_CATALOG["igbo-conversational"], "female_blessing"],
                ],
                inputs=[text_prompt],
            )

        with gr.Column(scale=2):
            model_drop = gr.Dropdown(
                choices=list(MODELS_CATALOG.values()),
                value=MODELS_CATALOG[DEFAULT_MODEL_KEY],
                label="Select XTTS Model",
            )
            voice_drop = gr.Dropdown(
                choices=["female_blessing", "male_victor"],
                value="female_blessing",
                label="Platform Voice",
            )
            custom_voice = gr.Audio(
                label="Or Upload Custom Reference Voice (3-10s)",
                type="filepath",
                sources=["upload", "microphone"],
            )
            with gr.Accordion("⚙️ Advanced Parameters", open=False):
                temp_slider = gr.Slider(0.1, 1.2, value=0.75, step=0.05, label="Temperature")
                rep_slider = gr.Slider(1.0, 3.0, value=2.0, step=0.1, label="Repetition Penalty")

    synth_btn.click(
        gradio_tts,
        inputs=[text_prompt, model_drop, voice_drop, custom_voice, temp_slider, rep_slider],
        outputs=[audio_out, status_text],
    )

    with gr.Accordion("⚖️ Licenses, Attributions & Ethical Voice Cloning Disclaimer", open=False):
        gr.Markdown("""
### 📜 Dual-License Architecture
This Space and the underlying fine-tuned models operate under a dual-licensing framework:
1. **Base Architecture License**: [Coqui Public Model License 1.0.0 (CPML)](https://coqui.ai/cpml).
2. **Dataset License**: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) governing the fine-tuning datasets (OpenBible, NaijaVoices, and FLEURS).
> **Non-Commercial Research & Development**: By using this Space and its API, you agree that commercial use is subject to the provisions of the CPML and CC BY-NC-SA 4.0 licenses.

### 🛡️ Ethical Voice Cloning & Consent Notice
- **Consent Requirement**: Users must obtain explicit, verifiable consent from any identifiable speaker before cloning their voice.
- **Prohibited Uses**: It is strictly forbidden to use this service for impersonation, deceptive synthetic media (deepfakes), defamation, harassment, or financial scams.
- **Attribution**: Built by [Samuel Olubukun](https://github.com/samolubukun) for linguistic preservation, educational access, and African language AI research.
""")

    with gr.Accordion("🔌 OpenAI-Compatible API Usage", open=False):
        gr.Markdown("""
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<YOUR-SPACE-NAME>.hf.space/v1",
    api_key="your-key",  # If XTTS_API_KEY is configured
)

res = client.audio.speech.create(
    model="samuelolubukun/xtts-v2-yoruba-openbible-conversational",
    voice="female_blessing",  # or "male_victor"
    input="Bawo ni o se wa loni? Se alafia ni gbogbo ile wa?",
)
res.stream_to_file("speech.wav")
```
""")

class ApiDelegateRoute(BaseRoute):
    def matches(self, scope):
        if scope["type"] == "http":
            p = scope.get("path", "")
            if p == "/healthz" or p == "/v1" or p.startswith("/v1/"):
                return Match.FULL, {}
        return Match.NONE, {}

    async def handle(self, scope, receive, send):
        await fastapi_app(scope, receive, send)

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False, prevent_thread_lock=True)
    demo.app.router.routes.insert(0, ApiDelegateRoute())
    logger.info("OpenAI API routes (/healthz, /v1/*) attached to Gradio server.")
    demo.block_thread()
