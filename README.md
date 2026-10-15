---
title: WaZoBia XTTS Voice Engine
emoji: 🗣️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: other
short_description: WaZoBia XTTS-v2 Speech Synthesis & Voice Cloning Studio
---

# 🗣️ WaZoBia XTTS Voice Engine: Sovereign Nigerian Speech Synthesis & Voice Cloning

A production Gradio speech application hosting all **6 fine-tuned XTTS-v2 voice models** for Nigerian Sovereign Languages: **Hausa**, **Yorùbá**, and **Igbo**.

Designed to run seamlessly as a **Hugging Face ZeroGPU Space** and as an independent **local repository**.

---

## 🌟 The 6 Fine-Tuned Models (Stage 1 & Stage 2)

The models were developed across a two-stage training curriculum to transition from formal studio reading to natural conversational Nigerian cadence:

### 🟡 Stage 1: Foundational Acoustic Models
Trained on studio-recorded scripture narration for foundational phoneme precision, accent preservation, and tone stability.
- **Hausa (Foundation)**: [`samuelolubukun/xtts-v2-hausa-openbible`](https://huggingface.co/samuelolubukun/xtts-v2-hausa-openbible)
- **Yorùbá (Foundation)**: [`samuelolubukun/xtts-v2-yoruba-openbible`](https://huggingface.co/samuelolubukun/xtts-v2-yoruba-openbible)
- **Igbo (Foundation)**: [`samuelolubukun/xtts-v2-igbo-openbible`](https://huggingface.co/samuelolubukun/xtts-v2-igbo-openbible)

### 🟢 Stage 2: Conversational Fine-Tuned Models (Recommended)
Further adapted on spontaneous conversational speech for natural pacing, micro-inflections, emotional range, and authentic tone diacritics.
- **Hausa (Conversational)**: [`samuelolubukun/xtts-v2-hausa-openbible-conversational`](https://huggingface.co/samuelolubukun/xtts-v2-hausa-openbible-conversational)
- **Yorùbá (Conversational)**: [`samuelolubukun/xtts-v2-yoruba-openbible-conversational`](https://huggingface.co/samuelolubukun/xtts-v2-yoruba-openbible-conversational)
- **Igbo (Conversational)**: [`samuelolubukun/xtts-v2-igbo-openbible-conversational`](https://huggingface.co/samuelolubukun/xtts-v2-igbo-openbible-conversational)

---

## 🎙️ Voice Cloning & Platform Voices

- **Bundled Platform Voices**:
  - `male_victor`: Authentic Nigerian male speaker reference ([`voices/male_victor.wav`](voices/male_victor.wav)).
  - `female_blessing`: Expressive Nigerian female speaker reference ([`voices/female_blessing.wav`](voices/female_blessing.wav)).
- **Zero-Shot Voice Cloning**: Upload or record a short (3–10s) audio clip to clone any speaker's voice in Hausa, Yorùbá, or Igbo while maintaining native language accents.

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/samolubukun/wazobia-xtts.git
cd wazobia-xtts

# Install dependencies
pip install -r requirements.txt

# Launch the Gradio web application
python app.py
```

The web interface will start at `http://localhost:7860`.

---

## ⚖️ Datasets, Attributions & Citations

1. **Base Architecture**: Coqui XTTS-v2 ([Coqui Public Model License](https://coqui.ai/cpml)).
2. **Datasets**:
   - **OpenBible Project**: Scripture narration corpus for Stage 1 acoustic foundation.
   - **NaijaVoices Community**: Spontaneous West African multi-speaker recordings ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
   - **Google FLEURS**: Multi-accent speech evaluation benchmarks.
3. **Fine-Tuning & Engineering**: [Samuel Olubukun](https://github.com/samolubukun).
