# 🧠 aicreators_video_agent  
**Open-Source AI Video Agent for Short-Form Content Automation**

---

## Overview

This is a lightweight, open-source version of a high-performance AI video generation system originally built for TikTok Shop creators. It combines scriptwriting, voice synthesis, eleven labs enhancements, avatars + lipsync, and video-clipping into a simple, modular pipeline — perfect for generating short-form content at scale.

🔧 Built for:
- AI avatar videos  
- Affiliate marketing content  
- Creator automation  
- TikTok Shop, reels, and shorts  
- Batch production using YAML jobs  

---

## 🚀 Core Features

- ✅ Script generation using OpenAI GPT
- ✅ Voice synthesis using ElevenLabs
- ✅ Avatar overlays (pre-recorded lip-synced clips)
- ✅ YAML-powered batch job creation (`campaigns.yaml`)
- ✅ Basic environment setup with `.env` support

> 🧠 Want the ZYRA PRO version with product overlays, Whisper timing, and out state-of-the-art Anti-Violation feature?
> Join here... coming soon → https://jonnyvandel.com/ai

---

## 🛠 Project Structure

```bash
aicreators_video_agent/
├── create_video.py         # Core script to generate videos
├── run_batch.py            # Batch job executor (reads from campaigns.yaml)
├── campaigns.yaml          # All job definitions (personas, products, etc.)
├── examples/               # Sample scripts (txt) used in generation
├── Avatars/                # Avatar base videos (add your own)
├── output/                 # Final generated videos
├── .env.example            # Template for required API keys
├── requirements.txt        # Project dependencies
```

---

## ⚙️ Requirements

- Python 3.10+  
- `ffmpeg` installed and accessible in your system PATH  
- Packages in `requirements.txt`

```bash
openai
elevenlabs
python-dotenv
PyYAML
requests
google-cloud-storage
```

---

## 🔧 Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/JonnyVandelNetwork/aicreators_video_agent.git
   cd aicreators_video_agent
   ```

2. **Create a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install requirements**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file from template**  
   ```bash
   cp .env.example .env
   ```

5. **Run batch jobs**  
   ```bash
   python run_batch.py
   ```

---

## 🧠 Notes

- All jobs are configured via `campaigns.yaml`  
- `examples/` contains scripts with tone/style references  
- Avatar base videos must be added but the user.

---

## 👤 Author

Built by [Jonny Vandel](https://www.instagram.com/jonnyvandel)  

---

## 🔒 Zyra... agent system coming soon.

The full commercial MCP system (`Zyra by @aicreators`) includes:
- ✅ Natural Language to video (Never done before)
- ✅ AI-Generated Unique Avatars
- ✅ AI edits the video (product overlays, images, videos, clipping, captions, etc.)
- ✅ Bypass content violation detection
- ✅ Multi-product support and advanced personas
- ✅ SaaS-ready MCP system. (For devs)
- ✅ Schedule & Post. (Multi-Platform)

🎯 Join the aicreators for early access → https://jonnyvandel.com/ai
