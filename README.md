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
- API Keys for:
  - OpenAI API
  - ElevenLabs
  - Dreamface API
  - Google Cloud Storage bucket

---

## 🔧 Detailed Setup Guide

### 1. Install Python

#### Windows:
1. Download Python from [python.org](https://www.python.org/downloads/) (Choose Python 3.10 or higher)
2. Run the installer and check "Add Python to PATH" before installing
3. Verify installation by opening Command Prompt and typing: `python --version`

#### macOS:
1. Install Homebrew if you don't have it: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. Install Python: `brew install python`
3. Verify installation: `python3 --version`

#### Linux:
1. Most distributions come with Python. If not, install with:
   - Ubuntu/Debian: `sudo apt install python3 python3-pip`
   - Fedora: `sudo dnf install python3 python3-pip`
2. Verify installation: `python3 --version`

### 2. Install ffmpeg

#### Windows:
1. Download ffmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract the zip file
3. Add the bin folder to your PATH:
   - Right-click on "This PC" > Properties > Advanced system settings > Environment Variables
   - Edit the PATH variable and add the path to the ffmpeg bin folder

#### macOS:
```bash
<<<<<<< HEAD
openai
elevenlabs
python-dotenv
PyYAML
requests
google-cloud-storage
=======
brew install ffmpeg
>>>>>>> 51cbd03 (Update README with detailed setup instructions)
```

#### Linux:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

### 3. Clone the Repository
```bash
# If you don't have git installed:
# Windows: Download from https://git-scm.com/download/win
# macOS: brew install git
# Linux: sudo apt install git

git clone https://github.com/JonnyVandelNetwork/aicreators_video_agent.git
cd aicreators_video_agent
```

### 4. Create a Virtual Environment
This isolates your project dependencies from other Python projects.

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` at the beginning of your command prompt when the virtual environment is active.

### 5. Install Requirements
```bash
# Windows
pip install -r requirements.txt

# macOS/Linux
pip3 install -r requirements.txt
```

This will also install `google-cloud-storage` which you need for the project.

### 6. Obtain Required API Keys

#### OpenAI API:
1. Create an account at [platform.openai.com](https://platform.openai.com/)
2. Go to [API keys section](https://platform.openai.com/account/api-keys)
3. Create a new secret key and save it

#### ElevenLabs:
1. Create an account at [elevenlabs.io](https://elevenlabs.io/)
2. Go to your profile settings
3. Copy your API key

#### Dreamface API:
1. Create an account at [newportai.com](https://api.newportai.com/)
2. Obtain your API key from the dashboard

### 7. Set Up Google Cloud Storage

1. Create a Google Cloud account at [cloud.google.com](https://cloud.google.com/)
2. Create a new project
3. Enable the Cloud Storage API:
   - Go to "APIs & Services" > "Library"
   - Search for "Cloud Storage"
   - Click "Enable"
4. Create a storage bucket:
   - Go to "Cloud Storage" > "Buckets"
   - Click "Create"
   - Choose a globally unique bucket name
   - Choose settings (defaults are fine)
   - Click "Create"
5. Create a service account with Storage Admin permissions:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Give it a name
   - Assign "Storage Admin" role
   - Click "Done"
6. Create a key for this service account:
   - Click on the service account you just created
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose JSON format
   - Download the key file to your computer


### 8. Configure Environment Variables

1. Create your .env file:
```bash
cp .env.example .env
```

2. Edit the .env file and add your API keys:
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
DREAMFACE_API_KEY=your_dreamface_api_key_here
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key-file.json"
GCS_BUCKET_NAME=your_gcs_bucket_name_here


3. Copy the Google Cloud JSON key file you downloaded in step 7 into your project folder. 
   The path in your .env file should match the filename of your JSON file.


### 9. Add Avatar Videos

1. Place at least one video file in the `Avatars` folder
2. The video should be:
   - MP4 format
   - Clear shot of a person talking
   - Good quality video
   - Preferably with a neutral expression

### 10. Update campaigns.yaml

The sample campaigns.yaml file contains example jobs. Update it with your own information:

1. Set `enabled: true` for the jobs you want to run
2. Update the `product`, `persona`, `setting`, `emotion`, and `hook` fields
3. Set `elevenlabs_voice_id` to an ID from your ElevenLabs account
4. Set `avatar_video_path` to point to a video in your Avatars folder
5. Set `example_script_file` to one in your examples folder
6. Set `brand_name` to the brand you want to mention

### 11. Run Your First Video

```bash
# Make sure your virtual environment is activated

# For batch mode (using campaigns.yaml):
python run_batch.py

# For single video mode:
python create_video.py --product "Your Product" --persona "Male, 35" --setting "Home Office" --emotion "Enthusiastic" --hook "Share a story about this product" --elevenlabs_voice_id "your_voice_id" --avatar_video_path "./Avatars/your_video.mp4" --example_script_file "./examples/your_script.txt" --remove_silence
```

The generated videos will be in the `output` folder.

---

## 🔍 Troubleshooting

### Common Issues:

1. **"ModuleNotFoundError" when running scripts**
   - Make sure you've activated your virtual environment
   - Try reinstalling dependencies: `pip install -r requirements.txt`

2. **"ffmpeg command not found"**
   - Make sure ffmpeg is installed and added to your PATH
   - Restart your terminal after installing ffmpeg

3. **API errors**
   - Double-check your API keys in the .env file
   - Ensure you have sufficient credits in your API accounts

4. **Google Cloud Storage errors**
   - Verify that your GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly
   - Check that your service account has Storage Admin permissions
   - Ensure your GCS bucket exists and is accessible

---

## 👤 Author

Built by [Jonny Vandel](https://www.instagram.com/jonnyvandel)  

---

## 🔒 Zyra: Full AI Agent System (Coming Soon)

The full commercial MCP system (`Zyra by @aicreators`) includes:
- ✅ Natural Language to video (Never done before)
- ✅ AI-Generated Unique Avatars
- ✅ AI edits the video (product overlays, images, videos, clipping, captions, etc.)
- ✅ Bypass content violation detection
- ✅ Multi-product support and advanced personas
- ✅ SaaS-ready MCP system. (For devs)
- ✅ Schedule & Post. (Multi-Platform)

🎯 Join the aicreators for early access → https://jonnyvandel.com/ai
