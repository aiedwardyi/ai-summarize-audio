# Audio Transcription & AI Analyzer 🎙️🤖

A local Python application that transcribes audio files using **OpenAI's Whisper** and generates structured key points using **Llama 3.2** via **IBM watsonx.ai**.

## Features
- **Local Transcription:** Uses `whisper-tiny.en` for fast, local audio processing.
- **AI Summary:** Uses Llama 3.2 to extract key points from the transcript.
- **Web UI:** Simple Gradio interface for easy file uploads.

## Prerequisites
- **Python 3.11+**
- **FFmpeg:** Installed and added to your system PATH.
- **IBM Cloud Account:** For watsonx.ai API access.

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com
cd speech-to-text
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables:**
Create a .env file in the root directory and add your credentials:

```env
WATSONX_APIKEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
```

5. **Usage:**
Run the application:
```bash
python speech_analyzer.py
```

Open the local URL provided in the terminal (usually http://127.0.0.1:7860) in your browser.

## License
This project is licensed under the MIT License

Copyright (c) 2024 aiedwardyi