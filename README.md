# Live Transcription Server

A real-time speech-to-text application built with FastAPI, WebSockets, and Vosk. The project supports live transcription for both English and Hindi audio streams with automatic text post-processing for improved readability.

## Features

* Real-time transcription using WebSockets
* English and Hindi language support
* Automatic punctuation insertion
* Date and time normalization
* Number-to-digit conversion
* Hindi text normalization
* Duplicate word removal
* Session-based client handling
* FastAPI backend with static frontend support

## Technologies Used

* Python
* FastAPI
* WebSockets
* Vosk
* NumPy
* SoundFile
* word2number

## Project Structure

```text
live-transcription-server/
│
├── app.py
├── requirements.txt
├── models/
│   ├── vosk-model-en-in-0.5/
│   └── vosk-model-small-hi-0.22/
└── static/
    └── index.html
```

## How It Works

1. A client connects to the FastAPI WebSocket endpoint.
2. The user selects either English or Hindi transcription.
3. Audio is streamed to the server in real time.
4. Vosk processes the audio and returns partial and final transcripts.
5. Custom post-processing functions normalize:

   * Dates and times
   * Numbers
   * Capitalization
   * Punctuation
   * Hindi text formatting

## Running the Project

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Download the required Vosk models and place them inside the `models` directory.

3. Start the server:

```bash
uvicorn app:app --reload
```

4. Open the application in your browser:

```text
http://localhost:8000
```

## Example Use Cases

* Meeting transcription
* Live note taking
* Voice-enabled applications
* Accessibility tools
* Multilingual speech processing

## Future Improvements

* Whisper integration
* Speaker diarization
* Support for additional languages
* Transcript export functionality
* Cloud deployment support

## Author

Developed by Advaith Rakshan.
