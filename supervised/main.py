from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import logging
from ollloo import process_input  # Import the existing process_input function

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to allow specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    input_text: str

# Replace with your actual Eleven Labs API key and voice ID
ELEVEN_LABS_API_KEY = "12a9c30d597d6b7154c7ad69d6b39138"
VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"
ELEVEN_LABS_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

def text_to_speech(text: str) -> bytes:
    headers = {
        "Accept": "application/json",
        "xi-api-key": ELEVEN_LABS_API_KEY,
    }
    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }

    try:
        response = requests.post(ELEVEN_LABS_URL, json=payload, headers=headers, stream=True)
        response.raise_for_status()

        # Return the audio content if Content-Type is correct
        if response.headers.get('Content-Type') == 'audio/mpeg':
            return response.content
        else:
            logging.error("Received invalid Content-Type from Eleven Labs API")
            raise HTTPException(status_code=500, detail="Invalid response from Eleven Labs API")
    except requests.RequestException as e:
        logging.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")

@app.post("/process_query")
async def process_query(query: QueryRequest):
    input_text = query.input_text
    try:
        # Step 1: Process the input text using the existing process_input function
        processed_text = process_input(input_text)

        # Step 2: Convert the processed text to speech using Eleven Labs API
        audio_data = text_to_speech(processed_text)

        # Step 3: Return the audio data as the response
        return Response(content=audio_data, media_type="audio/mpeg")
    except HTTPException as e:
        raise e
