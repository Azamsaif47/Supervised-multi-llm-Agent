import os
import requests
import logging
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supervised_agent import process_input  # Import the existing process_input function

# Load environment variables from .env file
load_dotenv(".env")

# Fetch Eleven Labs API credentials and URL from the .env file
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")
ELEVEN_LABS_URL = os.getenv("ELEVEN_LABS_URL")

# Check if required environment variables are loaded
if not ELEVEN_LABS_API_KEY or not VOICE_ID or not ELEVEN_LABS_URL:
    raise ValueError(
        "Eleven Labs API key, Voice ID, or URL not found. Please ensure they are set in the .env file."
    )

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
