import os
import re
import streamlit as st
import speech_recognition as sr
import pyttsx3
import pythoncom



from ollloo import process_input
# Set the environment variable to avoid OpenMP runtime errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def initialize_tts_engine():
    pythoncom.CoInitialize()
    engine = pyttsx3.init()
    return engine

def listen(engine):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        #st.write("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=5)

        #st.write("Okay, go!")
        while True:
            st.write("Listening now...")
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=30)
                #st.write("Recognizing...")
                
                # Recognize the speech using Whisper
                text = r.recognize_whisper(
                    audio,
                    model="base.en",
                    show_dict=True
                )["text"]

                st.write(f"Transcription: {text}")
                

                output_message = process_input(text)
                response_text = output_message
                st.write(response_text)

                # Convert text response to speech
                engine.say(response_text)
                engine.runAndWait()
                
            except Exception as e:
                st.write(f"Error: {e}")

def main():
    # Initialize the COM environment and TTS engine
    pythoncom.CoInitialize()
    engine = initialize_tts_engine()
    
    st.title("Continuous Voice Assistant")

    if st.button("Start Listening"):
        listen(engine)

if __name__ == "__main__":
    main()
