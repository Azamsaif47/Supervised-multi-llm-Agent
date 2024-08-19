# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import your LLM processing function
from ollloo import process_input

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

class QueryResponse(BaseModel):
    response_text: str

@app.post("/process_query", response_model=QueryResponse)
async def process_query(query: QueryRequest):
    input_text = query.input_text
    response_text = process_input(input_text)  # Use your LLM processing function here
    return QueryResponse(response_text=response_text)

# To run the server: uvicorn main:app --reload
