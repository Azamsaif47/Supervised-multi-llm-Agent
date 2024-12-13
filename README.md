# Supervised Multi-LLM Agent

This repository contains a **Supervised Multi-LLM Agent**, an open-source project that integrates several LLM agents for performing SQL operations, processing PDF contexts, and responding in audio. The agents are controlled by a central "Supervised Agent" built using LangChain and LangGraph, enabling seamless interactions through a Streamlit interface.

## Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Files Description](#files-description)
- [Requirements](#requirements)
- [License](#license)

## Overview

This project enables a multi-agent system where different LLM agents handle specific tasks, including SQL database operations, PDF context extraction, and audio responses. The Supervised Agent, powered by LangChain and LangGraph, acts as the coordinator, managing the different agents and ensuring smooth operations. The project also includes a Streamlit UI for easy interaction with the system.

## Components

1. **SQL Agent (`sql_agent.py`)**  
   This agent is responsible for handling all operations related to SQL databases. It can execute queries, retrieve data, and interact with any SQL-based database.

2. **PDF Agent (`pdf_agent.py`)**  
   This agent processes and extracts relevant information from PDF files. It can answer questions based on the context provided by the PDFs.

3. **Supervised Agent (`supervised_agent.py`)**  
   The Supervised Agent is built using LangGraph and controls both the **SQL Agent** and **PDF Agent**. It coordinates the agents to work together efficiently and responds based on the context of the user's input.

4. **Streamlit UI (`streamlit_ui.py`)**  
   The Streamlit interface allows users to interact with the system. It sends inputs to the Supervised Agent and displays the outputs in an easy-to-use format.

5. **FastAPI (`main.py`)**  
   The FastAPI framework is used to create a backend that takes user input in text form and returns the response in audio format using ElevenLabs APIs.

6. **Audio Response via ElevenLabs API**  
   The system integrates ElevenLabs APIs for providing spoken responses in audio format. This enhances user experience by converting text-based outputs into speech.

## Features

- Perform **SQL operations** with ease through the SQL Agent.
- Extract and answer questions based on **PDF files** with the PDF Agent.
- **Supervised Agent** manages both agents, ensuring smooth coordination.
- Interactive UI using **Streamlit** for seamless user interactions.
- **Audio output** using ElevenLabs APIs, allowing the system to speak back to users.
- Scalable architecture with the ability to add more agents for other tasks.

## Installation

To get started with the project, clone the repository and install the dependencies.

```
git clone https://github.com/your-username/supervised-llm-agent.git
cd supervised-llm-agent
```
## Setup Python Environment
It’s recommended to set up a virtual environment.

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
## Install Dependencies
Install the required libraries from requirements.txt.

```
pip install -r requirements.txt
```

## Set Environment Variables
Create a .env file using the .env.example as a template and fill in the required keys.

```
cp .env.example .env
```
## Usage
# Running the Streamlit UI
To start the interactive Streamlit UI, run:

```
streamlit run streamlit_ui.py
```
This will launch a web interface where you can interact with the Supervised LLM Agent.

# Running the FastAPI Server
To start the FastAPI backend, run:

```
uvicorn main:app --reload
```
This will start the FastAPI server, and you can interact with the system via API, receiving text inputs and audio outputs.

## Using the Agents
SQL Agent: To perform SQL operations, pass SQL queries through the Streamlit interface, and the SQL Agent will process them.
PDF Agent: Upload PDFs, and ask questions about their content. The PDF Agent will extract and provide relevant answers.
Supervised Agent: This central agent coordinates the actions of both the SQL Agent and the PDF Agent based on user queries.

## Files Description
# sql_agent.py
Handles SQL operations and queries.

# pdf_agent.py
Extracts information from PDFs and responds with context-based answers.

# supervised_agent.py
Coordinates the SQL Agent and PDF Agent to work together seamlessly.

# streamlit_ui.py
Frontend interface using Streamlit to interact with the agents.

main.py
FastAPI server for handling user inputs and generating audio responses.

requirements.txt
Lists all dependencies required to run the project.

.env.example
Example environment configuration file, which needs to be filled with actual values.

Requirements
Python 3.10+
Streamlit
FastAPI
LangChain
LangGraph
ElevenLabs API
Required libraries (listed in requirements.txt)
