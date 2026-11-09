# Voice Assistant Application

## Inputs
- intent_dataset.csv
  - Contains example texts and their corresponding intents
- user_last_command.wav
  - This file is used to temporarily store the user's current voice command and is overwritten for every new command to prevent the application from creating multiple audio files during a session

## Outputs
- evaluation.csv
  - Contains the detailed results of every command recorded during the session
- result.png
  - A bar chart showing:
    - Number of Recognized commands
    - Number of Not Recognized commands
    - Overall Recognition Rate

## Features
- Microphone-based voice input
- Speech-to-Text using OpenAI Whisper
- Text cleaning and preprocessing
- Zero-shot Intent Classification
- Confidence-based recognition
- Action execution using Intent-to-Action mapping
- Response generation
- Conversation loop
- Session evaluation
- Recognition statistics
- Visualization of Recognized and Not Recognized commands
- Random joke selection
- Single reusable audio file for all commands

## Pipeline
- Microphone → Audio → Whisper ASR → Transcript → Text Cleaning → Intent Classification → Confidence Threshold → Recognized / Not Recognized → Intent Router → Action → Response → Next Command → Goodbye

## Models
- ASR
  - OpenAI Whisper Base
    - Used for converting recorded speech into text
- Intent Classification
  - MoritzLaurer/deberta-v3-large-zeroshot-v2.0
    - Used for zero-shot intent classification

## Frameworks & Libraries
- Python
- OpenAI Whisper
- PyTorch
- Hugging Face Transformers
- Pandas
- Matplotlib
- PyAudio
- Pathlib
- Datetime
- Random
- String
- Regex
- Wave

## Concepts
- Speech-to-Text
- Automatic Speech Recognition (ASR)
- Text Cleaning
- Zero-Shot Classification
- Intent Detection
- Confidence Threshold
- Intent-to-Action Mapping
- Action Execution
- Response Generation
- Conversation Loop
- Session Evaluation
- Data Visualization

## Learning Outcomes
- Building an end-to-end ASR pipeline
- Recording audio from a microphone
- Integrating Whisper into a Python application
- Cleaning transcribed text
- Performing zero-shot intent classification
- Working with confidence scores
- Handling unrecognized commands safely
- Detecting user intent
- Integrating ASR and NLP
- Connecting intents to application actions
- Executing actions based on intent
- Building a conversation loop
- Maintaining session results
- Evaluating application performance
- Generating a visual performance report
- Building a simple Voice Assistant
