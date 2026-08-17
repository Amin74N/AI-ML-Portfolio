# ASR + NLP + Intent Detection

## Inputs
- .wav files located in "audio" folder
  - contains user speech files
- intent_dataset.csv located in "data" folder
  - contains example texts and their corresponding intents

## Outputs
- evaluation.csv located in "output" folder
  - Contains the detailed evaluation results for each audio sample
- result.png located in "output" folder
  - Visualizes the number of correct and incorrect predictions and displays the final accuracy

## Pipeline
- Audio → Whisper ASR → Transcript → Text Cleaning → Zero-Shot Intent Classification → Confidence Threshold → Detected Intent / NOT_RECOGNIZED

## Models
- ASR
  - OpenAI Whisper Base
- Intent Classification
  - MoritzLaurer/deberta-v3-large-zeroshot-v2.0

## Frameworks & Libraries
- Python
- OpenAI Whisper
- Hugging Face Transformers
- PyTorch
- Pandas
- Matplotlib
- Regex
- pathlib

## Concepts
- Automatic Speech Recognition (ASR)
- Speech-to-Text
- Audio-to-Text Pipeline
- Text Preprocessing
- Text Cleaning
- Lowercasing
- Punctuation Removal
- Whitespace Normalization
- Natural Language Processing (NLP)
- Zero-Shot Text Classification
- Intent Classification
- Candidate Labels
- Confidence Score
- Confidence Thresholding
- Batch Processing
- Classification Accuracy
- Model Comparison
- End-to-End AI Pipeline

## Learning Outcomes
- Building an end-to-end ASR pipeline
- Using Whisper for speech recognition
- Cleaning ASR-generated text
- Performing zero-shot text classification
- Detecting user intent
- Working with confidence scores and thresholds
- Batch processing audio files
- Evaluating classification accuracy
- Comparing different NLP models
- Combining ASR and NLP into a single AI pipeline
