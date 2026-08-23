# Voice Assistant v2 (Speech-to-Intent Classification System)

A complete voice assistant pipeline built with Python, PyTorch, Whisper ASR, TF-IDF, and a neural-network-based Intent Classifier.

The system is designed to recognize user voice commands, classify them into predefined intents, execute the corresponding action, and generate a spoken response.

## Features:

- Intent Classification using a PyTorch MLP neural network
- TF-IDF based text feature extraction with unigram and bigram features
- Text preprocessing including lowercasing, punctuation removal, and whitespace normalization
- Training and comparison of multiple MLP architectures with different hidden-layer sizes
- Model evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- Automatic selection and saving of the best-performing intent classification model
- Speech-to-Text using OpenAI Whisper
- Real-time microphone input and audio recording
- Audio preprocessing including silence removal and amplitude normalization
- Intent-based action routing for:
  - Greeting
  - Time
  - Date
  - Weather
  - Timer
  - Music playback
  - Reminders
  - Jokes
  - Motivation
  - Goodbye
- Confidence-based intent rejection using a configurable threshold
- Text-to-Speech response generation using Google TTS
- Automated evaluation using predefined voice-command scenarios
- CSV-based evaluation results and performance logging
- Error handling for microphone, audio processing, ASR, classification, action, and TTS failures

## Pipeline:

The system consists of two connected pipelines: **Intent Classification Training** and **End-to-End Voice Assistant Execution**.
- The intent classification training pipeline is:
  - Intent Dataset → Text Preprocessing → Label Encoding → TF-IDF Feature Extraction → PyTorch Tensor Conversion → DataLoader → MLP Training → Validation → Model Evaluation → Model Comparison → Best Model Selection → Model Saving
- The Voice Assistant follows an end-to-end speech processing and intent classification pipeline:
  - Microphone → Audio Recording → Audio Preprocessing → Silence Removal → Audio Normalization → Whisper ASR → Speech-to-Text → Text Cleaning → TF-IDF Feature Extraction → PyTorch Intent Classification → Confidence Check → Intent Router → Action Execution → Response Generation → Google TTS → Voice Response

## System Architecture:

```text
                    ┌──────────────────┐
                    │    Microphone    │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │   Audio Record   │
                    │    PyAudio       │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Audio Processing │
                    │ Silence Removal  │
                    │ Normalization    │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │     Whisper      │
                    │       ASR        │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │  Text Cleaning   │
                    │ Lowercase        │
                    │ Punctuation      │
                    │ Whitespace       │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │     TF-IDF       │
                    │  Feature Vector  │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ PyTorch MLP      │
                    │ Intent Classifier│
                    └────────┬─────────┘
                             ▼
                 ┌─────────────────────────┐
                 │     Confidence Check    │
                 │   Threshold = 0.75      │
                 └───────────┬─────────────┘
                  ┌──────────┴──────────┐
                  ▼                     ▼
          ┌───────────────┐     ┌─────────────────┐
          │ Recognized    │     │ NOT_RECOGNIZED  │
          └───────┬───────┘     └────────┬────────┘
                  ▼                      ▼
          ┌───────────────┐     ┌─────────────────┐
          │ Intent Router │     │ Error Response  │
          └───────┬───────┘     └─────────────────┘
                  ▼
          ┌────────────────┐
          │     Action     │
          └───────┬────────┘
                  ▼
          ┌────────────────┐
          │    Response    │
          └───────┬────────┘
                  ▼
          ┌────────────────┐
          │      TTS       │
          │     gTTS       │
          └────────────────┘
```

## Technologies, Frameworks & Libraries:

- | Technology   | Purpose                                    |
  | ------------ | ------------------------------------------ |
  | Python       | Main programming language                  |
  | Pandas       | Dataset handling                           |
  | NumPy        | Numerical/audio processing                 |
  | Scikit-learn | TF-IDF, Label Encoding, Evaluation Metrics |
  | PyTorch      | Intent Classification Neural Network       |
  | Whisper      | Automatic Speech Recognition               |
  | PyAudio      | Microphone recording                       |
  | Wave         | WAV file processing                        |
  | gTTS         | Text-to-Speech                             |
  | playsound    | Audio playback                             |
  | winsound     | Music playback                             |
  | Matplotlib   | Visualization                              |
  | Seaborn      | Confusion Matrix                           |
  | pathlib      | File/path management                       |
  | logging      | Application logging                        |

## Installation:
- Install the required Python packages:
  - pip install pandas numpy matplotlib seaborn torch scikit-learn openai-whisper pyaudio gTTS playsound

## Project Structure:

```text
Voice-Assistant v2/
│
├── input/
│   ├── model_train.csv
│   ├── model_validation.csv
│   ├── assistant_test.csv
│   └── user_last_command.wav
│
├── music/
│   ├── music_01.wav
│   ├── music_02.wav
│   ├── music_03.wav
│   ├── music_04.wav
│   └── music_05.wav
│
├── output/
│   ├── best_model.pth
│   ├── assistant.log
│   ├── assistant_evaluation.csv
│   ├── assistant_result.png
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── confusion_matrix.png
│
├── 01-Intent_Classification_Model.py
├── 02-Voice_Assistant_v2.py
├── tested_scenario.txt
└── README.md
```

## Model Training:

- The training dataset (`input/ model_train.csv`) contains:
  - 969 training samples
  - 10 intent classes
- Training Distribution:
  - GET_WEATHER     102
  - GREETING         99
  - MOTIVATION       99
  - PLAY_MUSIC       99
  - SET_TIMER        98
  - GOODBYE          97
  - GET_TIME         97
  - TELL_JOKE        96
  - GET_DATE         91
  - SET_REMINDER     91
- Training configuration:
  - Epochs: 10
  - Batch Size: 32
  - Optimizer: Adam
  - Loss Function: CrossEntropyLoss
  - Random Seed: 42

## Model Validation:

- The validation dataset (`input/ model_validation.csv`) contains:
  - 300 validation samples
  - exactly 30 examples for each intent
- Model Selection:
  - | Model   | Accuracy | Precision | Recall |   F1-Score |
    | ------- | -------: | --------: | -----: | ---------: |
    | MLP_64  |   0.9900 |    0.9909 | 0.9900 | **0.9900** |
    | MLP_128 |   0.9900 |    0.9903 | 0.9900 |     0.9899 |
    | MLP_256 |   0.9900 |    0.9902 | 0.9900 |     0.9899 |
  - The best model (MLP_64) is saved as `output/best_model.pth`
- Classification Report:
  -             recision    recall  f1-score   support

    GET_DATE       1.00      1.00      1.00        30
    GET_TIME       1.00      1.00      1.00        30
    GET_WEATHER    1.00      1.00      1.00        30
    GOODBYE        1.00      1.00      1.00        30
    GREETING       1.00      1.00      1.00        30
    MOTIVATION     0.91      1.00      0.95        30
    PLAY_MUSIC     1.00      1.00      1.00        30
    SET_REMINDER   1.00      0.90      0.95        30
    SET_TIMER      1.00      1.00      1.00        30
    TELL_JOKE      1.00      1.00      1.00        30

    accuracy                            0.99       300
    macro avg       0.99      0.99      0.99       300
    weighted avg    0.99      0.99      0.99       300

## Voice Assistant Evaluation Results:

- The complete assistant was evaluated using test dataset (`input/ assistant_test.csv`) and scenarios included:
  - Valid commands
  - Different phrasings of the same intent
  - Unsupported commands
  - Silence
  - Timer commands
  - Music commands
  - Weather requests
  - Goodbye command
- The complete test scenario is documented in: `tested_scenario.txt`
- Evaluation Summary:
  - | Metric                |     Result |
    | --------------------- | ---------: |
    | Total Commands        |         40 |
    | Recognized            |         35 |
    | Not Recognized        |          5 |
    | Recognition Rate      | **87.50%** |
    | Correct Predictions   |         38 |
    | Incorrect Predictions |          2 |
    | End-to-End Accuracy   | **95.00%** |
- The detailed result of every test command is reported in: `output/ assistant_evaluation.csv`

## Visualization:

- The project generates several visualizations for analyzing both the intent classification model and the complete voice assistant.
  - Confusion Matrix
    - Shows the classification performance across all intent classes.
    - Helps identify correct predictions and misclassified intents.
    - Helps identify classes that are more difficult for the classifier to distinguish.
    - Saved to: `output/confusion_matrix.png`
  - Accuracy Curve
    - Compares training and validation accuracy across epochs.
    - Helps analyze learning progress and model convergence.
    - Can reveal possible overfitting or underfitting.
    - Saved to: `output/accuracy_curve.png`
  - Loss Curve
    - Compares training and validation loss across epochs.
    - Helps analyze optimization progress and convergence.
    - Provides insight into the relationship between training and validation performance.
    - Saved to: `output/loss_curve.png`
  - Voice Assistant Evaluation Plot
    - Provides a summary of the end-to-end evaluation.
    - Displays:
      - Recognized vs. Not Recognized commands
      - Correct vs. Incorrect predictions
      - Overall Recognition Rate
      - Overall Prediction Accuracy
      - Total number of test cases
    - Saved to: `output/assistant_result.png`

## Logging:

- The application uses Python's built-in `logging` module to record important events and diagnostic information during execution.
- The log records:
  - Command number
  - Expected intent
  - Transcribed speech
  - Cleaned text
  - Predicted intent
  - Classification confidence
  - Recognition status
  - Prediction accuracy
  - Executed action
  - Generated response
  - Microphone errors
  - Audio processing errors
  - ASR errors
  - Classification errors
  - Action execution errors
  - Text-to-Speech errors
  - Evaluation interruptions
  - Fatal application errors
- Logs are saved to: `output/assistant.log`

## Error Handling:

- The application implements exception handling across the major stages of the end-to-end pipeline:
  - Microphone Errors:
    - Microphone initialization and recording errors are captured and logged.
    - The current command is safely marked as `NOT_RECOGNIZED`.
    - The complete evaluation session can continue without terminating unexpectedly.
  - Audio Processing Errors:
    - Invalid, empty, or unusable audio data is detected during processing.
    - Silence-only input is automatically classified as `NOT_RECOGNIZED`.
    - Audio processing failures are logged.
  - ASR Errors:
    - Whisper transcription errors are caught and logged.
    - Failed transcription is safely treated as `NOT_RECOGNIZED`.
  - Classification Errors:
    - Errors during TF-IDF transformation or PyTorch inference are handled.
    - The prediction safely falls back to `NOT_RECOGNIZED`.
    - Classification failures are logged.
  - Confidence Threshold:
    - The classifier uses a configurable confidence threshold: `CONFIDENCE_THRESHOLD = 0.75`
    - Predictions below this threshold are rejected and classified as `NOT_RECOGNIZED`
    - This mechanism helps the assistant avoid confidently executing an action for an unsupported or ambiguous command.
  - Action Errors:
    - Errors during intent-specific action execution are caught and logged.
    - The application returns a safe fallback response instead of terminating the complete evaluation.
  - TTS Errors:
    - Text-to-Speech failures are handled independently.
    - A TTS failure does not interrupt the complete evaluation process.
  - Application-Level Errors:
    - The main application also handles:
      - KeyboardInterrupt
      - Fatal runtime errors
      - Unexpected exceptions
      - Logging of critical failures
    - This allows the system to fail gracefully, preserve diagnostic information, and continue the evaluation whenever possible.

## Learning Outcomes:

- Python application development
- Object-oriented neural network design using PyTorch
- Text preprocessing and normalization
- TF-IDF feature extraction for NLP
- Label encoding and multiclass classification
- Designing and training MLP neural networks
- Using Dropout and ReLU for neural network regularization
- Training and validation workflows
- Model selection based on evaluation metrics
- Accuracy, Precision, Recall, and F1-Score evaluation
- Confusion Matrix analysis
- Speech-to-Text using Whisper
- Digital audio processing with NumPy and Wave
- Microphone audio recording using PyAudio
- Silence detection and removal
- Audio amplitude normalization
- Intent-based application architecture
- Confidence thresholding and unknown-intent detection
- Text-to-Speech integration
- Logging and exception handling
- Building an end-to-end AI pipeline combining Audio Processing, ASR, NLP, Deep Learning, and Application Logic
- Designing and executing a predefined evaluation scenario for a voice-based application

## Limitations:
- Fixed Recording Duration:
  - Each command is recorded for 5 seconds.
- Limited Intent Dataset:
  - The classifier currently supports only 10 intents.
- Timer Language Understanding:
  - The current timer implementation extracts numerical digits using _"re.findall(r"\b\d+\b", command_text)"_. Therefore, commands containing written numbers such as _"fifteen seconds"_ are not directly converted into numerical values by the action function.
- Weather Is Not Live:
  - The weather intent is recognized correctly, but the current implementation does not connect to a live weather service.
- English Language:
  - The current ASR, intent dataset, and TTS pipeline are designed for English command.
- Windows Audio Playback:
  - Music playback currently uses _"winsound.PlaySound()"_ which makes this part of the implementation Windows-specific.
