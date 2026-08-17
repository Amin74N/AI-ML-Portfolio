import pandas as pd
import string
import re
from transformers import pipeline
# ----------------------------------------------------------------------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# ----------------------------------------------------------------------------------------------------------------------
# Loading Intent Dataset:
def load_intent_dataset(dataset):
    df = pd.read_csv(dataset)
    required_columns = {"text", "intent"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
    return df
# ----------------------------------------------------------------------------------------------------------------------
# Cleaning Dataset Text:
def lowercase_text(text):                                                       # Converting to Lowercase
    return text.lower()
def remove_punctuations(text):                                                  # Removal of Punctuations
    punctuations = string.punctuation
    return text.translate(str.maketrans('', '', punctuations))
def remove_extra_whitespaces(text):                                             # Removal of Extra Whitespaces
    return re.sub(r'\s+', ' ', text).strip()
def clean_text(text):
    text = lowercase_text(text)
    text = remove_punctuations(text)
    text = remove_extra_whitespaces(text)
    return text
# ----------------------------------------------------------------------------------------------------------------------
# Intent Classification Pipeline:
def load_model():                                                               # Load Hugging Face Models
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return classifier
def classify_intent(classifier, dataset, text):                                 # Intent Classification
    candidate_labels = dataset["intent"].unique().tolist()
    result = classifier(text, candidate_labels=candidate_labels)
    predicted_intent = result["labels"][0]
    confidence = result["scores"][0]
    return predicted_intent, confidence, result
def display_classification_results(result, text, actual_intent):                # Display Classification Results:
    print("\nIntent Classification Result:")
    best_label = result["labels"][0]
    best_score = result["scores"][0]
    print(f"Text                  : {text}")
    print(f"Prediction Confidence : {best_score:.4f}")
    if best_score >= 0.50:
        print(f"Predicted Intent      : {best_label}")
        print(f"Actual Intent         : {actual_intent}")
    else:
        print("Intent Not Recognized!!")
# ----------------------------------------------------------------------------------------------------------------------
# Main:
df = load_intent_dataset('data/intent_dataset.csv')
print("Intent Dataset:")
print(df)
df["clean_text"] = df["text"].apply(clean_text)
sample = df.sample(n=1).iloc[0]
text = sample["text"]
clean_text = sample["clean_text"]
actual_intent = sample["intent"]
classifier = load_model()
predicted_intent, confidence, result = classify_intent(classifier, df, clean_text)
display_classification_results(result, text, actual_intent)
