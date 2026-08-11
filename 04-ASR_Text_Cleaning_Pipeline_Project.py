from pathlib import Path
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# ----------------------------------------------------------------------------------------------------------------------
INPUT_FILE = Path("transcripts.txt")
OUTPUT_FILE = Path("clean_transcript.txt")
# ----------------------------------------------------------------------------------------------------------------------
# 1)Reading Raw Transcript:
def raw_transcript_reader(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    print("-----------------")
    print("Raw Transcripts:")
    print("-----------------")
    print(text)
    return text
# ----------------------------------------------------------------------------------------------------------------------
# 2)Cleaning Module:
def text_cleaner(text):
    lower = text.lower()                                                 # Lowercasing
    remove_metadata = re.sub(r'file name :.*?\ntranscript:', '', lower)  # Metadata Removal
    remove_url = re.sub(r'https?://\S+', '', remove_metadata)            # URL Removal
    remove_numbers = re.sub(r'\d+', '', remove_url)                      # Digits Removal
    remove_punctuations = re.sub(r'[^\w\s]', '', remove_numbers)         # Signs Removal
    print("-----------------")
    print("Clean Transcripts:")
    print("-----------------")
    print(remove_punctuations)
    return remove_punctuations
# ----------------------------------------------------------------------------------------------------------------------
# 3)Tokenization:
def tokenizer(text):
    tokens = word_tokenize(text)
    print("-----------------")
    print("Tokens:")
    print("-----------------")
    print(tokens)
    print("Number of Tokens:", len(tokens))
    return tokens
# ----------------------------------------------------------------------------------------------------------------------
# 4)Stopwords Removal:
def stopwords_remover(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    print("-----------------")
    print("Filtered Words:")
    print("-----------------")
    print(filtered_words)
    print("Number of Filtered Words:", len(filtered_words))
    return filtered_words
# ----------------------------------------------------------------------------------------------------------------------
# 5)Lemmatization:
def lemmatizing(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word, pos="v") for word in tokens]
    print("-----------------")
    print("Lemmatized:")
    print("-----------------")
    print(lemmatized)
    print("Number of Lemmatized:", len(lemmatized))
    return lemmatized
# ----------------------------------------------------------------------------------------------------------------------
# 6)Saving Clean Transcript:
def clean_transcript_saver(filename, clean_text):
    with open(filename, "w", encoding="utf-8") as file:
        clean_text = " ".join(clean_text)
        file.write(clean_text)
# ----------------------------------------------------------------------------------------------------------------------
# 7)Main:
text = raw_transcript_reader(INPUT_FILE)
clean_text = text_cleaner(text)
tokenized_text = tokenizer(clean_text)
filtered_words = stopwords_remover(tokenized_text)
lemmatized_text = lemmatizing(filtered_words)
clean_transcript_saver(OUTPUT_FILE, lemmatized_text)
