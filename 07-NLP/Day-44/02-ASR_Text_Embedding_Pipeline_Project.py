from pathlib import Path
import json
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
# ----------------------------------------------------------------------------------------------------------------------
DATA_DIR = Path("data")
RAW_TRANSCRIPT = DATA_DIR / "transcript.txt"
CLEAN_TRANSCRIPT = DATA_DIR / "cleaned_transcript.txt"
OUTPUT_DIR = Path("output")
VOCABULARY_FILE = OUTPUT_DIR / "vocabulary.json"
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.json"
MODEL_FILE = OUTPUT_DIR / "word2vec.model"
# ----------------------------------------------------------------------------------------------------------------------
# 1)Read transcript:
def raw_transcript_reader(filename):
    text = filename.read_text(encoding="utf-8")
    return text
# ----------------------------------------------------------------------------------------------------------------------
# 2)Clean text (Regex):
def text_cleaner(text):
    lower = text.lower()                                                 # Lowercasing
    remove_numbers = re.sub(r'\d+', '', lower)                           # Digits Removal
    remove_punctuations = re.sub(r'[^\w\s]', '', remove_numbers)         # Signs Removal
    return remove_punctuations
# ----------------------------------------------------------------------------------------------------------------------
# 3)Tokenization + Stopwords Removal + Lemmatization::
def sentences_preprocessor(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        filtered_words = [word for word in tokens if word not in stop_words]
        lemmatized = [lemmatizer.lemmatize(word, pos="v") for word in filtered_words]
        if lemmatized:
            clean_sentences.append(lemmatized)
    return clean_sentences
# ----------------------------------------------------------------------------------------------------------------------
# 4)Save cleaned text:
def clean_transcript_saver(filename, clean_sentences):
    with open(filename, "w", encoding="utf-8") as file:
        for sentence in clean_sentences:
            file.write(" ".join(sentence) + "\n")
# ----------------------------------------------------------------------------------------------------------------------
# 5)Build vocabulary:
def vocabulary_builder(clean_sentences):
    unique_words = sorted(set(word for sentence in clean_sentences for word in sentence))
    vocabulary = {word: i for i, word in enumerate(unique_words)}
    return vocabulary
# ----------------------------------------------------------------------------------------------------------------------
# 6)Save vocabulary:
def vocabulary_saver(filename, vocabulary):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(vocabulary, file, ensure_ascii=False, indent=4)
# ----------------------------------------------------------------------------------------------------------------------
# 7)One-Hot Encoding:
def onehot_encoder(vocabulary):
    onehot_vectors = {}
    for i, word in enumerate(vocabulary):
        vector = [0] * len(vocabulary)
        vector[i] = 1
        onehot_vectors[word] = vector
    return onehot_vectors
# ----------------------------------------------------------------------------------------------------------------------
# 8)Train Word2Vec:
def word2vec_trainer(clean_sentences):
    model = Word2Vec(clean_sentences, vector_size=100, window=5, min_count=1, sg=1)
    return model
# ----------------------------------------------------------------------------------------------------------------------
# 9)Save model:
def word2vec_model_saver(file_name, model):
    model.save(str(file_name))
# ----------------------------------------------------------------------------------------------------------------------
# 10)Load model and test:
def word2vec_model_loader(filename):
    model = Word2Vec.load(str(filename))
    return model
# ----------------------------------------------------------------------------------------------------------------------
# 11)Extract embeddings:
def embeddings_extractor(model):
    words = ["python", "learn", "machine"]
    embeddings = {}
    for word in words:
        if word in model.wv:
            embeddings[word] = model.wv[word]
    return embeddings
# ----------------------------------------------------------------------------------------------------------------------
# 12)Save embeddings:
def embeddings_saver(file_name, embeddings):
    embeddings_json = {word: vector.tolist() for word, vector in embeddings.items()}
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(embeddings_json, file, ensure_ascii=False, indent=4)
# ----------------------------------------------------------------------------------------------------------------------
# 13)Find similar words:
def similar_words_finder(model):
    words = ["python", "learn", "machine"]
    similar_words = {}
    for word in words:
        if word in model.wv:
            similar_word = model.wv.most_similar(word, topn=5)
            print(f"{word}'s Similar Words:\n {similar_word} \n")
            similar_words[word] = similar_word
    return similar_words
# ----------------------------------------------------------------------------------------------------------------------
# 14)Main:
text = raw_transcript_reader(RAW_TRANSCRIPT)
clean_text = text_cleaner(text)
clean_sentences = sentences_preprocessor(clean_text)
clean_transcript_saver(CLEAN_TRANSCRIPT, clean_sentences)
vocabulary = vocabulary_builder(clean_sentences)
vocabulary_saver(VOCABULARY_FILE, vocabulary)
onehot_vectors = onehot_encoder(vocabulary)
words = ["python", "learn", "machine"]
for word in words:
    print(f"{word}'s Vector:\n {onehot_vectors[word]} \n")
sg_model = word2vec_trainer(clean_sentences)
word2vec_model_saver(MODEL_FILE, sg_model)
loaded_model = word2vec_model_loader(MODEL_FILE)
embeddings = embeddings_extractor(loaded_model)
embeddings_saver(EMBEDDINGS_FILE, embeddings)
similar_words_finder(loaded_model)
