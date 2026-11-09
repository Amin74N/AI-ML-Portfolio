from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
# ----------------------------------------------------------------------------------------------------------------------
text = "I love Python. Python is powerful. I love machine learning. I want to learn speech recognition because I love AI."
print("Original Text:")
print(text, "\n")
# ----------------------------------------------------------------------------------------------------------------------
# Word Tokenization:
tokens = word_tokenize(text.lower())
print("Word Tokenization:")
print(tokens, "\n")
# ----------------------------------------------------------------------------------------------------------------------
# Vocabulary:
vocabulary = sorted(set(tokens))
print("Vocabulary:")
print(vocabulary, "\n")
# ----------------------------------------------------------------------------------------------------------------------
# ID Assignment:
print("ID Assignment:")
for i, word in enumerate(vocabulary):
    print(word, "→", i)
print("")
# ----------------------------------------------------------------------------------------------------------------------
# One-Hot Encoding:
print("One-Hot Encoding:")
for i, word in enumerate(vocabulary):
    vector = [0] * len(vocabulary)
    vector[i] = 1
    print(word, "→", vector)
print("")
# ----------------------------------------------------------------------------------------------------------------------
# Sentence Tokenization:
sentences = sent_tokenize(text)
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
print("Sentence Tokenization:")
print(tokenized_sentences, "\n")
# ----------------------------------------------------------------------------------------------------------------------
# Embedding:
cbow_model = Word2Vec(tokenized_sentences, vector_size=15, window=2, min_count=1, sg=0)        # Word2Vec (CBOW)
vector = cbow_model.wv["python"]
print("------------------------")
print("Word2Vec (CBOW)")
print("------------------------")
print("\"Python\" Vector:")
print(vector, "\n")
similar_words = cbow_model.wv.most_similar("python", topn=5)
print("\"Python\" Similar Words:")
print(similar_words, "\n")
sg_model = Word2Vec(tokenized_sentences, vector_size=15, window=2, min_count=1, sg=1)          # Word2Vec (Skip-Gram)
vector = sg_model.wv["python"]
print("------------------------")
print("Word2Vec (Skip-Gram)")
print("------------------------")
print("\"Python\" Vector:")
print(vector, "\n")
similar_words = sg_model.wv.most_similar("python", topn=5)
print("\"Python\" Similar Words:")
print(similar_words)
