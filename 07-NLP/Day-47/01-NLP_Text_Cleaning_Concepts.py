# Reading Input Text:
import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.read_csv('data/twitter_sentiments.csv')
df = df.drop(columns=['id', 'label'], axis=1)
print("Raw Text:")
print(df.head())
# ----------------------------------------------------------------------------------------------------------------------
# Converting to Lowercase:
df['clean_text'] = df['tweet'].str.lower()
print("\n----------------------------------------------------\nClean Text (Lowercase):")
print(df.head())
# ----------------------------------------------------------------------------------------------------------------------
# Removal of URLs:
import re
def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)
df['clean_text'] = df['clean_text'].apply(lambda x: remove_url(x))
print("\n----------------------------------------------------\nClean Text (No URLs):")
print(df.head())
# ----------------------------------------------------------------------------------------------------------------------
# Removal of HTML Tags:
def remove_html_tags(text):
    return re.sub(r'<.*?>', '', text)
df['clean_text'] = df['clean_text'].apply(lambda x: remove_html_tags(x))
print("\n----------------------------------------------------\nClean Text (No HTML Tags):")
print(df.head())
# ----------------------------------------------------------------------------------------------------------------------
# Removal of Punctuations:
import string
def remove_punctuations(text):
    punctuations = string.punctuation
    return text.translate(str.maketrans('', '', punctuations))
df['clean_text'] = df['clean_text'].apply(lambda x: remove_punctuations(x))
print("\n----------------------------------------------------\nClean Text (No Punctuations):")
print(df.head())
# ----------------------------------------------------------------------------------------------------------------------
# Removal of Stopwords:
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in STOPWORDS])
df['clean_text'] = df['clean_text'].apply(lambda x: remove_stopwords(x))
print("\n----------------------------------------------------\nClean Text (No Stopwords):")
print(df.head())
# ----------------------------------------------------------------------------------------------------------------------
# Removal of Frequent Words:
from collections import Counter
word_count = Counter()
for text in df['clean_text']:
    for word in text.split():
        word_count[word] += 1
FREQUENT_WORDS = set(word for (word, wc) in word_count.most_common(10))
def remove_freq_words(text):
    return " ".join([word for word in text.split() if word not in FREQUENT_WORDS])
df['clean_text'] = df['clean_text'].apply(lambda x: remove_freq_words(x))
print("\n----------------------------------------------------\nClean Text (No Frequent Words):")
print(df.head())
# ----------------------------------------------------------------------------------------------------------------------
# Removal of Rare Words:
RARE_WORDS = set(word for (word, wc) in word_count.most_common()[:-10:-1])
def remove_rare_words(text):
    return " ".join([word for word in text.split() if word not in RARE_WORDS])
df['clean_text'] = df['clean_text'].apply(lambda x: remove_rare_words(x))
print("\n----------------------------------------------------\nClean Text (No Rare Words):")
print(df.head())
# ----------------------------------------------------------------------------------------------------------------------
# Removal of Special Characters:
def remove_spl_chars(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text
df['clean_text'] = df['clean_text'].apply(lambda x: remove_spl_chars(x))
print("\n----------------------------------------------------\nClean Text (No Special Characters):")
print(df.head())
# ----------------------------------------------------------------------------------------------------------------------
# Stemming:
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])
df['stemmed_text'] = df['clean_text'].apply(lambda x: stem_words(x))
print("\n----------------------------------------------------\nStemmed Text:")
print(df.head())
# ----------------------------------------------------------------------------------------------------------------------
# Lemmatization & POS Tagging:
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
def lemmatize_words(text):
    pos_text = pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text])
df['lemmatized_text'] = df['clean_text'].apply(lambda x: lemmatize_words(x))
print("\n----------------------------------------------------\nLemmatized Text:")
print(df.head())
