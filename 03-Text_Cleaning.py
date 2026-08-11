import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# ----------------------------------------------------------------------------------------------------------------------
text = "Hello!!! I am learning AI in 2026. Visit https://example.com to see my works. Contact at amin_ai.work@gmail.com."
print("Text      :", text)
# ----------------------------------------------------------------------------------------------------------------------
# Lowercasing:
lower = text.lower()
print("Lowercase :" ,lower, "\n")
# ----------------------------------------------------------------------------------------------------------------------
# Regex (URL Removal):
no_url = re.sub(r'https?://\S+', '', lower)
print("No URL    :" ,no_url)
# ----------------------------------------------------------------------------------------------------------------------
# Regex (Digits Removal):
no_digits = re.sub(r'\d+', '', no_url)
print("No Digits :" ,no_digits)
# ----------------------------------------------------------------------------------------------------------------------
# Regex (Email Removal):
no_email = re.sub(r'\S+@\S+', '', no_digits)
print("No Email  :" ,no_email)
# ----------------------------------------------------------------------------------------------------------------------
# Regex (Signs Removal):
no_signs = re.sub(r'[^\w\s]', '', no_email)
print("No Signs  :" ,no_signs, "\n")
# ----------------------------------------------------------------------------------------------------------------------
# NLTK (Tokenization):
sentences = sent_tokenize(text)
print("Sentences      :", sentences)
tokens = word_tokenize(no_signs)
print("Words          :", tokens)
# ----------------------------------------------------------------------------------------------------------------------
# NLTK (Stopwords Removal):
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word.lower() not in stop_words]
print("Filtered Words :", filtered_words)
# ----------------------------------------------------------------------------------------------------------------------
# NLTK (Lemmatization):
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word, pos="v") for word in filtered_words]
print("Lemmatized     :", lemmatized)
