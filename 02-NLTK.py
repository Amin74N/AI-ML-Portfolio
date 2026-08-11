from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
# ----------------------------------------------------------------------------------------------------------------------
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# ----------------------------------------------------------------------------------------------------------------------
print("---------------------------------------------------------------------------------------------------")
sentence = "Life can be challenging, tiring; but the challenges lead to transformation."
print(sentence, "\n---------------------------------------------------------------------------------------------------\n")
# ----------------------------------------------------------------------------------------------------------------------
porter = PorterStemmer()
print(porter.stem('Spiritually'))
print(porter.stem('Friendship'))
def stem_sentence(sentence):
    token_word = word_tokenize(sentence)
    StemSentence = []
    for word in token_word:
        StemSentence.append(porter.stem(word))
        StemSentence.append(" ")
    return "".join(StemSentence)
print(stem_sentence(sentence), "\n")
# ----------------------------------------------------------------------------------------------------------------------
lancaster = LancasterStemmer()
print(lancaster.stem('Spiritually'))
print(lancaster.stem('Friendship'))
def stem_sentence(sentence):
    token_word = word_tokenize(sentence)
    StemSentence = []
    for word in token_word:
        StemSentence.append(lancaster.stem(word))
        StemSentence.append(" ")
    return "".join(StemSentence)
print(stem_sentence(sentence), "\n")
# ----------------------------------------------------------------------------------------------------------------------
snowball = SnowballStemmer('english')
print(snowball.languages)
print(len(snowball.languages), "Supported Languages\n")
# ----------------------------------------------------------------------------------------------------------------------
stop = stopwords.words('english')
print(stop)
print(len(stop), "Stop-Words\n")
# ----------------------------------------------------------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
print("corpora  (Noun) :", lemmatizer.lemmatize("corpora"))                  # Noun (Default)
print("better    (Adj) :", lemmatizer.lemmatize("better", pos="a"))          # Adjective (Denoted in 'pos')
print("swimming (Verb) :", lemmatizer.lemmatize("swimming", pos="v"), "\n")  # Verb (Denoted in 'pos')
# ----------------------------------------------------------------------------------------------------------------------
print("---------------------------------------------------------------------------------------------------")
text = "Natural language processing is fascinating. It helps computers understand human English language."
print(text, "\n---------------------------------------------------------------------------------------------------\n")
# ----------------------------------------------------------------------------------------------------------------------
words = word_tokenize(text)
print("Words:", words, "\n")
sentence = sent_tokenize(text)
print("Sentences:", sentence, "\n")
# ----------------------------------------------------------------------------------------------------------------------
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print("Filtered Words:", filtered_words, "\n")
# ----------------------------------------------------------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]
print("Lemmatized:", lemmatized)
