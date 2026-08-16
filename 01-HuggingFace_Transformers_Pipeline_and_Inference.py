# Hugging Face Transformers Pipeline:
from transformers import pipeline
# ----------------------------------------------------------------------------------------------------------------------
# Text Classification:
classifier = pipeline("zero-shot-classification")
text = "Python is widely used for artificial intelligence and machine learning."
candidate_labels = ["technology", "sports", "politics", "food"]
result = classifier(text, candidate_labels=candidate_labels)
print("Text Classification:")
print(result)
# ----------------------------------------------------------------------------------------------------------------------
# Sentiment Analysis:
sentiment = pipeline("sentiment-analysis")
result = sentiment("This movie was amazing!")
print("\nSentiment Analysis:")
print(result)
# ----------------------------------------------------------------------------------------------------------------------
# Multiple Inputs:
sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
texts = ["I love Python.",
         "This movie was terrible.",
         "Oranges are orange."]
results = sentiment(texts)
print("\nSentiment Analysis with Multiple Inputs:")
for result in results:
    print(result)
# ----------------------------------------------------------------------------------------------------------------------
# Text Generation:
generator = pipeline("text-generation", model="distilgpt2")
result = generator("Artificial intelligence will", max_new_tokens=20)
print("\nText Generation:")
print(result)
# ----------------------------------------------------------------------------------------------------------------------
# Question Answering:
question_answerer = pipeline("question-answering")
context = "Python is a programming language.\nIt was created by Guido van Rossum.\nPython was first released in 1991."
question = "Who created Python?"
result = question_answerer(question=question, context=context)
print("\nQuestion Answering:")
print(result)
# ----------------------------------------------------------------------------------------------------------------------
# Hugging Face Inference (Model & Tokenizer):
from transformers import AutoTokenizer, AutoModel
# ----------------------------------------------------------------------------------------------------------------------
# Tokenizer:
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "I love machine learning."
inputs = tokenizer(text, return_tensors="pt")
print("\nInputs Ids:")
print(inputs)
print("\nTokens:")
print(tokenizer.tokenize(text))
# ----------------------------------------------------------------------------------------------------------------------
# Model:
model = AutoModel.from_pretrained("bert-base-uncased")
print("\nModel Outputs:")
outputs = model(**inputs)
print(outputs)
print("\nLast Hidden State Shape:")
print(outputs.last_hidden_state.shape, "([batch_size, sequence_length, hidden_size])")
