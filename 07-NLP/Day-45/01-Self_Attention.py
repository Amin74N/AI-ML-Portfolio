import numpy as np
# ----------------------------------------------------------------------------------------------------------------------
# 1)Sentence:
sentence = "The cat drank milk"
words = sentence.split()
print("Words:")
print(words)
# ----------------------------------------------------------------------------------------------------------------------
# 2)Assumed Embeddings:
assumed_embeddings = {"The":   [1.0, 0.0],
                      "cat":   [0.0, 1.0],
                      "drank": [1.0, 1.0],
                      "milk":  [0.5, 1.0]}
X = np.array([assumed_embeddings[word] for word in words])
print("\nAssumed Embeddings:")
print(X)
# ----------------------------------------------------------------------------------------------------------------------
# 3)Assumed Query, Key, Value:
Q = X
K = X
V = X
# ----------------------------------------------------------------------------------------------------------------------
# 4)Attention Scores:
scores = Q @ K.T
print("\nAttention Scores:")
print(scores)
# ----------------------------------------------------------------------------------------------------------------------
# 5)Scale Scores:
d_k = K.shape[1]
scaled_scores = scores / np.sqrt(d_k)
print("\nScaled Scores:")
print(scaled_scores)
# ----------------------------------------------------------------------------------------------------------------------
# 6)Softmax:
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
attention_weights = softmax(scaled_scores)
print("\nAttention Weights:")
print(attention_weights)
# ----------------------------------------------------------------------------------------------------------------------
# 7)Attention Output:
output = attention_weights @ V
print("\nAttention Output:")
print(output)
# ----------------------------------------------------------------------------------------------------------------------
# 8)Focus on one word:
word_index = words.index("milk")
print("\nAttention for 'milk':")
for word, weight in zip(words, attention_weights[word_index]):
    print(f"{word} -> {weight:.3f}")
