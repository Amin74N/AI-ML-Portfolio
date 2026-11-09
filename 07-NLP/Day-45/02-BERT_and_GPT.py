from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
# ----------------------------------------------------------------------------------------------------------------------
# 1)Load BERT:
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
# ----------------------------------------------------------------------------------------------------------------------
# 2)Tokenization:
text = "I love machine learning."
inputs = tokenizer(text, return_tensors="pt")
print("Tokenized Input:")
print(inputs)
print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
print("\nInput IDs:")
print(inputs["input_ids"][0].tolist())
print("\nAttention Mask:")
print(inputs["attention_mask"][0].tolist())
# ----------------------------------------------------------------------------------------------------------------------
# 3)Run BERT:
outputs = model(**inputs)
print("\nBERT Output:")
print(outputs)
print("\nOutput Type:")
print(type(outputs))
print("\nOutput Keys:")
print(outputs.keys())
last_hidden_state = outputs.last_hidden_state
print("\nLast Hidden State Shape:")
print(last_hidden_state.shape)
# ----------------------------------------------------------------------------------------------------------------------
# 4)Contextual Embedding:
sentences = ["I went to the bank to deposit money.", "I sat near the river bank."]
for text in sentences:
    print("\nSentence:", text)
    inputs = tokenizer(text, return_tensors="pt")                                   # Tokenization
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])                # Show tokens
    print("Tokens:", tokens)
    outputs = model(**inputs)                                                       # Run BERT
    hidden_states = outputs.last_hidden_state                                       # Get contextual embeddings
    bank_indices = [i for i, token in enumerate(tokens) if token == "bank"]         # Find "bank"
    for index in bank_indices:
        bank_embedding = hidden_states[0, index]
        print("Bank index:", index)
        print("Bank embedding shape:", bank_embedding.shape)
        print("First 10 values:")
        print(bank_embedding[:10])
# ----------------------------------------------------------------------------------------------------------------------
# 5)Load GPT:
gpt_model_name = "distilgpt2"
gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
gpt_model = AutoModelForCausalLM.from_pretrained(gpt_model_name)
# ----------------------------------------------------------------------------------------------------------------------
# 6)Tokenization:
prompt = "Artificial intelligence is"                                               # Prompt
inputs = gpt_tokenizer(prompt, return_tensors="pt")
print("\nPrompt:")
print(prompt)
# ----------------------------------------------------------------------------------------------------------------------
# 7)Run GPT:
generated_ids = gpt_model.generate(**inputs, max_new_tokens=20, do_sample=False)    # Generate text
generated_text = gpt_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\nGenerated Text:")
print(generated_text)
