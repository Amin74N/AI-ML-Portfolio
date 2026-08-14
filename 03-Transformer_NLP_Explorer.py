from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from pathlib import Path
# ----------------------------------------------------------------------------------------------------------------------
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
# ----------------------------------------------------------------------------------------------------------------------
# 1)Load BERT:
bert_model_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name)
print("BERT loaded successfully!\n")
# ----------------------------------------------------------------------------------------------------------------------
# 2)Load GPT:
gpt_model_name = "distilgpt2"
gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
gpt_model = AutoModelForCausalLM.from_pretrained(gpt_model_name)
print("GPT loaded successfully!\n")
# ----------------------------------------------------------------------------------------------------------------------
# 3)Run BERT:
user_text = input("Enter a sentence for BERT: ")
bert_inputs = bert_tokenizer(user_text, return_tensors="pt")
bert_outputs = bert_model(**bert_inputs)
embedding = bert_outputs.last_hidden_state
print("\nBERT Result:")
tokens = bert_tokenizer.convert_ids_to_tokens(bert_inputs["input_ids"][0])
print("\nTokens:")
print(tokens)
input_ids = bert_inputs["input_ids"][0].tolist()
print("\nInput IDs:")
print(input_ids)
attention_mask = bert_inputs["attention_mask"][0].tolist()
print("\nAttention Mask:")
print(attention_mask)
print("\nLast Hidden State Shape:")
print(embedding.shape, "=> ([Batch, Sequence Length, Hidden Size])")
# ----------------------------------------------------------------------------------------------------------------------
# 4)Run GPT:
user_prompt = input("\nEnter a prompt for GPT: ")
gpt_inputs = gpt_tokenizer(user_prompt, return_tensors="pt")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
generated_ids = gpt_model.generate(**gpt_inputs, max_new_tokens=20, do_sample=False, pad_token_id=gpt_tokenizer.eos_token_id)
generated_text = gpt_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\nGPT Result:")
print(generated_text)
# ----------------------------------------------------------------------------------------------------------------------
# 5)Save BERT Output:
bert_output_file = OUTPUT_DIR / "bert_output.txt"
with open(bert_output_file, "w", encoding="utf-8") as file:
    file.write("BERT Tokenization:\n\n")
    file.write(f"Text: {user_text}\n")
    file.write(f"Tokens: {tokens}\n")
    file.write(f"Input IDs: {input_ids}\n")
    file.write(f"Attention Mask: {attention_mask}\n\n")
    file.write("BERT Output:\n\n")
    file.write(f"Last Hidden State Shape: {embedding.shape} => ([Batch, Sequence Length, Hidden Size])")
# ----------------------------------------------------------------------------------------------------------------------
# 6)Save GPT Output:
gpt_output_file = OUTPUT_DIR / "gpt_output.txt"
with open(gpt_output_file, "w", encoding="utf-8") as file:
    file.write("GPT Text Generation:\n\n")
    file.write(f"Prompt: {user_prompt}\n")
    file.write(f"Generated Text: {generated_text}")
