# Transformer NLP Explorer

## Outputs
- BERT Output
  - Input text
  - Tokens
  - Input IDs
  - Attention Mask
  - Last Hidden State Shape
- GPT Output
  - Input prompt
  - Generated text

## Frameworks & Libraries
- Python
- NumPy
- PyTorch
- Hugging Face Transformers
- BERT
- DistilGPT2

## Concepts
- Attention
  - Attention allows a model to assign different importance to different tokens when creating a representation
- Self-Attention
  - Each token can attend to other tokens within the same sequence
- Query / Key / Value
  - Query → What information am I looking for?
  - Key   → What information do I contain?
  - Value → What information should actually be passed?
- Contextual Embeddings
  - A token's representation can change depending on the context in which it appears
- Transformer
  - A Transformer uses mechanisms such as Self-Attention and Feed Forward Networks to process sequences and build contextual representations
- Encoder
  - The Encoder processes the input sequence and creates contextual representations of the tokens
- Decoder
  - The Decoder generates or processes output tokens based on the available context
- Positional Encoding
  - Positional Encoding provides information about the position of each token because the Self-Attention mechanism does not inherently encode token order
- BERT
  - BERT is an Encoder-only Transformer primarily designed for understanding and representing text
- GPT
  - GPT is a Decoder-only Transformer designed for autoregressive text generation

## Learning Outcomes
- Understanding the basic idea of Attention
- Understanding the Transformer architecture
- Explaining BERT
- Explaining GPT
- Using Hugging Face Transformer models
- Understanding how Transformers can be used after ASR
