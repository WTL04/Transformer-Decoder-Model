# Character-Level GPT Transformer in PyTorch
![transformer output](/transformer_output.png)

## Overview

This project implements a **character-level GPT-style decoder-only Transformer model** from scratch using PyTorch. Inspired by the decoder design in the seminal paper [*Attention is All You Need*](https://arxiv.org/abs/1706.03762), the model learns to generate coherent text one character at a time by predicting the next character in a sequence. The model is trained on the script of William Shakespeare's Romeo and Juliet.

## Dependencies
- Python 3.8+
- PyTorch (2.0 or higher recommended)
```
pip install torch
```
## Features
- Manual implementation of **masked multi-head self-attention**
- Decoder-only Transformer block (GPT-style)
- Character-level tokenization and generation
- Clean modular code with training and inference
- Support for CUDA acceleration

## Architecture
  - **Embedding Layers**:
  - Token embedding
  - Positional embedding

- **Multiple Transformer Blocks**:
  - Manual multi-head attention with masking
  - Feed-forward layers with ReLU
  - Layer normalization and residual connections

- **Output Layer**:
  - Final linear projection to vocabulary size
