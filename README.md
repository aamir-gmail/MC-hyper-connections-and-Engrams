# MC-hyper-connections-and-Engrams
This repo contain experimental implementation of DeepSeek hyper connections and the Engrams research paper
# DeepSeek Engram & Hyper-Connections Implementation

[cite_start]This repository contains experimental implementations of the architectures described in the DeepSeek-AI papers **"Conditional Memory via Scalable Lookup"** [cite: 3] [cite_start]and **"Manifold-Constrained Hyper-Connections (mHC)"**[cite: 138].

The project explores treating **static memory (lookup)** as a distinct primitive alongside **dynamic computation (neural processing)**, implemented via a custom Transformer backbone.

## 📚 Core Concepts

### 1. Conditional Memory (Engram)
Standard Transformers waste computational depth reconstructing static facts (e.g., "United States", "San Francisco"). [cite_start]The **Engram Module** solves this by offloading these patterns to an $O(1)$ lookup table[cite: 11].
* [cite_start]**Mechanism:** Hashes the input suffix (N-grams) to retrieve a pre-learned embedding vector[cite: 71].
* [cite_start]**Gating:** A context-aware gate ($\alpha_t$) determines if the retrieved memory is relevant to the current hidden state[cite: 94].
* [cite_start]**Impact:** Frees up attention heads for complex reasoning and global context[cite: 16].

### 2. Manifold-Constrained Hyper-Connections (mHC)
[cite_start]Replaces standard residual connections with a multi-branch "highway" system[cite: 138].
* **Dynamic Topology:** The network learns routing coefficients per token to balance information flow between branches.
* [cite_start]**Sinkhorn Projection:** Ensures routing matrices remain doubly stochastic (stable) during training[cite: 141].

---

## 🛠️ Implementations

### Version 1 & 2: Character-Level Transformer
A lightweight implementation designed to validate the gradient flow and architectural mechanics on small datasets (*The Time Machine*, *Text8*).

* **Tokenizer:** Character-level (Vocabulary ~100)
* **Engram Config:**
    * **N-Grams:** 2-gram, 3-gram
    * **Hash Table:** Small prime sizes (4096)
    * [cite_start]**Placement:** Layers 0 & 2 ("Early + Deep" strategy) [cite: 198]
* **Observation:** Successfully demonstrated "gating" behavior, where the model learned to activate memory for frequent letter sequences (e.g., "the", "ing").

### Version 3: Sub-word BPE Transformer
A scaled-up implementation unlocking the semantic potential of Engrams using Byte Pair Encoding (BPE).

* **Tokenizer:** `tiktoken` (GPT-2, ~50k Vocab)
* **Engram Config:**
    * **Hash Table:** Scaled to **65,521** slots (Prime number) to minimize collisions.
    * **Impact:** Shifts memorization from spelling patterns (v1) to semantic concepts (idioms, named entities).
* **Optimization:**
    * **BF16 Training:** `torch.amp.autocast` for mixed-precision efficiency.
    * **Streams:** 4 Parallel mHC streams (Width 512).

---

## 🚀 Key Features

### N-Gram Hashing (Torch-Native)
[cite_start]We implemented a vectorised, GPU-friendly hashing mechanism that requires no external index construction[cite: 88].
```python
# Multi-Head XOR Hashing
mixed = current_grams * multipliers
hashed = bitwise_xor(mixed) % prime_modulo
