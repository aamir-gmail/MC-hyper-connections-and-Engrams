# DeepSeek Engram & Hyper-Connections: PyTorch Implementation

**Author:** Aamir  
**Status:** Experimental / Active Development

This repository contains a ground-up PyTorch implementation of two cutting-edge architectures proposed by DeepSeek-AI:
1.  **Conditional Memory (Engram):** A module that replaces expensive neural computation with $O(1)$ static memory lookups for N-grams.
2.  **Manifold-Constrained Hyper-Connections (mHC):** A dynamic routing backbone that replaces standard residual connections.

The project demonstrates these concepts, scaling from a lightweight **Character-Level Transformer** up to a **Sub-word (BPE) Language Model**.

---

## 📚 Core Architectures

### 1. The Engram Module (Memory)
*Paper: "Conditional Memory via Scalable Lookup"*
Standard Transformers waste compute reconstructing static facts (e.g., "United States"). Engram offloads this to a hash table.
* **Mechanism:** Hashes input N-grams (2-gram, 3-gram) to retrieve a pre-learned static embedding.
* **Gating:** A context-aware gate ($\alpha_t$) decides whether to use the memory (Gate=1) or ignore it (Gate=0) based on the current hidden state.
* **Visual Proof:** Our visualisation tools confirm the model activates Engram (Red regions in heatmaps) for static entities like dates ("1840") or names ("Napoleon").

### 2. Hyper-Connections (Compute)
*Paper: "Manifold-Constrained Hyper-Connections (mHC)"*
Replaces the fixed residual stream $x + f(x)$ with a dynamic multi-branch highway.
* **Dynamic Routing:** The model learns coefficients to mix information between $N$ parallel latent streams.
* **Sinkhorn Projection:** We implement Sinkhorn iterations to project routing matrices onto the Birkhoff polytope (doubly stochastic), ensuring gradient stability during training.

---

## 🛠️ Model Versions & Results

### Version 1 & 2: Character-Level (Proof of Concept)
Designed to validate gradient flow and the gating mechanism.
* **Tokenizer:** Character (Vocab ~100)
* **Dataset:** *The Time Machine* -> *Text8* (30MB chunk)
* **Config:** 4 Layers, 4 Streams, Emb Dim 128.
* **Results:**
    * **Loss:** Converged to **1.26** (Excellent for Char-level).
    * **Behaviour:** Perfectly learned spelling, punctuation, and sentence structure.
    * **Engram:** Learned to memorise common suffixes ("-ing", "-tion") and frequent words.

### Version 3: Sub-word BPE (Semantic Scale-up)
Designed to unlock semantic reasoning using GPT-2 tokenization.
* **Tokeniser:** `tiktoken` (GPT-2, Vocab 50,257)
* **Dataset:** *Text8* (Full 100MB)
* **Config:** 8 Layers, 4 Streams, Emb Dim 128, **BF16 Mixed Precision**.
* **Engram Capacity:** Scaled hash tables to **65,521** slots (Prime) to handle the larger vocabulary.
* **Results (Current):**
    * **Loss:** Plateaued ~4.51 (Expected due to small `d_model=128` vs large `vocab=50k`).
    * **Generation:** Produces valid, complex English words and grammar (e.g., *"conspiracies such as taking at least repeatedly"*).
    * **Observation:** The model shifted from memorizing spelling (V1) to memorizing semantic concepts (V3).

---

## 🚀 Key Technical Features

### N-Gram Hashing (Torch-Native)
We implemented a vectorised, GPU-friendly hashing mechanism that requires no external index construction.
```python
# Multi-Head XOR Hashing
mixed = current_grams * multipliers
hashed = bitwise_xor(mixed) % prime_modulo

### Deep seek was kind enough to release a sample implementation of the Engram paper please find their original implementation here.
https://github.com/deepseek-ai/Engram
