## 📊 Understanding the Visualization

The training script generates a heatmap (e.g., `Figure_X.png`) visualizing the **Gating Scalar ($\alpha_t$)** of the Engram module. This visualization provides a window into the model's decision-making process: *Is it "remembering" or "thinking"?*

### How to Read the Heatmap
* **Dark Red (High Value $\approx$ 1.0):** **Memory Retrieval.**
    The model has identified a familiar static pattern (N-gram) in the input. The Gating mechanism is **OPEN**, actively retrieving a pre-learned vector from the hash table and injecting it into the residual stream.
* **White/Light (Low Value $\approx$ 0.0):** **Dynamic Computation.**
    The sequence is novel, ambiguous, or requires complex reasoning. The Gating mechanism is **CLOSED**, suppressing the static memory to let the Transformer backbone process the information dynamically.

### What to Expect

#### 1. The "Fact Detector"
In a trained model, you should see strong activation (Solid Red) over **Named Entities**, **Dates**, and **Facts**.
* *Example:* In the phrase *"The French Revolution started in 1789"*, the tokens `French`, `Revolution`, and `1789` will likely be dark red. The model "knows" these concepts as static units and doesn't need to compute them.

#### 2. Layer 0 vs. Deeper Layers
* **Layer 0 (The Super-Tokenizer):** The first layer often lights up for almost all valid words or common sub-word chunks. It effectively acts as a "Super-Tokenizer," patching small tokens (like characters or sub-words) into larger, meaningful concepts before deep processing begins.
* **Deeper Layers:** Activations become more sparse and selective, focusing only on complex idioms or long-range patterns that require context to resolve.

#### 3. Character-Level vs. Sub-word
* **Character Model:** You will see red blocks over common spelling patterns and suffixes (e.g., `i-n-g`, `t-i-o-n`, `t-h-e`). The memory is solving "spelling".
* **Sub-word (BPE) Model:** You will see activation over entire semantic concepts (e.g., `United States`, `machine learning`). The memory is solving "meaning".
