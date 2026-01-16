""" This script combines two very hot topics from Deep Seek
   MC Hyper connections and Engrams. This character-level transformer implementation is
   simple to run focuses on the core concepts of this research paper, while lightweight to
    run on my consumer-grade hardware. The script is self-contained with all dependencies included."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import requests
import os
import matplotlib

# Force standard window instead of PyCharm interactive mode
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# Required for downloading dataset
import zipfile
import io

# --- Configuration ---
BATCH_SIZE = 32
BLOCK_SIZE = 128
LEARNING_RATE = 3e-4
MAX_ITERS = 36000  # Reduced for demo speed
EVAL_INTERVAL = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DROPOUT = 0.05
WEIGHT_DECAY = 0.1
GRAD_CLIP = 0.8

# Initial Warmup Setting (Will be overridden if loading model)
WARMUP_STEPS = 600

# mHC Specifics
N_STREAMS = 4
EMBED_DIM = 128  # (Total Width = 128 * 4 = 512)
N_LAYERS = 4
SINKHORN_ITERS = 20

# Engram Specifics
ENGRAM_ACTIVE = True
ENGRAM_LAYERS = [0, 2]  # Apply Engram to 1st and 3rd layers
# Smaller hash tablle sutiable for charter level transformer.
ENGRAM_VOCAB_SIZES = [4096, 4096]  # Size of hash tables for 2-gram and 3-gram
ENGRAM_EMBED_DIM = 64  # Dimension of retrieved static embeddings
MAX_NGRAM = 3
# Chage this to save your model as required.
MODEL_SAVE_PATH = './model/engrams_hyper_connection_v3.pt'

torch.manual_seed(1337)

# Cleaned input string (No newlines, only valid dictionary chars)
# We will be using this text to test the memory retention capacity of our ngrams.
TEXT_CHUNK = """within french working class movements and his followers were active in the revolution of one eight 
four eight in france"""


# --- Part 1: Engram Components ---

class ShortConv(nn.Module):
    def __init__(self, hidden_size, kernel_size=4, dilation=1, hc_mult=4):
        super().__init__()
        self.hc_mult = hc_mult
        total_channels = hidden_size * hc_mult

        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,  # Depthwise
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
        self.act = nn.SiLU()

    def forward(self, x):
        B, T, G, C = x.shape
        normed_chunks = [self.norms[i](x[:, :, i, :]) for i in range(G)]
        x_norm = torch.cat(normed_chunks, dim=-1)

        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        y = self.act(y_bct)
        return y.transpose(1, 2).view(B, T, G, C)


class NgramHashMapping(nn.Module):
    def __init__(self, vocab_sizes, max_ngram, num_heads=2, seed=42):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.max_ngram = max_ngram
        self.num_heads = num_heads

        self.register_buffer(
            'multipliers',
            torch.randint(1, 10000, (max_ngram - 1, num_heads, max_ngram))
        )
        self.register_buffer(
            'modulos',
            torch.tensor([v for v in vocab_sizes]).view(-1, 1)
        )

    def forward(self, input_ids):
        B, T = input_ids.shape
        padded = F.pad(input_ids, (self.max_ngram - 1, 0), value=0)
        windows = padded.unfold(dimension=1, size=self.max_ngram, step=1)
        all_hashes = []

        for n_idx in range(self.max_ngram - 1):
            ngram_len = n_idx + 2
            current_grams = windows[:, :, -ngram_len:]
            mults = self.multipliers[n_idx, :, :ngram_len]
            mixed = current_grams.unsqueeze(2) * mults.unsqueeze(0).unsqueeze(0)

            hashed = mixed[..., 0]
            for k in range(1, ngram_len):
                hashed = torch.bitwise_xor(hashed, mixed[..., k])

            hashed = hashed % self.modulos[n_idx]
            all_hashes.append(hashed)

        return torch.cat(all_hashes, dim=2)


class EngramModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.hasher = NgramHashMapping(ENGRAM_VOCAB_SIZES, MAX_NGRAM)
        total_slots = sum(ENGRAM_VOCAB_SIZES) * self.hasher.num_heads
        self.embedding = nn.Embedding(total_slots, ENGRAM_EMBED_DIM)

        total_ngrams = len(ENGRAM_VOCAB_SIZES) * self.hasher.num_heads
        input_dim = total_ngrams * ENGRAM_EMBED_DIM

        self.val_proj = nn.Linear(input_dim, EMBED_DIM)
        self.key_projs = nn.ModuleList([
            nn.Linear(input_dim, EMBED_DIM) for _ in range(N_STREAMS)
        ])
        self.norm_q = nn.ModuleList([nn.RMSNorm(EMBED_DIM) for _ in range(N_STREAMS)])
        self.norm_k = nn.ModuleList([nn.RMSNorm(EMBED_DIM) for _ in range(N_STREAMS)])
        self.conv = ShortConv(EMBED_DIM, hc_mult=N_STREAMS)

    def forward(self, x, input_ids):
        hash_ids = self.hasher(input_ids)
        offsets = torch.tensor([0] + [v for v in ENGRAM_VOCAB_SIZES for _ in range(self.hasher.num_heads)],
                               device=x.device)
        offsets = torch.cumsum(offsets, dim=0)[:-1]
        hash_ids = hash_ids + offsets

        emb = self.embedding(hash_ids).flatten(start_dim=2)
        v_base = self.val_proj(emb)

        gates = []
        for i in range(N_STREAMS):
            k = self.norm_k[i](self.key_projs[i](emb))
            q = self.norm_q[i](x[:, :, i, :])
            score = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(EMBED_DIM)
            g = torch.sigmoid(score)
            gates.append(g)

        gates = torch.stack(gates, dim=2)
        v_gated = v_base.unsqueeze(2) * gates
        y = v_gated + self.conv(v_gated)
        return y


# --- Part 2: mHC Components ---

class SinkhornProjection(nn.Module):
    def __init__(self, iterations=SINKHORN_ITERS):
        super().__init__()
        self.iterations = iterations

    def forward(self, x):
        x_safe = x - x.max(dim=-1, keepdim=True).values
        matrix = torch.exp(x_safe)
        for _ in range(self.iterations):
            matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + 1e-6)
            matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + 1e-6)
        return matrix


class MHCWrapper(nn.Module):
    def __init__(self, layer_f, dim, n_streams=N_STREAMS):
        super().__init__()
        self.layer_f = layer_f
        self.dim = dim
        self.n = n_streams
        self.total_dim = dim * n_streams

        self.coef_norm = nn.RMSNorm(self.total_dim)
        self.proj_pre = nn.Linear(self.total_dim, n_streams)
        self.proj_post = nn.Linear(self.total_dim, n_streams)
        self.proj_res = nn.Linear(self.total_dim, n_streams * n_streams)

        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))

        self.sinkhorn = SinkhornProjection()
        self.layer_norm = nn.RMSNorm(dim)

    def get_dynamic_mappings(self, x_flat):
        x_norm = self.coef_norm(x_flat)
        H_pre = torch.sigmoid(self.alpha_pre * self.proj_pre(x_norm)).unsqueeze(1)
        H_post = 2 * torch.sigmoid(self.alpha_post * self.proj_post(x_norm)).unsqueeze(1)
        h_res_logits = self.alpha_res * self.proj_res(x_norm)
        H_res = self.sinkhorn(h_res_logits.view(-1, self.n, self.n))
        return H_pre, H_post, H_res

    def forward(self, x):
        B, T, n, C = x.shape
        x_flat = x.view(B * T, -1)
        H_pre, H_post, H_res = self.get_dynamic_mappings(x_flat)
        x_per_token = x.view(B * T, n, C)

        x_aggregated = torch.einsum('bjn, bnc -> bjc', H_pre, x_per_token).squeeze(1)
        x_for_layer = x_aggregated.view(B, T, C)
        x_layer_out = self.layer_f(self.layer_norm(x_for_layer))
        x_layer_out_flat = x_layer_out.view(B * T, C)

        x_branch = torch.einsum('bjn, bc -> bnc', H_post.transpose(1, 2), x_layer_out_flat)
        x_highway = torch.einsum('bnm, bmc -> bnc', H_res, x_per_token)
        return (x_highway + x_branch).view(B, T, n, C)


# --- Part 3: Transformer Components ---

class CausalSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.num_heads = 4
        self.head_dim = dim // 4
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)
        self.register_buffer("bias", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                             .view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x): return self.net(x)


# --- Part 4: Unified Block & Model ---

class MHCBlock(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.engram = None
        if ENGRAM_ACTIVE and layer_id in ENGRAM_LAYERS:
            print(f"Adding Engram to Layer {layer_id}")
            self.engram = EngramModule()

        self.attn = MHCWrapper(CausalSelfAttention(EMBED_DIM), dim=EMBED_DIM)
        self.mlp = MHCWrapper(MLP(EMBED_DIM), dim=EMBED_DIM)

    def forward(self, x, input_ids):
        # 1. Engram (Conditional Memory)
        if self.engram is not None:
            x = x + self.engram(x, input_ids)

        # 2. mHC Attention (Conditional Computation)
        x = self.attn(x)

        # 3. mHC MLP
        x = self.mlp(x)
        return x


class MHCEngramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM * N_STREAMS)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM * N_STREAMS)

        self.layers = nn.ModuleList([
            MHCBlock(i) for i in range(N_LAYERS)
        ])

        self.final_norm = nn.RMSNorm(EMBED_DIM * N_STREAMS)
        self.head = nn.Linear(EMBED_DIM * N_STREAMS, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=DEVICE))
        x = x.view(B, T, N_STREAMS, EMBED_DIM)

        for layer in self.layers:
            x = layer(x, idx)

        x = x.view(B, T, -1)
        x = self.final_norm(x)
        logits = self.head(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- LR Scheduler ---
def get_lr(it):
    # 1) Linear Warmup
    if it < WARMUP_STEPS:
        return LEARNING_RATE * (it + 1) / (WARMUP_STEPS + 1)
    # 2) If it > MAX_ITERS, return min lr
    if it > MAX_ITERS:
        return LEARNING_RATE * 0.1
    # 3) Cosine Decay
    decay_ratio = (it - WARMUP_STEPS) / (MAX_ITERS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LEARNING_RATE * 0.1 + coeff * (LEARNING_RATE * 0.9)


# --- Visualization ---
def visualize_engram_gates(model, text_segment, layer_idx=0):
    """
    Visualizes the Engram gating values.
    """
    model.eval()

    # 1. Prepare Input
    ctx_len = BLOCK_SIZE
    enc = [stoi[c] for c in text_segment]
    input_ids = torch.tensor(enc, dtype=torch.long).unsqueeze(0).to(DEVICE)

    if input_ids.size(1) < ctx_len:
        padding = torch.zeros((1, ctx_len - input_ids.size(1)), dtype=torch.long, device=DEVICE)
        input_ids = torch.cat([padding, input_ids], dim=1)
    else:
        input_ids = input_ids[:, -ctx_len:]

    engram_module = model.layers[layer_idx].engram
    if engram_module is None:
        print(f"No Engram at layer {layer_idx}")
        return

    with torch.no_grad():
        # Get embeddings from backbone up to this point (approximation for demo)
        x = model.token_embedding(input_ids) + model.position_embedding(
            torch.arange(input_ids.size(1), device=DEVICE))
        x = x.view(1, input_ids.size(1), N_STREAMS, EMBED_DIM)

        # Hash and Retrieve
        hash_ids = engram_module.hasher(input_ids)
        offsets = torch.tensor([0] + [v for v in ENGRAM_VOCAB_SIZES for _ in range(engram_module.hasher.num_heads)],
                               device=DEVICE)
        offsets = torch.cumsum(offsets, dim=0)[:-1]
        hash_ids = hash_ids + offsets
        emb = engram_module.embedding(hash_ids).flatten(start_dim=2)

        # Calculate Gates
        gates_list = []
        for i in range(N_STREAMS):
            k = engram_module.norm_k[i](engram_module.key_projs[i](emb))
            q = engram_module.norm_q[i](x[:, :, i, :])
            score = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(EMBED_DIM)
            g = torch.sigmoid(score)
            gates_list.append(g)

        avg_gate = torch.stack(gates_list, dim=2).mean(dim=2).squeeze(-1).squeeze(0)  # (T,)

    # 3. Plot
    tokens = [itos[i.item()] for i in input_ids[0]]
    activations = avg_gate.cpu().numpy()

    valid_len = len(text_segment)
    tokens = tokens[-valid_len:]
    activations = activations[-valid_len:]

    plt.figure(figsize=(15, 2))
    sns.heatmap([activations], xticklabels=tokens, yticklabels=['Gate'], cmap="Reds", cbar=True)
    plt.title(f"Engram Activation Pattern (Layer {layer_idx})")
    plt.show()


# --- Training Execution ---
if __name__ == "__main__":
    def get_data(filename="./train_data/text8_chunk.txt"):
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        print("Downloading Text8 (~31MB Compressed)...")
        url = "http://mattmahoney.net/dc/text8.zip"
        r = requests.get(url)

        print("Extracting and slicing first 30MB...")
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open('text8') as f:
                text = f.read(30_000_000).decode('utf-8')

        # Check folder
        os.makedirs("./train_data", exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text)
        return text


    text = get_data()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]


    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix]).to(DEVICE)
        y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix]).to(DEVICE)
        return x, y


    print(f"Initializing Engram-mHC Model (Layers={N_LAYERS}, Streams={N_STREAMS})...")

    # Model Loading Logic with Dynamic Warmup Adjustment
    model_path = MODEL_SAVE_PATH
    os.makedirs("./model", exist_ok=True)

    if os.path.exists(model_path):
        model = MHCEngramLM(len(chars)).to(DEVICE)
        model = torch.load(model_path, weights_only=False).to(DEVICE)
        print("Model Loaded! Setting WARMUP_STEPS = 0")
        WARMUP_STEPS = 0  # <--- Warmup steps set to 0 if loading a pre-existing model
    else:
        model = MHCEngramLM(len(chars)).to(DEVICE)
        print("Model Not Loaded. Starting fresh with WARMUP_STEPS = 600")
        WARMUP_STEPS = 600

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for iter in range(MAX_ITERS):

        # --- NEW: APPLY SCHEDULER ---
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # ----------------------------

        if iter % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                losses = torch.zeros(10)
                for k in range(10):
                    X, Y = get_batch('val')
                    _, loss = model(X, Y)
                    losses[k] = loss.item()
            print(f"Step {iter}: Val Loss {losses.mean():.4f} | LR: {lr:.2e}")
            model.train()

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    print("\n--- Generating Text ---")
    ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(ctx, 300)[0].tolist()))

    print("Visualizing Gates...")
    # Using the cleaned string constant defined at top
    # If test has newline character then lease do this
    # TEXT_CHUNK.replace('\n', ' ')
    try:
        torch.save(model, model_path)
    except Exception as e:
        print(f"Error saving model: {e}")
    try:
        visualize_engram_gates(model, TEXT_CHUNK.replace('\n', ' '), layer_idx=2)
    except Exception as e:
        print(f"Error visualizing gates: {e}")
