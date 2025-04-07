import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import softmax, dropout

# Parameters
block_size = 256
batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embed = 384
n_layers = 6
n_head = 6
dropout = 0.2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)

# Load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    print("=" * 40)
    print(f"📄 Length of the text: {len(text)}")

# Unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"🔡 Vocabulary size: {vocab_size}")
print("=" * 40)

# Tokenizer
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Encode data
data = torch.tensor(encode(text), dtype=torch.long)

# Train/Val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Batch generation
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('trill', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.trill[:T, :T] == 0 , float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, n_heads, n_embed):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.Ln1 = nn.LayerNorm(n_embed)
        self.Ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.Ln1(x))
        x = x + self.ffwd(self.Ln2(x))
        return x

# Model definition
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed//4)
        self.ffw = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.blocks = nn.Sequential(
            *[Transformer(n_head, n_embed) for _ in range(n_layers)],
            nn.LayerNorm(n_embed),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emd = self.token_embedding_table(idx)  # (B, T, C)
        pos_embed = self.pos_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emd + pos_embed
        x = self.sa_heads(x)
        x = self.ffw(x)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size[C])




        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# Initial batch for testing
xb, yb = get_batch('train')

# Initialize model
model = BigramLanguageModel().to(device)
logits, loss = model(xb, yb)

# Sample generation before training
print("\n" + "=" * 40)
print("📌 Sample Generation [Before Training]")
print("=" * 40)
start = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(start, 100)[0].tolist()))
print()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
print("-" * 40)
for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


    if iter % eval_interval == 0:
        loss = estimate_loss()
        print(f"[Iteration {iter}] 🔁 Loss:")
        print(f"  🔹 Train: {loss['train']:.4f}")
        print(f"  🔸 Val:   {loss['val']:.4f}")
        print("-" * 40)

# Final generation after training
print("\n" + "=" * 40)
print("📌 Sample Generation [After Training]")
print("=" * 40)
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(context, 500)[0].tolist()))
