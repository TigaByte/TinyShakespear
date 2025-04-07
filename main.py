import torch
from torch import nn
from torch.nn import functional as F

# Parameters
block_size = 16
batch_size = 8
max_iters = 10000
eval_interval = 1000
learning_rate = 1e-3
eval_iters = 200

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)

# Load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    print("Length of the text:", len(text))

# Unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print("Vocabulary size:", vocab_size)

# Tokenizer
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Encode data
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# Train/Val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Debugging sample
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context} the target is {target}")

# Batch generation
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Batch inspection
xb, yb = get_batch('train')
print("Inputs:")
print(xb.shape)
print(xb)
print("Targets:")
print(yb.shape)
print(yb)
print("---")

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"When input is {context} the target is {target}")

# Model definition
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)

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
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# Initialize model
model = BigramLanguageModel(vocab_size).to(device)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)

# Sample generation
start = torch.zeros((1, 1), dtype=torch.long).to(device)
print("Sample Generation [befor training]")
print(decode(model.generate(start, 100)[0].tolist()))

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for step in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Final generation
print("Sample Generation [after training]")
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(context, 100)[0].tolist()))
