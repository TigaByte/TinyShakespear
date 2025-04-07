import torch
from numpy.random import logistic
from torch import nn, dtype
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    print("Lenth of the Text :", len(text))

chars = sorted(list(set(text)));
vocab_size = len(chars);
print("".join(chars));
print("Vocabulary Size:", vocab_size)

# Tokeniser
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # String --> List of Integers
decode = lambda l: "".join([itos[i] for i in l])

# Encode data into torch.Tensor

data = torch.LongTensor(encode(text))
print(data.shape, data.dtype)
print(data[:1000])

# Train and Val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 16
batch_size = 8

torch.manual_seed(42)
train_data[:block_size+1]

x = train_data[:block_size] # Inputs into transformer
y = train_data[1:block_size+1] # 'targets'
for t in range (block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context} the target is {target}")


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (block_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print(f"Inputs:")
print(xb.shape)
print(xb)
print(f"Targets:")
print(yb.shape)
print(yb)
print("---")

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"When input is {context} the target is {target}")


class BigrammLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C) batch time chanels(-> vocab size )

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # Expects B C T -ln

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

m = BigrammLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
idx = torch.zeros((1, 1) , dtype=torch.long) # first character for generation input
print(decode(m.generate(idx, 100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)
for step in range(1000):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(loss.item())

print(decode(m.generate(idx, 100)[0].tolist()))