# William Luu
# CECS 451 Transformer Decoder Model
# Dependencies: pip install torch

import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt
from collections import Counter

"""
Data Preprocessing
"""
def read_file_utf8(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# reading text file with utf8 encoding
file_path = 'input.txt'
text = read_file_utf8(file_path)

# sorted list of unique characters from input
vocabulary = sorted(list(set(text)))

# string to integer mapping`
stoi = {ch: i for i, ch in enumerate(vocabulary)}

# integer to string mapping
itos = {i: ch for i, ch in enumerate(vocabulary)}

"""
Encode Decode Functions
"""
def encode(s):
    output = []
    for char in s:
        output.append(stoi[char])
    return output

def decode(i):
    output = ""
    for j in i:
        output += itos[j]
    return output

"""
Tokenization and Training-Val Split (90/10)
"""
# character-level tokenization
tokens = encode(text)
tensor = torch.tensor(tokens) # 1D tensor of size [1115394] elements

split_idx = int(0.9 * len(tensor))

# split tensor into 90/10 training and validation sets
train_data = tensor[:split_idx]
val_data = tensor[split_idx:]


"""
Get Batch Function 
"""
batch_size = 64 # num of blocks
block_size = 256 # num of chars in a sequence

def get_batch(split):
    # returns random int tensor of size batch_size
    rand_indices = torch.randint(len(split) - block_size, (batch_size,))

    # returns tensors of size (batch_size, block_size)
    x = torch.stack([split[i:i+block_size] for i in rand_indices])

    # target sequence shifted by 1
    y = torch.stack([split[i+1:i+block_size+1] for i in rand_indices])
    return x, y 

"""
Model Architecture : Embedding Layer
"""
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

    def forward(self, x):
        B, T = x.shape  # x is (batch, seq_len) of token indices
        tok_emb = self.token_embedding(x)         # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))  # (T, n_embd)
        return tok_emb + pos_emb  # (B, T, n_embd)

"""
Model Architecture : Transformer Block
"""
class Transformer_Block(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()

        # n_heads hyperparameter
        self.n_head = n_head

        # embedding layer
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (num of embeddings, dimensions)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # linear layers for multi_head attention
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.output_proj = nn.Linear(n_embd, n_embd)

        # feed forward layer
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

        # norm layers
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        # dropout
        self.dropout = nn.Dropout(dropout)


    def masked_attention(self, Q, K, V):
        """
        Q, K, V: (batch, n_head, seq_len, head_dim)
        Returns: (batch, n_head, seq_len, head_dim)
        """

        # compute scores
        d_k = Q.shape[-1]
        scores = (Q @ K.transpose(-2, -1)) / sqrt(d_k)

        # creates a diagonal of -inf values to mask future comparisions
        T = scores.size(-1)
        mask = torch.triu(torch.ones(T, T), diagonal=1).to(Q.device).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # apply softmax
        weights = F.softmax(scores, dim=-1)
        return weights @ V # returns tensor

    def multihead_attention(self, x):
        """
        x: (batch, tokens, embeddings)
        Returns: (batch, tokens, embeddings)
        """
        # batch(num of sequences), tokens(num of tokens), embeddings(size of vector for each token)
        B, T, E = x.shape

        # divide embeddings for parallel attention
        head_dim = E // self.n_head

        # passing Q, K, V through linear layers to transform their shapes
        Q = self.q_proj(x) # (B, T, E)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # using view to add n_head as a dimension
        Q = Q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_head, head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_head, head_dim).transpose(1, 2)

        output = self.masked_attention(Q, K, V)                     # (B, n_head, T, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, T, E)  # (B, T, E)

        # go through final linear layer
        output = self.output_proj(output)

        # dropout for regularlization
        output = self.dropout(output)
        return output


    def forward(self, x):
        # masked multihead attention -> add & norm
        attn_out = self.multihead_attention(x)
        x = self.ln1(x + attn_out)

        # feedforward -> dropout -> add & norm
        ff_out = self.feed_forward(x)
        ff_out = self.dropout(ff_out)
        x = self.ln2(x + ff_out)
        return x


"""
Model Architecture :GPTLanguageModel 
"""
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        # embedding layer
        self.embedding = TransformerEmbedding(vocab_size, n_embd, block_size)

        # transformer blocks
        self.blocks = nn.Sequential(*[
            Transformer_Block(
                vocab_size=vocab_size,         
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,               
                block_size=block_size,
                dropout=dropout
            )
            for _ in range(n_layer)
        ])

        # norm and feedforward layer 
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        targets: (B, T) optional ground truth for loss
        """
        x = self.embedding(idx)  # (B, T, n_embd)
        x = self.blocks(x)       # (B, T, n_embd)
        x = self.ln_f(x)         # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            return logits, None
        
        # reshape logits and targets for cross entropy
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss    


    def generate(self, idx, max_new_tokens):
        """
        idx: (B, T) starting tokens
        Returns: (B, T + max_new_tokens) — generated sequence
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # crop to block size
            logits, _ = self.forward(idx_cond)    # (B, T, vocab_size)
            next_token_logits = logits[:, -1, :]  # last time step
            probs = F.softmax(next_token_logits, dim=-1)  # (B, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)  # append to sequence
        return idx

"""
Initalize Model with Hyperparameters 
"""
model = GPTLanguageModel(
    vocab_size=len(vocabulary),
    n_embd=384,
    n_head=6,
    n_layer=6,
    block_size=256,
    dropout=0.2
)


"""
Setting Tensor and Model to Device (GPU or CPU)
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# setting training and validation data to use device
train_data = tensor.to(device) 
val_data = tensor.to(device)
model = model.to(device)

print(train_data.device, val_data.device, next(model.parameters()).device)

"""
Training : Initalize estimate loss function for train and val sets 
"""
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        for _ in range(5): # run 5 testing batches
            x, y = get_batch(data)
            _, loss = model(x, y)
            losses[split] += loss.item()
        losses[split] /= 5 # get the average out of 5 runs
    model.train()
    return losses


"""
Training : Training Loop with 5000 iterations
"""
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

max_iters = 5000

print(f"training start with {max_iters} iterations...")

for step in range(max_iters):
    xb, yb = get_batch(train_data)
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    
    if step % 100 == 0:
        losses = estimate_loss()
        print(f"Step: {step}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

"""
Testing text generation
"""
# generate sample output
model.eval()
context = torch.zeros((1, 1), dtype=torch.long).to(device)  # start token (e.g. 0)
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print("Sample output:", decode(generated))
