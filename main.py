import tiktoken
import torch
import torch.nn as nn
from torch import device
from torch.nn import functional as F

# load dataset
with open('tiny shakespeare.txt', 'r', encoding='utf-8') as f:
    dataset = f.read()
print(len(dataset))

# all the unique characters in the text are group together in a list
chars = sorted(list(set(dataset)))
vocab_size = len(chars)
print(' '.join(chars))

# now we tokenize
# create a mapping from characters to integers


stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# print(encode("hii there"))
# print(decode(encode("hii there")))

###########################################################
# we can also use tiktoken here using the piece of code
# enc = tiktoken.get_encoding("cl100k_base")

# # To get the tokeniser corresponding to a specific model in the OpenAI API:
# enc = tiktoken.encoding_for_model("gpt-4")
###########################################################


# we encode the entire dataset amd store it into a torch.tensor
data = torch.tensor(encode(dataset), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# train and test split of data
# 90% is training dataset
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# block-size is the length of the chunk of the data set that we use to train the transformer at a time
block_size = 8
print(train_data[:block_size + 1])

x = train_data[:block_size]
y = val_data[1:block_size + 1]
for t in range(block_size):  # iterating over all the block-sizes of 8
    context = x[:t + 1]
    target = y[t]
    print(f"When input is: {context} the target: {target}")

# we're going to start sampling random locations from dataset to pull chunks from
# hyperparameters
torch.manual_seed(1337)
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # run on gpu if possible and use cuda
eval_iters = 200
n_embd = 384
n_head = 6  # 384/6=64 => every head is 64 dimensional
n_layer = 6  # there will be 6 layers of it
dropout = 0.2  # dropout 20% => in every forward-backward pass 20% of the intermediate calculations will be dropped to 0


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(
        device)  # if the device becomes cuda we need to make sure that when we laod the data we move it to device
    return x, y


@torch.no_grad()
# this context manager tells pytorch that we arent going to call backwards on whatever happens inside the functiom
# so pytorch doesnt have to store the intermediate variables and  can be more memory efficient
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
    # we will implement a single head of self attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size,
                             bias=False)  # bias = false so it will apply matrix multiply with some fixed weights
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # so now the size will become (B, T, hs) (read as B by T by hs) because that is the head size
        q = self.query(x)  # (B, T, hs)
        # compute the attention score or affinities
        # so far there is no communication all the queries will dot product with all the keys
        wei = q @ k.transpose(-2, -1)  # (B, T, 16) @ (B, 16, T) -------> (B,T,T) # @ means matrix multiply
        # wei = torch.zeros((T, T))  #dot product of queries with all the keys
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T) *****  exponentiate and normalize
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


# biagram language model
class BiagramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # n_embd is number of embedding dimensions
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # lm_head is language model head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both B, T tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.sa_head(x)  # apply ine head of self attention (B,T,C)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BiagramLanguageModel()
m = model.to(device)  # when we create a model we move the model parameters to device
print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
print("*********************************************************")

# create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# if you print this piece of code you will see that the model is making some progress than before
# print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
# print("****************************************************************")


# every single token emits 2 factors it emits a query and a key query vector is what we are looking for and key
# vector is what we contain we get affinities among these tokens by doing a dot product between the keys and queries
# so our query dot products with all the keys of all the other tokens so the dot product now becomes weigh if the key
# and the query are aligned they interact to a high amount, and we get to learn more about that specific token more
# than other tokens


xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('-------------------------')

for b in range(batch_size):  # batch dimension
    for t in range(block_size):  # time dimension
        context = xb[b, :t + 1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")

print(xb)  # the batch of inputs

# this is a very simple model and the tokens aren't talking to each other, so we see the very last
# character to check what comes next but now the tokens need to start talking to each other
# and figure out what comes next


# # generate from the model
# context = torch.zeros((1, 1), dtype=torch.long,
#                       device=device)  # the context that feeds into generate is created on the device
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# self attention
# seed is 1337
B, T, C = 4, 8, 2  # batch time and channels
# there are 8 tokens in a batch, and they aren't talking to each other, but we want them to
# the information only flows from previous location to current time stamp i.e, the token at
# position 3 should only talk to the ones at 1 and 2 and not 4 because it's a future token
x = torch.randn(B, T, C)
x.shape


# attention is a communication mechanism. We have a number of nodes in a directed graph. Every node has some vector
# of information, and it gets to aggregate the info via a weighted sum from all nodes that point to it this is done in
# a data dependent manner first node points to itself, second node points to itself and is pointed by node1,
# node3 is pointed by itself node1 and node2 and so on till node8 (Graph has been attached)


# this is called self attention because the keys, queries and values all come from same source x


# multi-head attention
# it is basically applying multiple attentions in parallel and concatenating the results

class MultiHeadAttention(nn.Module):
    """Multiple heads of self attention running in parallel"""

    # create desired number of heads of desired size
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)  # Dropout is a regularization technique and helps prevent over-fitting

    # run the heads in parallel into a list and concatenate the outputs
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # projection is the linear transformation of the outcome
        # of this layer
        out = self.dropout(self.proj(out))
        return out


# feed forward is an MLP (Multi Layer Perceptron)
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # we multiply by 4 because the dimensionality of input and out is 512 and
            # the dimensionaltiy of the inner layer of feed forward is 2048 so there is a multiplication of 4
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # this is the projection layer going back into the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# self attention is the communication and once the data is gathered
# thinking is done on this data, and it is done individually
# this is what feed forward does


class Block(nn.Module):
    # block intersperses communication and computation
    # communication is done by Multi-headed self attention
    # computation is done by feed forward network

    def __init__(self, n_embd, n_head):
        # n_embd = embedding dimensions and n_head = number of heads that we would like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # we fork off do communication and come back
        x = x + self.ffwd(self.ln2(x))  # we for off do computation and come back
        # the above statement are residual connections
        return x


class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # n_layer specifies the number of blocks that we will have
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


model = GPTModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M.parameters')

# create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

















# try with tiktoken
# check google's sentencepiece
# # dataset = str(dataset)
#
# # use tiktoken.get_encoding() to load an encoding by name
# encoding = tiktoken.get_encoding("cl100k_base")
# # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
# # turn text to token with encoding.encode()
# encoding.encode(dataset)
#
#
# # Count tokens by counting the length of the list returned by .encode().
#
# def num_tokens_from_string(string: str, encoding_name: str) -> int:
#     """Returns the number of tokens in a text string."""
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens
#
#
# num_tokens_from_string("dataset", "cl100k_base")
