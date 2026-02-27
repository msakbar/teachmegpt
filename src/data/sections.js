export const DEPTH_CONFIG = [
  { key: "picture", label: "Picture" },
  { key: "simple", label: "Simple" },
  { key: "technical", label: "Technical" },
  { key: "deep", label: "Deep Dive" },
  { key: "source", label: "Source" },
];

export const SECTIONS = [
  {
    id: "whats-a-gpt",
    title: "What Is a GPT?",
    image: "/image1.webp",
    analogy: `Sam runs a lemonade stand. Every day, she watches the line of customers and tries to guess what the NEXT person will order based on what the people before them ordered. \u201cThe last 3 people ordered lemon, lemon, strawberry\u2026 so the next one is probably lemon.\u201d That\u2019s a GPT. It looks at a sequence of things and predicts what comes next.`,
    simple: `**A GPT predicts the next token. That single operation, repeated, produces language.**

A GPT is a guessing game. You show it some letters, and it guesses the next letter. Show it "Emm" and it guesses "a". That\u2019s the whole trick. Do this really fast, one letter at a time, and you get words. Do it with enough practice, and the guesses get really good.

This file teaches a tiny GPT to guess letters in names. You show it the start of a name, and it guesses what letter comes next until it decides the name is done.`,
    technical: `A GPT is a next-token predictor. A "token" here is a single character (a, b, c...). The model takes a sequence of characters it\u2019s already seen and outputs a probability for every possible next character.

This file is a complete GPT in ~140 lines of Python. It downloads a list of real names (Emma, Olivia, Liam...), trains a tiny neural network to predict the next character in each name, and then generates brand-new names that sound plausible.

The file has zero dependencies \u2014 no PyTorch, no TensorFlow. Every operation is written out by hand so you can see exactly what happens.`,
    deep: `This is Karpathy\u2019s "microgpt" \u2014 a pedagogical artifact that implements the full GPT algorithm (training + inference) in pure Python with only standard library imports. Minor architectural differences from GPT-2: RMSNorm instead of LayerNorm, no biases, squared ReLU instead of GELU.

The model is character-level, trained on the "names" dataset. Default config: n_embd=16, n_layer=1, n_head=4, block_size=8 \u2014 roughly 5,000 parameters. It implements autoregressive language modeling with a KV cache for sequential inference, an autograd engine for backpropagation, and an Adam optimizer.

The thesis: "The contents of this file is everything algorithmically needed to train a GPT. Everything else is just efficiency." GPT-4\u2019s 1.8T parameters run the same core loop.`,
    source: `"""
The most atomic way to train and inference a GPT LLM
in pure, dependency-free Python.
Differences from GPT-2: rmsnorm instead of layer norm,
no biases, square ReLU instead of GeLU.
The contents of this file is everything algorithmically
needed to train a GPT. Everything else is just efficiency.
Art project by @karpathy.
"""

import os, math, random, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_embd', type=int, default=16)
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--block_size', type=int, default=8)
parser.add_argument('--num_steps', type=int, default=1000)
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-2)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()`,
  },
  {
    id: "tokenizer",
    title: "The Tokenizer",
    image: "/image2.webp",
    analogy: `Sam can\u2019t do math with drink names like \u201clemonade\u201d and \u201cstrawberry.\u201d So she assigns each drink a number: lemonade=2, strawberry=3, water=4. She also has two special codes: \u201cNEW CUSTOMER APPROACHING\u201d (=0) and \u201cCUSTOMER DONE ORDERING\u201d (=1). Now she can work with numbers instead of words.`,
    simple: `**Tokens are the atoms of language models \u2014 everything the model sees or produces is a token.**

Computers can\u2019t read letters. They only understand numbers. So before we start, we make a cheat sheet:
- "a" = number 3
- "b" = number 4
- "c" = number 5
- ... and so on

We also need two special codes: one that means "a name is starting" and one that means "a name is ending."

Now the name "Emma" becomes a list of numbers: [0, 7, 15, 15, 3, 1]. The computer can work with that.`,
    technical: `The tokenizer builds two dictionaries:

stoi (string-to-integer): every unique character gets a number. "a"\u21923, "b"\u21924, etc.
itos (integer-to-string): the reverse lookup to convert numbers back to letters.

Two special tokens are added:
\u2022 BOS (Beginning of Sequence, =0) \u2014 tells the model "start predicting"
\u2022 EOS (End of Sequence, =1) \u2014 tells the model "this name is finished"

The full vocabulary is about 28 tokens (26 letters + BOS + EOS). Real GPTs like ChatGPT use ~100,000 tokens representing word fragments instead of single characters, but the idea is identical.`,
    deep: `Character-level tokenization. The vocabulary is constructed dynamically from the dataset: extract all unique characters, sort them, prepend BOS/EOS sentinel tokens. This gives a vocab_size of ~28 for the names dataset.

The tradeoff versus subword tokenizers (BPE/SentencePiece): character-level means longer sequences (more compute per word) but zero out-of-vocabulary tokens and a tiny embedding table. BPE compresses "uncomfortable" from 13 characters to ~3 tokens, dramatically reducing sequence length at the cost of a 50-100K vocabulary.

The stoi/itos mappings are the complete tokenizer. No merge rules, no byte-fallback, no normalization. This is the minimal viable tokenizer.`,
    source: `# Dataset: the names dataset (one name per line)
if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/'
        'makemore/refs/heads/master/names.txt',
        'input.txt')
with open('input.txt', 'r') as file:
    text = file.read()
docs = [line.strip() for line in text.strip().split('\\n')
        if line.strip()]
random.shuffle(docs)

# Tokenizer: character-level with BOS/EOS
chars = ['<BOS>', '<EOS>'] + sorted(list(set(''.join(docs))))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
BOS, EOS = stoi['<BOS>'], stoi['<EOS>']`,
  },
  {
    id: "autograd",
    title: "The Autograd Engine",
    image: "/image3.webp",
    analogy: `Sam guesses wrong \u2014 she predicted lemonade but the customer ordered strawberry. She wants to get better, so she traces back through her reasoning: \u201cI looked at the weather, the time of day, and how long the line was\u2026 which of those things should I pay more attention to next time?\u201d She figures out exactly how much each factor contributed to her mistake, so she knows precisely what to tweak in her notebook.`,
    simple: `**Autograd traces every calculation forward, then walks backward to assign blame \u2014 which numbers caused the error.**

Imagine you build a tower of blocks, and the top block falls. You want to know which block lower down caused the problem. So you trace backwards from the top: "this block wobbled because THAT block was crooked, and THAT block was crooked because THIS one was too far left."

The Value class does this for math. Every time you add or multiply numbers, it remembers what went into the calculation. When the answer is wrong, it traces backward through all the math to figure out which original numbers need to change, and by how much.`,
    technical: `Every number in the model is wrapped in a Value object that stores two things:

1. data \u2014 the actual number
2. grad \u2014 "if I nudge this number up a tiny bit, how much does the final error change?"

When you do math like a + b = c, the Value class records "c came from a and b." This builds a graph of every calculation. When you call .backward() on the final error, it walks backward through this graph using the chain rule:

\u2022 For addition (c = a + b): if c\u2019s gradient is 1, both a and b get gradient 1
\u2022 For multiplication (c = a \u00d7 b): a\u2019s gradient = b\u2019s value \u00d7 c\u2019s gradient

This is called automatic differentiation. Without it, you\u2019d have to manually write gradient formulas for every possible computation. With it, you write the forward math and gradients come automatically.`,
    deep: `This is a scalar-valued autograd engine \u2014 essentially micrograd embedded into the GPT code. Each Value node stores data, grad, a backward function closure, and parent pointers.

Supported ops: add, mul, pow, log, exp, relu \u2014 sufficient to implement the full transformer forward pass. Each op\u2019s _backward closure implements the local chain rule derivative:
- mul: \u2202L/\u2202a = b \u00b7 \u2202L/\u2202out, \u2202L/\u2202b = a \u00b7 \u2202L/\u2202out
- exp: \u2202L/\u2202a = exp(a) \u00b7 \u2202L/\u2202out
- log: \u2202L/\u2202a = (1/a) \u00b7 \u2202L/\u2202out

The backward() method performs reverse-mode autodiff via topological sort of the computation graph, then processes nodes in reverse order. This is O(n) in the number of operations \u2014 same asymptotic complexity as the forward pass.

The critical insight: this tiny class replaces PyTorch\u2019s entire autograd system. It\u2019s dramatically slower (scalar ops vs. tensor ops, Python vs. C++/CUDA), but algorithmically identical.`,
    source: `class Value:
    """ stores a single scalar value and its gradient """
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()`,
  },
  {
    id: "parameters",
    title: "Model Parameters",
    image: "/image4.webp",
    analogy: `Sam keeps a notebook full of numbers that represent her \u201cbeliefs.\u201d One page says how much weight to give hot weather (0.7), another says how much weight to give the time of day (0.3). At first, these numbers are random guesses. Over weeks of practice, she adjusts them until her predictions get good. The notebook IS her expertise \u2014 without it, she\u2019s back to random guessing.`,
    simple: `**Parameters are just numbers \u2014 thousands of them \u2014 that start random and get tuned by training.**

The model starts with thousands of random numbers. These are its "guesses about how language works." At first they\u2019re nonsense. Training slowly adjusts each number so the model gets better at predicting the next letter.

There are two special groups:
- Token embeddings: a list of numbers for each letter (what does "a" mean?)
- Position embeddings: a list of numbers for each spot (what does "3rd letter" mean?)`,
    technical: `Parameters are organized in matrices (grids of numbers):

wte (token embeddings): one row per vocabulary token. Token "a" gets row 3, which might be [0.02, -0.01, 0.03, ...] \u2014 a 16-number "fingerprint" that the model will learn to associate with the letter "a." Similar letters end up with similar fingerprints.

wpe (position embeddings): one row per position (0th slot, 1st slot, ..., 7th slot). This is how the model knows order matters \u2014 "ab" \u2260 "ba."

Per transformer layer, 6 more matrices:
\u2022 Attention projections (wq, wk, wv, wo): 4 matrices that power the attention mechanism
\u2022 MLP layers (fc1, fc2): 2 matrices for the feed-forward network

All start as small random numbers (mean=0, std=0.02). Training adjusts every single one.`,
    deep: `Weight initialization follows standard practice: Gaussian(0, 0.02) for most matrices, with output projections (attn_wo, mlp_fc2) initialized at zero. This is a deliberate choice \u2014 it means each layer initially acts as an identity (due to residual connections), which stabilizes early training.

Weight tying: wte serves double duty as both the input embedding table and the output projection. This halves the embedding parameters and creates a geometric constraint: the model\u2019s "understanding" of a token (input embedding) must be consistent with its "prediction" of that token (output logits).

With defaults (n_embd=16, vocab=28, block_size=8, n_layer=1, n_head=4):
- wte: 28\u00d716 = 448 params
- wpe: 8\u00d716 = 128 params
- Per layer: 4 attention matrices (16\u00d716 each = 1,024) + 2 MLP matrices (64\u00d716 + 16\u00d764 = 2,048)
- Total: ~3,648 params. GPT-2-XL has 1.5B. GPT-4 has ~1.8T.`,
    source: `matrix = lambda nout, nin, std=0.02: \\
    [[Value(random.gauss(0, std)) for _ in range(nin)]
     for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4*n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4*n_embd, std=0)
params = [p for mat in state_dict.values()
          for row in mat for p in row]`,
  },
  {
    id: "helpers",
    title: "Building Blocks",
    image: "/image5.webp",
    analogy: `Sam uses three tools every day. Her RECIPE CARD mixes different inputs using her own personal weightings. Her PERCENTAGE CONVERTER turns raw gut-feel scores into actual chances \u2014 lemonade 82%, strawberry 14%, water 4%. Her VOLUME KNOB rescales everything back to a reasonable range so tomorrow\u2019s math doesn\u2019t go haywire.`,
    simple: `**Three functions \u2014 linear, softmax, rmsnorm \u2014 are the entire toolkit. Everything else is combinations of these.**

Three helper tools the model uses constantly:

LINEAR: "Mix these numbers together using these weights." It\u2019s like a recipe \u2014 2 cups of flour + 1 cup of sugar = batter. The model learns the best "recipe" for combining its numbers.

SOFTMAX: Turns any list of numbers into percentages that add up to 100%. Big numbers get big percentages, small numbers get small ones. This is how the model says "I\u2019m 80% sure the next letter is \u2018a\u2019."

RMSNORM: A volume knob that keeps numbers from getting too loud or too quiet. Without it, the math breaks after a few steps.`,
    technical: `LINEAR (matrix multiply): The core operation. Takes a vector x (a list of numbers) and multiplies it by a weight matrix W. Each output number is a weighted combination of ALL inputs. This is how the model applies its learned knowledge \u2014 the weights in W encode what combinations of inputs are meaningful.

SOFTMAX: Converts raw scores ("logits") into probabilities. Steps: (1) subtract the max for stability, (2) exponentiate each value (makes everything positive), (3) divide by the sum (makes everything sum to 1). Input [-2, 1, 3] \u2192 Output [0.01, 0.04, 0.95]. The model uses this every time it needs to make a choice.

RMSNORM: Computes root-mean-square of the vector, then divides each element by it. Keeps activations stable as data flows through layers. Without normalization, values would either explode toward infinity or collapse toward zero after several matrix multiplications.`,
    deep: `linear() is a pure matrix-vector multiply implemented as nested loops: output[o] = \u03a3 W[o][i] \u00d7 x[i]. In production frameworks this dispatches to BLAS/cuBLAS and runs in microseconds on GPU. Here it\u2019s O(n\u00b2) scalar Python operations \u2014 correct but ~10\u2076\u00d7 slower.

softmax() includes the max-subtraction stability trick (log-sum-exp). Without it, exp(large_number) overflows float64. The subtraction is mathematically neutral: softmax(x - c) = softmax(x) for any constant c.

rmsnorm() differs from LayerNorm by omitting mean-centering: scale = 1/\u221a(mean(x\u00b2) + \u03b5). This saves one pass over the data and empirically performs comparably. The \u03b5 = 1e-5 prevents division by zero. Modern LLMs (LLaMA, Mistral, this code) prefer RMSNorm; GPT-2 used full LayerNorm.`,
    source: `def linear(x, w):
    return [sum(w[o][i] * x[i] for i in range(len(x)))
            for o in range(len(w))]

def softmax(logits):
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]`,
  },
  {
    id: "attention",
    title: "Self-Attention",
    image: "/image6.webp",
    analogy: `Sam is watching customer #5 approach. To predict their order, she looks back at the previous 4 customers and asks: \u201cHow relevant is each person to what #5 might order?\u201d The mom with two kids is very relevant \u2014 families order similarly. The jogger grabbing water \u2014 barely relevant. She blends past information weighted by relevance. The clever part: she does this four different ways at once, each focusing on different patterns, then combines all four into one final read.`,
    simple: `**Attention lets each token ask: \u201cwhat earlier tokens are relevant to me?\u201d \u2014 and blend their information accordingly.**

Imagine you\u2019re reading a story and you get to a blank: "The cat sat on the ___." To fill in the blank, you look back at the earlier words. "Cat" is important \u2014 cats sit on things. "The" isn\u2019t very important. "Sat" tells you it\u2019s a place. Your brain automatically focuses on the useful words and ignores the rest.

Self-attention does this for the model. At each position, it looks back at everything before it and decides what to pay attention to. Important stuff gets high attention, unimportant stuff gets ignored.`,
    technical: `For each token, attention computes three things using the linear function:

Q (Query): "What am I looking for?" \u2014 a vector representing this token\u2019s question
K (Key): "What do I contain?" \u2014 a vector representing this token\u2019s identity
V (Value): "Here\u2019s my actual information" \u2014 a vector with this token\u2019s data

To decide relevance, compute the dot product of Q with every past K. High dot product = highly relevant. Divide by \u221a(head size) to keep numbers stable. Run through softmax to get attention weights (probabilities).

Then take a weighted sum of all V vectors using those weights. If "E" has weight 0.4 and "m" has weight 0.3, the output is 40% of E\u2019s value + 30% of m\u2019s value + ...

Multi-head: the embedding is split into 4 chunks, and attention runs independently on each chunk. This lets different heads specialize \u2014 maybe head 1 looks at the immediately previous letter while head 3 looks at the first letter of the name.

The K and V vectors are cached in lists (the "KV cache"). When processing the next token, past tokens aren\u2019t recomputed \u2014 just looked up.`,
    deep: `The implementation is single-query causal attention with an explicit KV cache, processing one token per call \u2014 identical to how production LLMs handle incremental decoding.

Per layer: normalize \u2192 project to Q,K,V via three n_embd\u00d7n_embd matrices \u2192 append K,V to layer-specific cache \u2192 split into n_head heads of dimension head_dim = n_embd/n_head \u2192 per head: compute causal attention scores as Q\u00b7K^T/\u221ad_k \u2192 softmax \u2192 weighted V sum \u2192 concatenate heads \u2192 output projection via attn_wo \u2192 residual add.

Causality is implicit: since tokens are processed left-to-right and K,V are only appended (never prepended), each token\u2019s attention can only see itself and earlier positions. No explicit causal mask is needed.

The scaling factor 1/\u221ad_k prevents softmax saturation. Without it, dot products grow proportionally to d_k, pushing softmax into regions where gradients vanish.

Memory: the KV cache stores 2 \u00d7 n_layer \u00d7 seq_len \u00d7 n_embd values. For large models, this is the dominant memory cost during inference (not the parameters themselves).`,
    source: `# Inside gpt() \u2014 per layer:
x_residual = x
x = rmsnorm(x)
q = linear(x, state_dict[f'layer{li}.attn_wq'])
k = linear(x, state_dict[f'layer{li}.attn_wk'])
val = linear(x, state_dict[f'layer{li}.attn_wv'])
keys[li].append(k)
values[li].append(val)
x_attn = []
for h in range(n_head):
    hs = h * head_dim
    q_h = q[hs:hs+head_dim]
    k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
    v_h = [vi[hs:hs+head_dim] for vi in values[li]]
    attn_logits = [
        sum(q_h[j]*k_h[t][j] for j in range(head_dim))
        / head_dim**0.5
        for t in range(len(k_h))]
    attn_weights = softmax(attn_logits)
    head_out = [
        sum(attn_weights[t]*v_h[t][j]
            for t in range(len(v_h)))
        for j in range(head_dim)]
    x_attn.extend(head_out)
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
x = [a + b for a, b in zip(x, x_residual)]`,
  },
  {
    id: "mlp",
    title: "The MLP Block",
    image: "/image7.webp",
    analogy: `After Sam scans the line and gathers relevant info from past customers, she goes to her back office to actually THINK. She spreads all her notes out on a big desk, crosses out anything that doesn\u2019t seem useful, then writes her conclusions on a small index card. One key habit: she always combines her new conclusions with what she already knew. She never throws away her original read of the situation \u2014 she only adds new insights on top.`,
    simple: `**Attention gathers information from other positions; the MLP processes it. Together they form one transformer layer.**

After looking at the other letters (attention), the model needs to actually THINK about what it saw. It does this in three steps:
1. Spread out the information into a bigger space (more room to think)
2. Throw away anything that seems useless (set negatives to zero)
3. Compress back down to normal size

Then it adds this new thinking to what it already knew. The "adding back" part is important \u2014 it means the model never forgets its original information, it only adds new insights on top.`,
    technical: `The MLP (Multi-Layer Perceptron) is a two-layer feed-forward network:

1. Normalize the input (rmsnorm)
2. Expand: linear projection from 16 \u2192 64 dimensions. 4\u00d7 expansion gives the model a bigger "workspace."
3. Squared ReLU: for each number, if it\u2019s negative, set it to 0. If positive, square it. This is the nonlinearity \u2014 without it, stacking layers would be pointless because multiple linear transforms collapse to a single one. The squaring makes positive values more extreme and creates sparsity (lots of zeros).
4. Contract: linear projection from 64 \u2192 16 dimensions. Compress the results.
5. Residual add: output = MLP_output + original_input. This means the MLP only needs to learn what to ADD, not what to produce from scratch.

The residual connection is critical. It creates a "highway" that lets information and gradients flow freely through the network. Without residuals, deep networks (many layers) can\u2019t train.`,
    deep: `The MLP implements a gated nonlinear transformation in the residual stream. Architecture: pre-norm \u2192 W1 (n_embd \u2192 4\u00d7n_embd) \u2192 SqReLU \u2192 W2 (4\u00d7n_embd \u2192 n_embd) \u2192 residual add.

Squared ReLU (max(0,x)\u00b2) was introduced by Primer (So et al., 2021). Compared to GELU (GPT-2\u2019s choice), it produces sparser activations and sharper gradients. The sparsity is potentially beneficial: it means each input activates only a subset of the expanded neurons, creating implicit specialization.

The zero initialization of mlp_fc2 (and attn_wo) means each layer starts as identity: output = 0 + x_residual = x_residual. This is important for training stability \u2014 the model begins by simply passing information through and gradually learns to add useful transformations.

In the residual stream interpretation (Elhage et al., 2021), each layer reads from and writes to a shared "residual stream." Attention writes information gathered from other positions; MLP writes information computed from the current position.`,
    source: `# MLP block (inside gpt(), per layer):
x_residual = x
x = rmsnorm(x)
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
x = [xi.relu() ** 2 for xi in x]  # squared ReLU
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
x = [a + b for a, b in zip(x, x_residual)]`,
  },
  {
    id: "forward-pass",
    title: "The Full Forward Pass",
    image: "/image8.webp",
    analogy: `Here\u2019s a complete cycle for Sam. A customer approaches. She checks her reference card for this type of customer and notes what position in line they are. She combines those two pieces of info, then does her two-step process: (1) scan the line and gather relevant info from past customers, (2) go to the back office and think about what she gathered. She comes back and announces: \u201c72% lemonade, 18% strawberry, 10% water.\u201d One token in, one prediction out.`,
    simple: `**Embed \u2192 attend \u2192 MLP \u2192 predict. That\u2019s the entire forward pass. One token in, probabilities for every possible next token out.**

Here\u2019s everything that happens when the model sees one letter:

1. Look up what the letter means (token embedding) and what position it\u2019s in (position embedding). Add them together.
2. Look at all the previous letters and gather useful info (attention).
3. Think about what you gathered (MLP).
4. Use the result to predict scores for every possible next letter.

That\u2019s it. One function call, one prediction. Feed in "E" as the first letter, get back "m is 40% likely, l is 20% likely, v is 10% likely..."`,
    technical: `The gpt() function takes a single token ID and position, returns scores for every possible next token.

Embed: Look up the token\u2019s 16-number vector from wte, and the position\u2019s 16-number vector from wpe. Add them element-wise. Now the model has a representation that encodes both "what letter" and "what position."

Transform: For each layer, run attention (gather cross-position info) then MLP (process within-position info), with residual connections around each.

Unembed: Project the final 16-number vector back to vocabulary size (28 numbers) using the SAME matrix as the token embeddings (weight tying). This is elegant \u2014 the same matrix that translates tokens\u2192embeddings also translates embeddings\u2192token predictions.

The result is 28 "logits" \u2014 raw scores. High score = model thinks that token is likely to come next. These get passed through softmax to become actual probabilities.`,
    deep: `The gpt() function signature reveals the KV cache design: keys and values are external lists-of-lists passed by reference. Each call appends to them, building up context across sequential calls. This is functionally identical to how inference engines like vLLM manage KV state.

Weight tying (sharing wte for embed and unembed) creates an interesting geometric constraint: the dot product between a token\u2019s embedding and the output hidden state determines that token\u2019s logit. This means the model\u2019s internal representation of "the next token should be \u2018a\u2019" must be geometrically close (high dot product) to the embedding of \u2018a\u2019 itself.

The function is purely functional with side effects only on the mutable KV cache \u2014 clean separation of model logic from training logic.`,
    source: `def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id % block_size]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]

    for li in range(n_layer):
        # 1) Attention
        x_residual = x
        x = rmsnorm(x)
        # ... (attention computes x_attn) ...
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() ** 2 for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['wte'])  # weight tying
    return logits`,
  },
  {
    id: "training",
    title: "The Training Loop",
    image: "/image9.webp",
    analogy: `Every day, Sam watches one real customer and compares her prediction to what actually happened. She checks how wrong she was \u2014 if she was really confident in the WRONG answer, she needs to change a lot. She traces back through her reasoning to figure out which numbers in her notebook to adjust. She tweaks every number just a little in the direction that would have made her less wrong. She does this every day for 1,000 days. By the end, she\u2019s very good.`,
    simple: `**Training is just: predict, measure error, trace blame, nudge parameters. Repeat 1,000 times.**

Training is like practice:

1. Pick a name, like "Emma"
2. Show the model one letter at a time and ask it to guess the next one
3. Tell it whether it was right or wrong (and how wrong)
4. Let it adjust its numbers a tiny bit to be less wrong next time
5. Do this 1,000 times with different names

After enough practice, the model gets good at guessing what letters follow other letters in names.`,
    technical: `Each training step:

1. Pick a name (e.g., "Emma"), tokenize it: [BOS, E, m, m, a, EOS]
2. Feed tokens one at a time through the model. At each position, the model predicts the next token:
   - See BOS \u2192 predict E
   - See BOS, E \u2192 predict m
   - See BOS, E, m \u2192 predict m
   - See BOS, E, m, m \u2192 predict a
   - See BOS, E, m, m, a \u2192 predict EOS

3. Compute cross-entropy loss at each position: -log(probability assigned to the correct answer). If the model said P(correct)=0.9, loss=-log(0.9)=0.105 (good). If P(correct)=0.01, loss=-log(0.01)=4.6 (bad). Average across all positions.

4. Call loss.backward() \u2014 the autograd engine computes gradients for every parameter.

5. Adam optimizer adjusts every parameter using those gradients. Adam is smarter than basic "gradient descent" \u2014 it maintains momentum (consistent direction = bigger steps) and adapts the learning rate per parameter (noisy gradients = smaller steps).

6. Learning rate decays linearly to zero over training, so early steps make bold adjustments and later steps fine-tune.`,
    deep: `The training loop processes one document per step (batch size 1, no gradient accumulation). Each document is tokenized with BOS/EOS, cropped to block_size, and fed through the transformer sequentially.

The loss is mean cross-entropy over positions: L = (1/T) \u03a3 -log P(x_{t+1} | x_{\u2264t}). The backward() call on each per-position loss accumulates gradients \u2014 equivalent to computing total loss and calling backward once, but without holding the entire computation graph simultaneously.

Adam optimizer (Kingma & Ba, 2014) with \u03b21=0.9, \u03b22=0.95, \u03b5=1e-8. Bias correction applied. Linear LR warmdown: lr_t = lr \u00d7 (1 - step/num_steps). No weight decay, no gradient clipping \u2014 acceptable at this scale but would cause instability in larger models.`,
    source: `# Adam optimizer setup
learning_rate = args.learning_rate
beta1, beta2, eps_adam = 0.9, 0.95, 1e-8
m = [0.0] * len(params)  # first moment
v = [0.0] * len(params)  # second moment

for step in range(args.num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [stoi[ch] for ch in doc] + [EOS]
    tokens = tokens[:block_size]

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    lossf = 0.0
    for pos_id in range(len(tokens) - 1):
        logits = gpt(tokens[pos_id], pos_id, keys, values)
        probs = softmax(logits)
        loss = -probs[tokens[pos_id + 1]].log()
        loss = (1 / (len(tokens) - 1)) * loss
        loss.backward()
        lossf += loss.data

    lr_t = learning_rate * (1 - step / args.num_steps)
    for i, p in enumerate(params):
        m[i] = beta1*m[i] + (1-beta1)*p.grad
        v[i] = beta2*v[i] + (1-beta2)*p.grad**2
        m_hat = m[i] / (1 - beta1**(step+1))
        v_hat = v[i] / (1 - beta2**(step+1))
        p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
        p.grad = 0`,
  },
  {
    id: "inference",
    title: "Generation",
    image: "/image10.webp",
    analogy: `After 1,000 days of practice, Sam puts her trained notebook to use \u2014 not to predict real customers, but to INVENT a plausible order from scratch. She starts by imagining a customer just walked up, checks her notebook, and rolls a weighted die \u2014 lemonade. She pretends that already happened and predicts the next item \u2014 strawberry. She keeps going until her notebook predicts \u201cthe order is done.\u201d The sequence she produced is a brand-new order that nobody ever actually placed, invented from learned patterns.`,
    simple: `**Generation is just the forward pass in a loop \u2014 predict next token, sample it, feed it back in. That\u2019s how every LLM writes.**

Now the model makes up its own names:

1. Start with the "beginning" signal
2. Ask the model: "what letter comes next?" It says "E is 30% likely, a is 20% likely..."
3. Pick a letter randomly (but letters with higher percentages get picked more often)
4. Feed that letter back in and repeat
5. Stop when the model outputs the "ending" signal

The result: a brand-new name that sounds real but was never in the training data. Like "Emily" or "Arvin."`,
    technical: `Generation is autoregressive \u2014 the model feeds its own output back as input.

1. Start with token BOS.
2. Run gpt() to get logits, softmax to get probabilities.
3. Sample: use random.choices with the probability distribution as weights. If P(m)=0.4, P(a)=0.3, P(e)=0.2, any could be picked \u2014 m is most likely but not guaranteed. This randomness is why each run produces different names.
4. If sampled token is EOS, stop. Otherwise, convert ID back to character and continue.
5. The KV cache carries over \u2014 each new token benefits from all past context without recomputing it.

The sampling strategy here is pure multinomial (sample proportional to probabilities). Production LLMs add temperature (scale randomness), top-k (only consider k most likely), and top-p (only consider tokens until cumulative probability hits p).`,
    deep: `Inference loop: initialize empty KV cache, seed with BOS, autoregressive decode for up to block_size tokens. Sampling is unmodified multinomial via random.choices \u2014 no temperature, no top-k/p truncation.

Each generated token feeds back as the next input with incremented pos_id. The KV cache grows by one entry per layer per step. This is the standard autoregressive decode loop \u2014 the only difference from production is speed (~1 tok/s CPU vs ~100 tok/s GPU).

After 1000 steps on names, typical outputs are 3-6 character strings exhibiting learned phonotactic patterns (consonant-vowel alternations, typical endings like -ly, -an, -ia) without reproducing training examples verbatim.

This completes the LLM lifecycle: data \u2192 tokenize \u2192 train \u2192 generate. Every commercial LLM follows this exact pipeline, differing only in scale.`,
    source: `print("\\n--- generation ---")
for sample_idx in range(5):
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    token_id = BOS
    generated = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        token_id = random.choices(
            range(vocab_size),
            weights=[p.data for p in probs])[0]
        if token_id == EOS:
            break
        generated.append(itos[token_id])
    print(f"sample {sample_idx}: {''.join(generated)}")`,
  },
  {
    id: "scale",
    title: "From Micro to Macro",
    image: "/image11.webp",
    analogy: `Sam\u2019s lemonade stand has 3 drinks, serves 10 customers/day, and uses a pocket notebook. McDonald\u2019s has 100+ menu items, serves 70 million customers/day, and uses a global supply chain with AI forecasting. But the LOGIC is the same: observe patterns, predict demand, adjust based on mistakes. A bigger notebook, more data to learn from, and more practice time \u2014 that\u2019s what separates a stand from an empire. The recipe never changed.`,
    simple: `**The algorithm in this 140-line file is the same one running inside GPT-4. The difference is only scale.**

This tiny model learns names with 5,000 numbers. GPT-4 uses 1,800,000,000,000 numbers (1.8 trillion). But the steps are IDENTICAL:
- Turn text into numbers
- Predict next token
- Measure how wrong you were
- Adjust and repeat

Making the model bigger doesn\u2019t change what it does \u2014 it changes how WELL it does it. With enough numbers and enough practice data, the same algorithm that makes up names can write essays, code, and poetry.`,
    technical: `Everything in this file scales directly to production LLMs:

Size: 5K params \u2192 GPT-2 1.5B \u2192 GPT-3 175B \u2192 GPT-4 ~1.8T
Layers: 1 \u2192 48 \u2192 96 \u2192 ~120
Context: 8 tokens \u2192 1024 \u2192 4096 \u2192 128K
Tokenizer: 28 characters \u2192 50K BPE tokens \u2192 100K+ tokens
Data: 32K names \u2192 40GB web text \u2192 trillions of tokens

Architectural improvements at scale: Mixture of Experts (only activate a fraction of params per token), Grouped-Query Attention (share KV heads to reduce memory), FlashAttention (fused GPU kernels), Rotary Position Embeddings (better long-context extrapolation).

None of these change the fundamental algorithm. They\u2019re all efficiency optimizations around the same core: embed \u2192 attend \u2192 MLP \u2192 predict \u2192 learn.`,
    deep: `Scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022 "Chinchilla") showed performance follows power laws in model size, data, and compute. This justified multi-billion-dollar training runs.

What changes at scale: distributed parallelism (data/tensor/pipeline), ZeRO, mixed precision (bf16/fp8), gradient checkpointing, SwiGLU, GQA, RoPE, MoE routing, speculative decoding, KV cache quantization, continuous batching, paged attention.

What stays identical: autoregressive next-token prediction, transformer (attention + MLP with residuals), softmax attention with QKV projections, cross-entropy loss, gradient-based optimization.

This file proves: the core algorithm fits in 140 lines. Everything else is engineering.`,
    source: `# microgpt defaults:
#   n_embd=16, n_layer=1, n_head=4, block_size=8
#   ~5,000 parameters
#   character-level tokenizer, names dataset

# GPT-2 (2019):
#   n_embd=1600, n_layer=48, n_head=25
#   block_size=1024, 1.5 billion parameters
#   BPE tokenizer (50K vocab), WebText

# GPT-3 (2020):
#   n_embd=12288, n_layer=96, n_head=96
#   block_size=2048, 175 billion parameters

# GPT-4 (2023, estimated):
#   ~1.8 trillion parameters, MoE architecture
#   128K context window, multimodal input
#   Same core algorithm. Just bigger.`,
  },
];
