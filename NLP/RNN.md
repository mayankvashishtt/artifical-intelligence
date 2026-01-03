# Recurrent Neural Networks (RNN) & LSTM - Complete Guide

**From Foundations to Advanced Concepts | Interview & Exam Ready**

---

## Table of Contents

1. [Prerequisites & Motivation](#1-prerequisites--motivation)
2. [Limitations of Traditional Neural Networks](#2-limitations-of-traditional-neural-networks)
3. [Introduction to RNNs](#3-introduction-to-rnns)
4. [RNN Architecture Types](#4-rnn-architecture-types)
5. [Word Embeddings vs RNNs](#5-word-embeddings-vs-rnns)
6. [Mathematical Formulation of RNNs](#6-mathematical-formulation-of-rnns)
7. [Forward Propagation in RNNs](#7-forward-propagation-in-rnns)
8. [Backpropagation Through Time (BPTT)](#8-backpropagation-through-time-bptt)
9. [Jacobians in RNN Training](#9-jacobians-in-rnn-training)
10. [Gradient Problems](#10-gradient-problems)
11. [Solutions to Gradient Problems](#11-solutions-to-gradient-problems)
12. [Introduction to LSTM](#12-introduction-to-lstm)
13. [Implementation Guide](#13-implementation-guide)
14. [Interview Questions & Answers](#14-interview-questions--answers)

---

## 1. Prerequisites & Motivation

### Why Sequential Data is Different

**Sequential data** has inherent order and temporal dependencies:
- Natural language text
- Time series (stock prices, weather)
- Speech and audio
- Video frames
- DNA sequences

**Key characteristic**: The order matters. Changing the order changes the meaning.

---

## 2. Limitations of Traditional Neural Networks

### Artificial Neural Networks (ANN/MLP) Limitations

Traditional feedforward networks process inputs **independently**:

```
Input → Hidden Layers → Output
```

**Problems:**

1. **No Memory**: Each input is processed in isolation
2. **Fixed Input Size**: Cannot handle variable-length sequences
3. **No Temporal Understanding**: "dog bites man" ≡ "man bites dog"
4. **Parameter Explosion**: Separate weights for each position

### Example Failure Case

```
Sentence 1: "The movie was not good"
Sentence 2: "The movie was good"
```

ANNs using Bag-of-Words would see identical word frequencies but opposite meanings.

---

## 3. Introduction to RNNs

### Core Concept

> **RNN = Neural Network with Memory**

RNNs process sequential data **one step at a time** while maintaining a **hidden state** that acts as memory.

### Key Innovation

Instead of processing all inputs simultaneously, RNNs:
1. Process input at time step `t`
2. Combine it with memory from step `t-1`
3. Update memory for step `t`
4. Repeat for next time step

### Visual Representation

```
Unrolled RNN across time:

x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃
      ↑  ↓         ↑  ↓         ↑  ↓
     h₀           h₁           h₂
```

**Key Point**: Same weights are shared across all time steps (parameter sharing).

---

## 4. RNN Architecture Types

### 4.1 One-to-One
**Not an RNN**, just a regular neural network.

```
Input → Output
```

**Example**: Image classification

---

### 4.2 Many-to-One
Multiple inputs → Single output

```
x₁, x₂, x₃, ..., xₙ → y
```

**Examples**:
- Sentiment analysis (sentence → positive/negative)
- Video classification (frames → category)

---

### 4.3 One-to-Many
Single input → Multiple outputs

```
x → y₁, y₂, y₃, ..., yₙ
```

**Examples**:
- Image captioning (image → sentence)
- Music generation (seed → sequence)

---

### 4.4 Many-to-Many (Same Length)
Sequence input → Sequence output (same length)

```
x₁, x₂, x₃ → y₁, y₂, y₃
```

**Examples**:
- Video frame labeling
- Part-of-speech tagging

---

### 4.5 Many-to-Many (Different Length)
Sequence input → Sequence output (different length)

```
x₁, x₂, x₃ → [Encoder-Decoder] → y₁, y₂, y₃, y₄
```

**Examples**:
- Machine translation (English → French)
- Text summarization

---

## 5. Word Embeddings vs RNNs

### Critical Distinction ⚠️

**Common Misconception**: RNNs create word embeddings
**Reality**: RNNs **use** word embeddings as input

### The Complete Pipeline

```
Raw Text → Tokenization → Word IDs → Embedding Layer → RNN → Output
```

### Word Embeddings Explained

**Word2Vec, GloVe**: Learn dense vector representations

```
"king" → [0.12, -0.44, 0.89, 0.56, ...]
"queen" → [0.09, -0.41, 0.85, 0.59, ...]
```

**Properties**:
- Similar words have similar vectors
- Captures semantic relationships
- Fixed-size representation

### What RNN Actually Does

1. **Receives** pre-trained or learnable embeddings
2. **Processes** sequences by learning context and order
3. **Updates** hidden state based on current input and previous memory

### Why This Matters

- **Bag-of-Words/TF-IDF**: No semantic understanding, no order
- **Word2Vec**: Semantic understanding, no sequential processing
- **RNN**: Sequential processing using embeddings

---

## 6. Mathematical Formulation of RNNs

### Core RNN Equation

At each time step `t`, the hidden state is computed as:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

Where:
- $h_t$ = hidden state at time `t` (memory)
- $h_{t-1}$ = previous hidden state
- $x_t$ = input at time `t`
- $W_h$ = recurrent weight matrix (memory → memory)
- $W_x$ = input weight matrix (input → memory)
- $b$ = bias vector
- $\tanh$ = activation function

### Output Equation

$$y_t = W_y h_t + b_y$$

Or with activation:

$$y_t = \text{softmax}(W_y h_t + b_y)$$

### Interpretation

The equation says:
> "Current memory = function of (previous memory + current input)"

### Dimensions

For vocabulary size `V`, embedding dimension `d`, hidden dimension `h`:

- $x_t \in \mathbb{R}^d$ (embedded input)
- $h_t \in \mathbb{R}^h$ (hidden state)
- $W_x \in \mathbb{R}^{h \times d}$ (input weights)
- $W_h \in \mathbb{R}^{h \times h}$ (recurrent weights)
- $b \in \mathbb{R}^h$ (bias)
- $y_t \in \mathbb{R}^V$ (output, e.g., vocabulary size)

---

## 7. Forward Propagation in RNNs

### Step-by-Step Process

**Initialization**:
$$h_0 = \mathbf{0}$$

**For each time step** `t = 1` to `T`:

1. **Compute hidden state**:
   $$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

2. **Compute output** (if needed at this step):
   $$y_t = W_y h_t + b_y$$

3. **Compute loss** (for training):
   $$L_t = \text{CrossEntropy}(y_t, \text{target}_t)$$

**Total Loss**:
$$L = \sum_{t=1}^{T} L_t$$

### Example Walkthrough

Sentence: "I love NLP"

```
Step 1: h₁ = tanh(Wₕ·0 + Wₓ·embed("I") + b)
Step 2: h₂ = tanh(Wₕ·h₁ + Wₓ·embed("love") + b)
Step 3: h₃ = tanh(Wₕ·h₂ + Wₓ·embed("NLP") + b)

Output: y = Wᵧ·h₃ + bᵧ
```

### Key Observations

1. **Same weights** ($W_h, W_x$) used at every step
2. **Memory flows forward**: $h_1 → h_2 → h_3$
3. **Information accumulation**: Later states contain information from all previous steps

---

## 8. Backpropagation Through Time (BPTT)

### Motivation

Standard backpropagation won't work because:
- Weights are shared across time steps
- Gradients must flow backward through time
- Each weight affects multiple time steps

### The Challenge

To update $W_h$, we need:

$$\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_h}$$

But $W_h$ affects $h_t, h_{t+1}, ..., h_T$ (all future states).

### BPTT Algorithm

**Forward Pass** (already described):
- Compute $h_1, h_2, ..., h_T$
- Store all intermediate values

**Backward Pass** (BPTT):

Starting from final time step `T`, going backward to `1`:

$$\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t} + \frac{\partial L_t}{\partial h_t}$$

### Gradient for Recurrent Weights

$$\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial L_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_h}$$

### Interpretation

> "The gradient at weight $W_h$ accumulates contributions from all time steps where it was used."

### Computational Cost

- **Time Complexity**: $O(T)$ per parameter
- **Space Complexity**: $O(T)$ (must store all hidden states)

---

## 9. Jacobians in RNN Training

### What is a Jacobian?

The Jacobian measures how one hidden state depends on the previous:

$$J_t = \frac{\partial h_t}{\partial h_{t-1}}$$

### Computing the Jacobian

From $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$:

$$J_t = \frac{\partial h_t}{\partial h_{t-1}} = W_h^T \cdot \text{diag}(1 - \tanh^2(W_h h_{t-1} + W_x x_t + b))$$

### Chain Rule Through Time

To find how $h_1$ affects $h_T$:

$$\frac{\partial h_T}{\partial h_1} = J_T \cdot J_{T-1} \cdot ... \cdot J_2$$

**This is a product of T-1 Jacobian matrices!**

### Why This Matters

During BPTT, gradients flow backward:

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=2}^{T} J_t$$

**Problem**: Long chains of matrix multiplication cause numerical instability.

---

## 10. Gradient Problems

### 10.1 Vanishing Gradient Problem ❄️

**Cause**: When Jacobian eigenvalues < 1:

$$\prod_{t=2}^{T} J_t \to 0 \text{ as } T \to \infty$$

**Effect**:
- Early time steps receive near-zero gradients
- Weights for early inputs barely update
- **Long-term dependencies cannot be learned**

**Mathematical Insight**:

If $||J_t|| < \gamma < 1$:

$$\left|\left| \prod_{t=k}^{T} J_t \right|\right| \leq \gamma^{T-k+1}$$

For $T=100, \gamma=0.9$: gradient shrinks by factor of $\sim 10^{-5}$

**Example Failure**:
```
"The cat, which was sitting on the mat, was very fluffy"
                                              ↑
Cannot link "was" to "cat" if they're far apart
```

---

### 10.2 Exploding Gradient Problem 💥

**Cause**: When Jacobian eigenvalues > 1:

$$\prod_{t=2}^{T} J_t \to \infty \text{ as } T \to \infty$$

**Effect**:
- Gradients become astronomically large
- Weight updates cause wild oscillations
- Training becomes unstable or diverges

**Mathematical Insight**:

If $||J_t|| > \gamma > 1$:

$$\left|\left| \prod_{t=k}^{T} J_t \right|\right| \geq \gamma^{T-k+1}$$

For $T=100, \gamma=1.1$: gradient explodes by factor of $\sim 10^4$

---

### Why $\tanh$ Contributes to Vanishing Gradients

$$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$

- Maximum derivative = 1 (at $x=0$)
- Typical derivative < 0.5
- Repeated multiplication: $0.5^{100} \approx 10^{-30}$

---

## 11. Solutions to Gradient Problems

### 11.1 Gradient Clipping (for Exploding)

**Idea**: Cap gradient magnitude

```python
if ||g|| > threshold:
    g = threshold * (g / ||g||)
```

**Mathematical Form**:

$$g_{\text{clipped}} = \begin{cases}
g & \text{if } ||g|| \leq \theta \\
\theta \cdot \frac{g}{||g||} & \text{if } ||g|| > \theta
\end{cases}$$

---

### 11.2 Truncated BPTT (for Both)

**Idea**: Limit backward propagation depth

Instead of:
$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=2}^{T} J_t$$

Use:
$$\frac{\partial L}{\partial h_{T-k}} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=T-k+1}^{T} J_t$$

**Trade-off**: Can't learn dependencies longer than `k` steps

---

### 11.3 Better Activation Functions

- **ReLU**: No saturation, but can cause instability
- **Leaky ReLU**: $\max(0.01x, x)$
- Problem: Still doesn't fully solve vanishing gradients

---

### 11.4 Proper Weight Initialization

- **Xavier initialization**
- **He initialization**
- Keeps gradient flow balanced initially

---

### 11.5 LSTM/GRU (Best Solution)

Use gated architectures that allow gradient flow through additive paths.

---

## 12. Introduction to LSTM

### Motivation

**RNN Problem**: Cannot maintain long-term dependencies due to vanishing gradients.

**LSTM Solution**: Introduce a separate **cell state** with additive updates.

### Architecture Overview

LSTM has **two states**:
- $c_t$: **Cell state** (long-term memory)
- $h_t$: **Hidden state** (short-term memory / output)

### The Four Gates

#### 1. Forget Gate $f_t$
**Decides what to forget from cell state**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- Output: 0 (forget completely) to 1 (keep completely)

#### 2. Input Gate $i_t$
**Decides what new information to add**

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

#### 3. Candidate Values $\tilde{c}_t$
**New candidate values for cell state**

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

#### 4. Output Gate $o_t$
**Decides what to output**

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

### Cell State Update (Key Innovation!)

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Notice**: This is **addition**, not multiplication!

Where $\odot$ is element-wise multiplication.

### Hidden State Update

$$h_t = o_t \odot \tanh(c_t)$$

### Why LSTM Solves Vanishing Gradients

**Gradient flow through cell state**:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

Instead of repeated multiplication of Jacobians:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

The gradient path is:
$$\frac{\partial L}{\partial c_1} = \frac{\partial L}{\partial c_T} \cdot \prod_{t=2}^{T} f_t$$

**Advantage**: If forget gates ≈ 1, gradient flows unimpeded!

### Visual Summary

```
         ┌─────────────────┐
         │   Cell State    │
    c(t-1)────→ [×] ─→ [+] ────→ c(t)
         │      ↑      ↑   │
         │      ft     it  │
         │             │   │
         │   ┌────┐ ┌───┐ │
    h(t-1)───┤ σ  │ │ σ │ │
         │   └────┘ └───┘ │
    x(t)─────────┘    │   │
         │      ot  ┌───┐ │
         │      ↓   │tanh│ │
         └──────[×]─┴───┘ └────→ h(t)
```

---

## 13. Implementation Guide

### 13.1 Basic RNN in Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Model architecture
model = Sequential([
    # Convert word IDs to embeddings
    Embedding(input_dim=10000,  # vocabulary size
              output_dim=64,     # embedding dimension
              input_length=100), # sequence length
    
    # RNN layer
    SimpleRNN(units=64,          # hidden state size
              activation='tanh',
              return_sequences=False),  # only final output
    
    # Output layer
    Dense(1, activation='sigmoid')  # binary classification
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, 
          epochs=5, 
          batch_size=32,
          validation_data=(X_val, y_val))
```

---

### 13.2 LSTM in Keras

```python
from tensorflow.keras.layers import LSTM

model = Sequential([
    Embedding(input_dim=10000, output_dim=64),
    LSTM(units=64, return_sequences=False),
    Dense(1, activation='sigmoid')
])
```

---

### 13.3 Stacked RNN/LSTM

```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=64),
    
    # First LSTM layer (return sequences for next layer)
    LSTM(128, return_sequences=True),
    
    # Second LSTM layer
    LSTM(64, return_sequences=False),
    
    Dense(1, activation='sigmoid')
])
```

---

### 13.4 Bidirectional RNN

**Process sequence in both directions**

```python
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Embedding(input_dim=10000, output_dim=64),
    Bidirectional(LSTM(64)),
    Dense(1, activation='sigmoid')
])
```

---

### 13.5 Many-to-Many (Sequence-to-Sequence)

```python
# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=10000, output_dim=64)(encoder_inputs)
encoder_lstm = LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=10000, output_dim=64)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, 
                                      initial_state=encoder_states)
decoder_dense = Dense(10000, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
```

---

## 14. Interview Questions & Answers

### Q1: What is the main difference between RNN and feedforward neural networks?

**Answer**: 
RNNs process sequential data by maintaining a hidden state that acts as memory. The same weights are applied at each time step, allowing the network to handle variable-length sequences and capture temporal dependencies. Feedforward networks process inputs independently without memory.

---

### Q2: Explain the vanishing gradient problem in RNNs.

**Answer**:
During BPTT, gradients are computed by multiplying Jacobian matrices across time steps. When these Jacobians have eigenvalues less than 1, their repeated multiplication causes gradients to exponentially decay, preventing the network from learning long-term dependencies. Mathematically: $\|\prod_{t=k}^{T} J_t\| \to 0$ as $T-k$ increases.

---

### Q3: How does LSTM solve the vanishing gradient problem?

**Answer**:
LSTM introduces a cell state with **additive** updates rather than multiplicative ones. The cell state equation $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ creates a gradient highway. When forget gates are close to 1, gradients can flow backward through time without vanishing, as the derivative $\frac{\partial c_t}{\partial c_{t-1}} = f_t$ doesn't involve repeated matrix multiplication.

---

### Q4: What is the purpose of the forget gate in LSTM?

**Answer**:
The forget gate decides what information from the previous cell state to retain or discard. It outputs values between 0 (completely forget) and 1 (completely retain). This allows the network to forget irrelevant information and focus on important patterns.

---

### Q5: Explain Backpropagation Through Time (BPTT).

**Answer**:
BPTT is the training algorithm for RNNs. It unfolds the RNN across time steps and applies backpropagation. The key difference from standard backprop is that gradients must be accumulated across all time steps where shared weights were used. The gradient for $W_h$ is: $\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_h}$.

---

### Q6: When would you use a many-to-many vs many-to-one RNN architecture?

**Answer**:
- **Many-to-many**: When both input and output are sequences (machine translation, video captioning, text generation)
- **Many-to-one**: When input is a sequence but output is single value (sentiment analysis, document classification)

---

### Q7: What is truncated BPTT and why is it used?

**Answer**:
Truncated BPTT limits the number of time steps for backpropagation (e.g., backprop only 20 steps instead of 100). This reduces computational cost and helps stabilize gradients, but at the cost of not learning dependencies longer than the truncation window.

---

### Q8: How do word embeddings relate to RNNs?

**Answer**:
Word embeddings (Word2Vec, GloVe) convert words into dense vectors that capture semantic meaning. RNNs don't create these embeddings—they **consume** them as input. The typical pipeline is: Text → Tokenization → Word IDs → Embedding Layer → RNN → Output.

---

### Q9: What causes exploding gradients and how do you fix them?

**Answer**:
Exploding gradients occur when Jacobian matrices have eigenvalues > 1, causing gradients to grow exponentially during BPTT. **Solution**: Gradient clipping—cap gradient norm at a threshold: $g_{\text{clipped}} = \theta \cdot \frac{g}{\|g\|}$ if $\|g\| > \theta$.

---

### Q10: Compare RNN, LSTM, and GRU.

**Answer**:
- **RNN**: Simple, fast, but suffers from vanishing gradients
- **LSTM**: Three gates (forget, input, output) + cell state. Best for long sequences but computationally expensive
- **GRU**: Two gates (reset, update). Simpler than LSTM, faster training, often comparable performance

---

### Q11: What is the purpose of `return_sequences=True` in Keras?

**Answer**:
`return_sequences=True` makes the RNN/LSTM layer output hidden states for **all** time steps (shape: `[batch, time, features]`) instead of just the final time step. This is necessary when:
- Stacking multiple RNN layers
- Building sequence-to-sequence models
- Needing per-timestep predictions

---

### Q12: Why do we use `tanh` activation in RNNs?

**Answer**:
`tanh` outputs values in [-1, 1], allowing the network to learn both positive and negative signals. It's zero-centered (unlike sigmoid), which helps with training. However, its derivative is small for large inputs, contributing to vanishing gradients—one reason LSTMs were developed.

---

### Q13: Explain the dimensions in the RNN equation $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$

**Answer**:
If hidden dimension is $h$ and input dimension is $d$:
- $x_t$: $[d \times 1]$ (input vector)
- $h_{t-1}$: $[h \times 1]$ (previous hidden state)
- $W_x$: $[h \times d]$ (input-to-hidden weights)
- $W_h$: $[h \times h]$ (hidden-to-hidden weights)
- $b$: $[h \times 1]$ (bias)
- $h_t$: $[h \times 1]$ (new hidden state)

---

### Q14: What is teacher forcing?

**Answer**:
In sequence-to-sequence models, teacher forcing means using the **true previous output** as input to the next time step during training, rather than the model's prediction. This accelerates training but can cause exposure bias (model never sees its own errors during training).

---

### Q15: How would you handle variable-length sequences in RNNs?

**Answer**:
1. **Padding**: Pad shorter sequences to match longest sequence (add special tokens)
2. **Masking**: Use masking layer to ignore padded values
3. **Packing**: Use packed sequences (PyTorch) to process only valid timesteps
4. **Bucketing**: Group similar-length sequences together in batches

---

## Key Takeaways for Interviews

✅ **RNNs process sequences with memory through recurrent connections**  
✅ **BPTT trains RNNs by unfolding through time**  
✅ **Vanishing gradients prevent learning long-term dependencies**  
✅ **LSTM uses additive cell state updates to solve vanishing gradients**  
✅ **Word embeddings are INPUT to RNNs, not created by them**  
✅ **Truncated BPTT and gradient clipping help stabilize training**  
✅ **Architecture choice (many-to-one, many-to-many) depends on task**

---

## Mathematical Cheat Sheet

| Concept | Equation |
|---------|----------|
| RNN Hidden State | $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$ |
| RNN Output | $y_t = W_y h_t + b_y$ |
| Total Loss | $L = \sum_{t=1}^{T} L_t$ |
| Jacobian | $J_t = \frac{\partial h_t}{\partial h_{t-1}}$ |
| BPTT Gradient Flow | $\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=2}^{T} J_t$ |
| LSTM Cell Update | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ |
| LSTM Hidden State | $h_t = o_t \odot \tanh(c_t)$ |
| Gradient Clipping | $g = \min(1, \frac{\theta}{\|g\|}) \cdot g$ |

---
