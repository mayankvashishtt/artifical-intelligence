# Complete Guide to Transformers and Attention Mechanisms

## Table of Contents
1. [What is a Transformer?](#what-is-a-transformer)
2. [Tokenization: Breaking Down Text](#tokenization)
3. [Embedding Vectors: Words as Numbers](#embedding-vectors)
4. [Attention Mechanism: The Heart of Transformers](#attention-mechanism)
5. [Multi-Head Attention: Multiple Perspectives](#multi-head-attention)
6. [Geometry of High-Dimensional Spaces](#geometry-of-spaces)
7. [Query, Key, Value Spaces](#query-key-value-spaces)
8. [Complete Transformer Architecture](#complete-architecture)

---

## What is a Transformer?

### Simple Definition

A **Transformer** is a neural network architecture that processes sequences of data (text, audio, images) by learning which parts to focus on using an **attention mechanism**.

**Key Innovation**: The attention mechanism allows the network to focus on relevant parts of the input, regardless of distance.

### The "Attention is All You Need" Paper (2017)

```
Before Transformers:
- RNNs/LSTMs processed sequentially: word 1 → word 2 → word 3 → ...
- Slow to train (can't parallelize)
- Bad at understanding distant relationships

The Transformer (2017):
- Process ALL words at once (parallel)
- Each word attends to all other words
- Much faster training
- Better understanding of context
- Revolutionized NLP!

Modern LLMs (GPT, Claude, Llama):
All built on transformer architecture
```

### Why Transformers Won

```
Processing Speed Comparison:

RNN (Sequential):
Word 1 → Word 2 → Word 3 → Word 4 → Word 5
Takes: 5 steps (slow!)

Transformer (Parallel):
Word 1 ─┐
Word 2 ─├→ Process all at once!
Word 3 ─┤
Word 4 ─┤
Word 5 ─┘
Takes: 1 step (100x faster!)

Result:
- Training time: From weeks to days
- Can handle longer sequences
- Scales to billions of parameters
```

---

## Tokenization: Breaking Down Text

### What is a Token?

A **token** is a small unit of text that the model processes. It's not characters, it's chunks that make sense together.

### Why Not Just Use Characters?

#### The Character Approach (Bad Idea ❌)

```
Text: "Hello, world!"

Characters:
['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']
13 items to process

Problems:
1. TOO MANY items (waste of computation)
2. Loses meaning ("Hel" by itself is meaningless)
3. Hard to learn patterns (need to learn "H" + "e" + "l" + "l" + "o" = "hello")
4. Inefficient use of attention (attending to each character)
```

#### The Token Approach (Good Idea ✓)

```
Text: "Hello, world!"

Tokens:
['Hello', ',', 'world', '!']
4 items to process

Benefits:
1. Fewer items (efficient computation)
2. Preserves meaning ("Hello" is a word)
3. Easier to learn ("Hello" → greeting)
4. Efficient attention (attend to meaningful units)
```

### How Tokenization Works

#### Byte-Pair Encoding (BPE) - Most Common

```
Step 1: Start with all characters
"hello world" = ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

Step 2: Find most frequent pair
'l' + 'l' appears most often
Merge: ['h', 'e', 'll', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

Step 3: Repeat
'o' + ' ' appears often
Merge: ['h', 'e', 'll', 'o ', 'w', 'o', 'r', 'l', 'd']

Step 4: Keep merging
Eventually: ['hello', ' ', 'world']

Result: Vocabulary of tokens!
```

#### Real Example: GPT Tokenization

```
Text: "I love pizza"

GPT Tokens:
['I', ' love', ' pizza']  (3 tokens)

Text: "Supercalifragilisticexpialidocious"

Tokens:
['Super', 'cal', 'ifrag', 'ilist', 'ic', 'exp', 'ial', 'i', 'doc', 'ious']  (10 tokens)

Rule of thumb:
1 token ≈ 4 characters in English
100 words ≈ 75 tokens
```

### Tokenization for Different Modalities

#### Text Tokens
```
"The quick brown fox"
→ ['The', ' quick', ' brown', ' fox']
```

#### Audio Tokens
```
Speech waveform (44.1 kHz audio)
Split into 20ms chunks
Each chunk = 1 token
→ Allows processing speech like text!
```

#### Image Tokens
```
Image (224×224 pixels)
Divide into 16×16 patches
Each patch = 1 token
Total: (224/16) × (224/16) = 196 tokens

Image becomes a sequence!
→ Vision Transformers (ViT)
```

---

## Embedding Vectors: Words as Numbers

### What is an Embedding?

An **embedding** is a list of numbers that represents the meaning of a word.

**Analogy**: Instead of just having the word "king", we represent it as a point in multi-dimensional space where related words are nearby.

### Simple Example: 2D Embedding

```
Imagine a simple 2D world:

Axis 1: Masculine ← → Feminine
Axis 2: Royalty ← → Commoner

Word positions:
King:   (0.8,  0.9)   ← Masculine, Royal
Queen:  (-0.8, 0.9)   ← Feminine, Royal
Prince: (0.7,  0.7)   ← Masculine, Less royal
Girl:   (-0.9, 0.2)   ← Feminine, Less royal
Boy:    (0.9,  0.2)   ← Masculine, Less royal

Relationships:
King - Man ≈ Queen - Woman
(0.8, 0.9) - (0.8, 0.1) ≈ (-0.8, 0.9) - (-0.8, 0.1)
(0.0, 0.8) ≈ (0.0, 0.8) ✓ Works!
```

### Real Embeddings: 768 or 12,288 Dimensions

```
Modern models use MANY dimensions:

BERT: 768-dimensional embeddings
GPT-2: 768-dimensional embeddings
Claude: 4,096-dimensional embeddings
Llama 2 70B: 4,096-dimensional embeddings

Why so many dimensions?
More dimensions = more information = better representations
```

### What Information is Encoded?

```
In a 768-dimensional embedding, different dimensions might represent:

Dimension 1: Part of speech (noun, verb, adjective...)
Dimension 2: Gender (masculine, feminine, neutral...)
Dimension 3: Concreteness (abstract vs physical...)
Dimension 4: Sentiment (positive, negative, neutral...)
Dimension 5: Formality (formal, casual, slang...)
...
Dimension 768: Other patterns learned during training

These dimensions are learned automatically!
Not explicitly programmed!
```

### Visualizing High-Dimensional Embeddings

```
We can't visualize 768 dimensions, so we use tricks:

Technique 1: t-SNE (t-Distributed Stochastic Neighbor Embedding)
Compress 768 → 2 dimensions while preserving relationships
Plot on 2D graph

Result:
- Similar words cluster together
- "King", "Queen", "Prince" group together
- "Dog", "Cat", "Mouse" group together
- Related concepts near each other
```

### Word Analogies in Embedding Space

```
Word2Vec discovered beautiful relationships:

Vector Arithmetic:
King - Man + Woman ≈ Queen
(You subtract "male-ness", add "female-ness")

Paris - France + Germany ≈ Berlin
(You substitute the country)

Playing - Play + Walked ≈ Walking
(You change the tense)

These relationships emerge from the embeddings!
```

---

## Attention Mechanism: The Heart of Transformers

### What Does Attention Do?

**Purpose**: Figure out which words should "look at" which other words to understand context.

**Analogy**: When you read a sentence, you don't focus on every word equally. You focus on relevant words. Attention does this automatically.

### Simple Example

```
Sentence: "The bank executive was tired because the bank had too many customers"

When processing the second "bank":
Without Attention:
- Treat "bank" (pos 2) and "bank" (pos 8) equally
- Don't understand they have different meanings
→ Confusion!

With Attention:
- Word "bank" at position 8 looks back at context
- HIGH attention to "had" (position 7) - verbs are key
- HIGH attention to "customers" (position 11) - reveals meaning
- HIGH attention to "many" (position 10)
- LOW attention to "executive" (position 2) - different meaning

Result: Understands this "bank" = financial institution, not riverbank!
```

### The Three Concepts: Query, Key, Value

#### Query (Q): "What am I looking for?"
```
Each word asks: "What information do I need?"

Example: Processing "bank" at position 8
Query: "I'm looking for context about financial activities"
```

#### Key (K): "What information do I contain?"
```
Each word broadcasts: "Here's what I contain"

Examples:
"had" broadcasts: "I'm a past-tense action verb"
"customers" broadcasts: "I'm a people-related noun"
"executive" broadcasts: "I'm a professional person"
"riverbank" broadcasts: "I'm geography-related"
```

#### Value (V): "Here's the information I contain"
```
If a Query matches this Key, use my Value information

If word asks: "Give me action context"
And "had" says: "I have action context!"
Then "had" gives: Its embedding with action information
```

### Attention in 5 Steps

#### Step 1: Create Q, K, V

```
Input embedding: x = [0.5, 0.3, 0.2, 0.8, ...]  (4096 dimensions)

Create three versions:
Query (Q) = Weight_Q × x + bias_Q
Key (K) = Weight_K × x + bias_K
Value (V) = Weight_V × x + bias_V

Example numbers:
x = [0.5, 0.3, 0.2, 0.8]
Q = [0.6, 0.2, 0.5, ...]  (128 dimensions for Llama 2)
K = [0.4, 0.7, 0.1, ...]  (128 dimensions)
V = [0.3, 0.5, 0.6, ...]  (4096 dimensions)
```

#### Step 2: Calculate Attention Scores

```
Score = Q · K^T / √(dimension)

For word "bank" (position 8):
Q_bank = [0.6, 0.2, 0.5, ...]

Score with "had":
K_had = [0.4, 0.7, 0.1, ...]
Score = (0.6×0.4 + 0.2×0.7 + 0.5×0.1) / √128
      = (0.24 + 0.14 + 0.05) / 11.3
      = 0.43 / 11.3
      = 0.038

Score with "customers":
K_customers = [0.5, 0.8, 0.2, ...]
Score = (0.6×0.5 + 0.2×0.8 + 0.5×0.2) / √128
      = (0.30 + 0.16 + 0.10) / 11.3
      = 0.56 / 11.3
      = 0.050  ← Higher!

Score with "executive":
K_executive = [0.3, 0.4, 0.3, ...]
Score = (0.6×0.3 + 0.2×0.4 + 0.5×0.3) / √128
      = (0.18 + 0.08 + 0.15) / 11.3
      = 0.41 / 11.3
      = 0.036  ← Lower!

Scores: [0.038, 0.045, 0.050, 0.036, ...]
(One score for each word in the sentence)
```

#### Step 3: Convert Scores to Probabilities (Softmax)

```
Raw scores: [0.038, 0.045, 0.050, 0.036, 0.040, 0.042, ...]

Apply softmax to convert to probabilities:
e^0.038 = 1.039
e^0.045 = 1.046
e^0.050 = 1.051
...
Sum all = 11.5

Probabilities:
Position 1 ("The"): 1.039 / 11.5 = 0.090  (9%)
Position 2 ("bank"): 1.046 / 11.5 = 0.091  (9%)
Position 3 ("executive"): 1.051 / 11.5 = 0.091  (9%)
...
Position 7 ("had"): Higher prob (let's say 0.15 = 15%)
Position 11 ("customers"): Higher prob (let's say 0.18 = 18%)
...

Total: 100% (sums to 1.0)
```

#### Step 4: Multiply by Values

```
Attention weights: [0.09, 0.09, 0.09, ..., 0.15, ..., 0.18, ...]

Values from each word:
V_the = [0.1, 0.2, 0.3, ...]
V_bank = [0.2, 0.1, 0.4, ...]
V_had = [0.7, 0.6, 0.5, ...]
V_customers = [0.8, 0.7, 0.6, ...]
...

Weighted combination:
Output = 0.09×V_the + 0.09×V_bank + ... + 0.15×V_had + 0.18×V_customers + ...
       = [0.09×0.1 + 0.09×0.2 + ... + 0.15×0.7 + 0.18×0.8 + ...]
       = [computed vector reflecting high-attention items]

Result: "bank" embedding now enriched with context about financial activities!
```

#### Step 5: Output

```
Output embedding = Weighted sum of all values
                 = Focused on relevant words
                 = Understanding context!

This output goes to the next layer
Next layer sees enriched embeddings with context
```

### Why Divide by √(dimension)?

```
Without scaling:
Q = [0.6, 0.2, 0.5, 0.1, ..., 0.4]  (128 dims)
K = [0.4, 0.7, 0.1, 0.2, ..., 0.3]

Score = 0.6×0.4 + 0.2×0.7 + 0.5×0.1 + 0.1×0.2 + ... + 0.4×0.3
      = Very large number (sum of 128 multiplications)

Problem:
Large scores → softmax produces extreme probabilities
[0.99, 0.01, 0.00, 0.00, ...]
→ Only attends to one word!

With scaling by √128 = 11.3:
Score / 11.3 = smaller number
softmax produces balanced probabilities
[0.15, 0.14, 0.13, 0.12, ...]
→ Attends to multiple words!

Result: More informative attention!
```

---

## Multi-Head Attention: Multiple Perspectives

### Why Multiple Heads?

**Problem with single attention**: Only learns ONE type of relationship

**Solution**: Multiple attention heads learn different types simultaneously!

### How It Works

```
Single Head Attention:
Input Embeddings
   ↓
[Attention with 1 set of Q, K, V]
   ↓
Output (one perspective)

Multi-Head Attention (8 heads):
Input Embeddings
   ├→ Head 1: [Attention Q1, K1, V1] → learns syntax
   ├→ Head 2: [Attention Q2, K2, V2] → learns semantics
   ├→ Head 3: [Attention Q3, K3, V3] → learns coreference
   ├→ Head 4: [Attention Q4, K4, V4] → learns position
   ├→ Head 5: [Attention Q5, K5, V5] → learns tense
   ├→ Head 6: [Attention Q6, K6, V6] → learns relations
   ├→ Head 7: [Attention Q7, K7, V7] → learns structure
   └→ Head 8// filepath: /Users/mayankvashisht/Desktop/AI-ML/AI-ML/Transformers/Transformers_Complete_Guide.md
# Complete Guide to Transformers and Attention Mechanisms

## Table of Contents
1. [What is a Transformer?](#what-is-a-transformer)
2. [Tokenization: Breaking Down Text](#tokenization)
3. [Embedding Vectors: Words as Numbers](#embedding-vectors)
4. [Attention Mechanism: The Heart of Transformers](#attention-mechanism)
5. [Multi-Head Attention: Multiple Perspectives](#multi-head-attention)
6. [Geometry of High-Dimensional Spaces](#geometry-of-spaces)
7. [Query, Key, Value Spaces](#query-key-value-spaces)
8. [Complete Transformer Architecture](#complete-architecture)

---

## What is a Transformer?

### Simple Definition

A **Transformer** is a neural network architecture that processes sequences of data (text, audio, images) by learning which parts to focus on using an **attention mechanism**.

**Key Innovation**: The attention mechanism allows the network to focus on relevant parts of the input, regardless of distance.

### The "Attention is All You Need" Paper (2017)

```
Before Transformers:
- RNNs/LSTMs processed sequentially: word 1 → word 2 → word 3 → ...
- Slow to train (can't parallelize)
- Bad at understanding distant relationships

The Transformer (2017):
- Process ALL words at once (parallel)
- Each word attends to all other words
- Much faster training
- Better understanding of context
- Revolutionized NLP!

Modern LLMs (GPT, Claude, Llama):
All built on transformer architecture
```

### Why Transformers Won

```
Processing Speed Comparison:

RNN (Sequential):
Word 1 → Word 2 → Word 3 → Word 4 → Word 5
Takes: 5 steps (slow!)

Transformer (Parallel):
Word 1 ─┐
Word 2 ─├→ Process all at once!
Word 3 ─┤
Word 4 ─┤
Word 5 ─┘
Takes: 1 step (100x faster!)

Result:
- Training time: From weeks to days
- Can handle longer sequences
- Scales to billions of parameters
```

---

## Tokenization: Breaking Down Text

### What is a Token?

A **token** is a small unit of text that the model processes. It's not characters, it's chunks that make sense together.

### Why Not Just Use Characters?

#### The Character Approach (Bad Idea ❌)

```
Text: "Hello, world!"

Characters:
['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']
13 items to process

Problems:
1. TOO MANY items (waste of computation)
2. Loses meaning ("Hel" by itself is meaningless)
3. Hard to learn patterns (need to learn "H" + "e" + "l" + "l" + "o" = "hello")
4. Inefficient use of attention (attending to each character)
```

#### The Token Approach (Good Idea ✓)

```
Text: "Hello, world!"

Tokens:
['Hello', ',', 'world', '!']
4 items to process

Benefits:
1. Fewer items (efficient computation)
2. Preserves meaning ("Hello" is a word)
3. Easier to learn ("Hello" → greeting)
4. Efficient attention (attend to meaningful units)
```

### How Tokenization Works

#### Byte-Pair Encoding (BPE) - Most Common

```
Step 1: Start with all characters
"hello world" = ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

Step 2: Find most frequent pair
'l' + 'l' appears most often
Merge: ['h', 'e', 'll', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

Step 3: Repeat
'o' + ' ' appears often
Merge: ['h', 'e', 'll', 'o ', 'w', 'o', 'r', 'l', 'd']

Step 4: Keep merging
Eventually: ['hello', ' ', 'world']

Result: Vocabulary of tokens!
```

#### Real Example: GPT Tokenization

```
Text: "I love pizza"

GPT Tokens:
['I', ' love', ' pizza']  (3 tokens)

Text: "Supercalifragilisticexpialidocious"

Tokens:
['Super', 'cal', 'ifrag', 'ilist', 'ic', 'exp', 'ial', 'i', 'doc', 'ious']  (10 tokens)

Rule of thumb:
1 token ≈ 4 characters in English
100 words ≈ 75 tokens
```

### Tokenization for Different Modalities

#### Text Tokens
```
"The quick brown fox"
→ ['The', ' quick', ' brown', ' fox']
```

#### Audio Tokens
```
Speech waveform (44.1 kHz audio)
Split into 20ms chunks
Each chunk = 1 token
→ Allows processing speech like text!
```

#### Image Tokens
```
Image (224×224 pixels)
Divide into 16×16 patches
Each patch = 1 token
Total: (224/16) × (224/16) = 196 tokens

Image becomes a sequence!
→ Vision Transformers (ViT)
```

---

## Embedding Vectors: Words as Numbers

### What is an Embedding?

An **embedding** is a list of numbers that represents the meaning of a word.

**Analogy**: Instead of just having the word "king", we represent it as a point in multi-dimensional space where related words are nearby.

### Simple Example: 2D Embedding

```
Imagine a simple 2D world:

Axis 1: Masculine ← → Feminine
Axis 2: Royalty ← → Commoner

Word positions:
King:   (0.8,  0.9)   ← Masculine, Royal
Queen:  (-0.8, 0.9)   ← Feminine, Royal
Prince: (0.7,  0.7)   ← Masculine, Less royal
Girl:   (-0.9, 0.2)   ← Feminine, Less royal
Boy:    (0.9,  0.2)   ← Masculine, Less royal

Relationships:
King - Man ≈ Queen - Woman
(0.8, 0.9) - (0.8, 0.1) ≈ (-0.8, 0.9) - (-0.8, 0.1)
(0.0, 0.8) ≈ (0.0, 0.8) ✓ Works!
```

### Real Embeddings: 768 or 12,288 Dimensions

```
Modern models use MANY dimensions:

BERT: 768-dimensional embeddings
GPT-2: 768-dimensional embeddings
Claude: 4,096-dimensional embeddings
Llama 2 70B: 4,096-dimensional embeddings

Why so many dimensions?
More dimensions = more information = better representations
```

### What Information is Encoded?

```
In a 768-dimensional embedding, different dimensions might represent:

Dimension 1: Part of speech (noun, verb, adjective...)
Dimension 2: Gender (masculine, feminine, neutral...)
Dimension 3: Concreteness (abstract vs physical...)
Dimension 4: Sentiment (positive, negative, neutral...)
Dimension 5: Formality (formal, casual, slang...)
...
Dimension 768: Other patterns learned during training

These dimensions are learned automatically!
Not explicitly programmed!
```

### Visualizing High-Dimensional Embeddings

```
We can't visualize 768 dimensions, so we use tricks:

Technique 1: t-SNE (t-Distributed Stochastic Neighbor Embedding)
Compress 768 → 2 dimensions while preserving relationships
Plot on 2D graph

Result:
- Similar words cluster together
- "King", "Queen", "Prince" group together
- "Dog", "Cat", "Mouse" group together
- Related concepts near each other
```

### Word Analogies in Embedding Space

```
Word2Vec discovered beautiful relationships:

Vector Arithmetic:
King - Man + Woman ≈ Queen
(You subtract "male-ness", add "female-ness")

Paris - France + Germany ≈ Berlin
(You substitute the country)

Playing - Play + Walked ≈ Walking
(You change the tense)

These relationships emerge from the embeddings!
```

---

## Attention Mechanism: The Heart of Transformers

### What Does Attention Do?

**Purpose**: Figure out which words should "look at" which other words to understand context.

**Analogy**: When you read a sentence, you don't focus on every word equally. You focus on relevant words. Attention does this automatically.

### Simple Example

```
Sentence: "The bank executive was tired because the bank had too many customers"

When processing the second "bank":
Without Attention:
- Treat "bank" (pos 2) and "bank" (pos 8) equally
- Don't understand they have different meanings
→ Confusion!

With Attention:
- Word "bank" at position 8 looks back at context
- HIGH attention to "had" (position 7) - verbs are key
- HIGH attention to "customers" (position 11) - reveals meaning
- HIGH attention to "many" (position 10)
- LOW attention to "executive" (position 2) - different meaning

Result: Understands this "bank" = financial institution, not riverbank!
```

### The Three Concepts: Query, Key, Value

#### Query (Q): "What am I looking for?"
```
Each word asks: "What information do I need?"

Example: Processing "bank" at position 8
Query: "I'm looking for context about financial activities"
```

#### Key (K): "What information do I contain?"
```
Each word broadcasts: "Here's what I contain"

Examples:
"had" broadcasts: "I'm a past-tense action verb"
"customers" broadcasts: "I'm a people-related noun"
"executive" broadcasts: "I'm a professional person"
"riverbank" broadcasts: "I'm geography-related"
```

#### Value (V): "Here's the information I contain"
```
If a Query matches this Key, use my Value information

If word asks: "Give me action context"
And "had" says: "I have action context!"
Then "had" gives: Its embedding with action information
```

### Attention in 5 Steps

#### Step 1: Create Q, K, V

```
Input embedding: x = [0.5, 0.3, 0.2, 0.8, ...]  (4096 dimensions)

Create three versions:
Query (Q) = Weight_Q × x + bias_Q
Key (K) = Weight_K × x + bias_K
Value (V) = Weight_V × x + bias_V

Example numbers:
x = [0.5, 0.3, 0.2, 0.8]
Q = [0.6, 0.2, 0.5, ...]  (128 dimensions for Llama 2)
K = [0.4, 0.7, 0.1, ...]  (128 dimensions)
V = [0.3, 0.5, 0.6, ...]  (4096 dimensions)
```

#### Step 2: Calculate Attention Scores

```
Score = Q · K^T / √(dimension)

For word "bank" (position 8):
Q_bank = [0.6, 0.2, 0.5, ...]

Score with "had":
K_had = [0.4, 0.7, 0.1, ...]
Score = (0.6×0.4 + 0.2×0.7 + 0.5×0.1) / √128
      = (0.24 + 0.14 + 0.05) / 11.3
      = 0.43 / 11.3
      = 0.038

Score with "customers":
K_customers = [0.5, 0.8, 0.2, ...]
Score = (0.6×0.5 + 0.2×0.8 + 0.5×0.2) / √128
      = (0.30 + 0.16 + 0.10) / 11.3
      = 0.56 / 11.3
      = 0.050  ← Higher!

Score with "executive":
K_executive = [0.3, 0.4, 0.3, ...]
Score = (0.6×0.3 + 0.2×0.4 + 0.5×0.3) / √128
      = (0.18 + 0.08 + 0.15) / 11.3
      = 0.41 / 11.3
      = 0.036  ← Lower!

Scores: [0.038, 0.045, 0.050, 0.036, ...]
(One score for each word in the sentence)
```

#### Step 3: Convert Scores to Probabilities (Softmax)

```
Raw scores: [0.038, 0.045, 0.050, 0.036, 0.040, 0.042, ...]

Apply softmax to convert to probabilities:
e^0.038 = 1.039
e^0.045 = 1.046
e^0.050 = 1.051
...
Sum all = 11.5

Probabilities:
Position 1 ("The"): 1.039 / 11.5 = 0.090  (9%)
Position 2 ("bank"): 1.046 / 11.5 = 0.091  (9%)
Position 3 ("executive"): 1.051 / 11.5 = 0.091  (9%)
...
Position 7 ("had"): Higher prob (let's say 0.15 = 15%)
Position 11 ("customers"): Higher prob (let's say 0.18 = 18%)
...

Total: 100% (sums to 1.0)
```

#### Step 4: Multiply by Values

```
Attention weights: [0.09, 0.09, 0.09, ..., 0.15, ..., 0.18, ...]

Values from each word:
V_the = [0.1, 0.2, 0.3, ...]
V_bank = [0.2, 0.1, 0.4, ...]
V_had = [0.7, 0.6, 0.5, ...]
V_customers = [0.8, 0.7, 0.6, ...]
...

Weighted combination:
Output = 0.09×V_the + 0.09×V_bank + ... + 0.15×V_had + 0.18×V_customers + ...
       = [0.09×0.1 + 0.09×0.2 + ... + 0.15×0.7 + 0.18×0.8 + ...]
       = [computed vector reflecting high-attention items]

Result: "bank" embedding now enriched with context about financial activities!
```

#### Step 5: Output

```
Output embedding = Weighted sum of all values
                 = Focused on relevant words
                 = Understanding context!

This output goes to the next layer
Next layer sees enriched embeddings with context
```

### Why Divide by √(dimension)?

```
Without scaling:
Q = [0.6, 0.2, 0.5, 0.1, ..., 0.4]  (128 dims)
K = [0.4, 0.7, 0.1, 0.2, ..., 0.3]

Score = 0.6×0.4 + 0.2×0.7 + 0.5×0.1 + 0.1×0.2 + ... + 0.4×0.3
      = Very large number (sum of 128 multiplications)

Problem:
Large scores → softmax produces extreme probabilities
[0.99, 0.01, 0.00, 0.00, ...]
→ Only attends to one word!

With scaling by √128 = 11.3:
Score / 11.3 = smaller number
softmax produces balanced probabilities
[0.15, 0.14, 0.13, 0.12, ...]
→ Attends to multiple words!

Result: More informative attention!
```

---

## Multi-Head Attention: Multiple Perspectives

### Why Multiple Heads?

**Problem with single attention**: Only learns ONE type of relationship

**Solution**: Multiple attention heads learn different types simultaneously!

### How It Works

```
Single Head