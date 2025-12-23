# Word Embeddings: A Complete Guide from Basics to Advanced

A comprehensive guide covering word embeddings, Word2Vec, contextual embeddings, and practical implementations with detailed explanations, comparisons, and real-world examples.

---

## Table of Contents

1. [Introduction to Word Embeddings](#1-introduction-to-word-embeddings)
2. [Historical Context: From Bag of Words to Embeddings](#2-historical-context-from-bag-of-words-to-embeddings)
3. [Word2Vec: The Game Changer](#3-word2vec-the-game-changer)
4. [Word2Vec Architectures](#4-word2vec-architectures)
5. [Word2Vec Implementation Guide](#5-word2vec-implementation-guide)
6. [Understanding Word2Vec Arguments](#6-understanding-word2vec-arguments)
7. [Gensim Library](#7-gensim-library)
8. [Word2Vec Limitations and Problems](#8-word2vec-limitations-and-problems)
9. [Advanced Topics](#9-advanced-topics)
10. [Transformers and Contextual Embeddings](#10-transformers-and-contextual-embeddings)
11. [Comparison: BoW vs TF-IDF vs Word2Vec](#11-comparison-bow-vs-tfidf-vs-word2vec)
12. [Interview Questions](#12-interview-questions)
13. [Practical Applications](#13-practical-applications)
14. [Summary](#14-summary)

---

## 1. Introduction to Word Embeddings

### What are Word Embeddings?

**Definition:** Word embeddings are dense numerical vectors that represent words in a continuous vector space, where semantically similar words are close to each other.

**Key Characteristics:**
- **Dense:** Typically 50-300 dimensions (vs. sparse BoW with thousands)
- **Semantic:** Captures meaning and relationships between words
- **Learned:** Derived from large text corpora using neural networks
- **Continuous:** Values are real numbers, not just 0s and 1s

### Simple Example

```
One-Hot Encoding (Sparse):
word "king" = [0, 0, 1, 0, 0, ..., 0]  (10,000 dimensions, one 1)
word "queen" = [0, 0, 0, 1, 0, ..., 0]

Word Embeddings (Dense):
word "king" = [0.2, -0.5, 0.8, 0.3, ...]  (50 dimensions, many values)
word "queen" = [0.1, -0.4, 0.9, 0.25, ...]
word "prince" = [0.15, -0.45, 0.75, 0.28, ...]

Notice: "queen" and "prince" are closer to "king" than random words
```

### Why Word Embeddings Matter

```python
# Bag of Words: No semantic understanding
"king" and "queen" are completely unrelated vectors

# Word Embeddings: Capture semantics
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
# This is the famous analogy property!
```

---

## 2. Historical Context: From Bag of Words to Embeddings

### Evolution of Text Representation

#### Stage 1: One-Hot Encoding
```python
vocabulary = ["cat", "dog", "bird", "fish"]

"cat" = [1, 0, 0, 0]
"dog" = [0, 1, 0, 0]
"bird" = [0, 0, 1, 0]

# Problems:
# - Sparse (mostly zeros)
# - No semantic similarity
# - Vocab size = dimensions
# - Can't handle new words
```

#### Stage 2: Bag of Words (BoW)
```python
# Document: "the cat and the dog are playing"
vocabulary = ["the", "cat", "dog", "are", "and", "playing"]

BoW representation = [2, 1, 1, 1, 1, 1]
# Count how many times each word appears

# Improvement: Captures word frequency
# Problems:
# - Still sparse
# - No semantic understanding
# - Order doesn't matter ("cat ate dog" = "dog ate cat")
# - Heavy words dominate
```

#### Stage 3: TF-IDF (Term Frequency-Inverse Document Frequency)

**Formula:**
$$\text{TF-IDF}(word, doc) = TF(word, doc) \times IDF(word)$$

Where:
- $TF(word, doc) = \frac{\text{count of word in doc}}{\text{total words in doc}}$
- $IDF(word) = \log\left(\frac{\text{total documents}}{\text{documents containing word}}\right)$

**Example:**
```
Document 1: "the cat sat on the mat"
Document 2: "the dog sat on the log"
Document 3: "cats and dogs are pets"

Word "the":
- Appears in all 3 documents
- IDF("the") = log(3/3) = 0 (not important)
- TF-IDF is LOW

Word "cat":
- Appears in 1 document
- IDF("cat") = log(3/1) = 1.099 (important)
- TF-IDF is HIGH

# Improvement: Emphasizes important words
# Problems:
# - Still sparse
# - No semantic relationships
# - Fixed vocabulary
```

```python
# Code Example: TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print(f"Shape: {tfidf_matrix.shape}")  # (3, vocab_size)
print(f"Sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.2%}")
# Output: Very sparse (lots of zeros)

# Get feature names
features = vectorizer.get_feature_names_out()
print(f"Features: {features}")
```

#### Stage 4: Word Embeddings (Word2Vec, GloVe, FastText)

```python
# Dense representation - captures semantics!
"cat" = [0.25, -0.48, 0.81, -0.12, 0.56, ...]  (50 dims)
"dog" = [0.28, -0.45, 0.78, -0.10, 0.54, ...]  (50 dims)
"king" = [0.15, 0.75, -0.20, 0.62, -0.30, ...]
"queen" = [0.17, 0.73, -0.22, 0.64, -0.28, ...]
"man" = [0.12, 0.68, -0.18, 0.58, -0.25, ...]
"woman" = [0.19, 0.78, -0.24, 0.66, -0.32, ...]

# Notice the relationships:
# "king" - "man" + "woman" ≈ "queen"
# "cat" is close to "dog"
# "king" is far from "cat"
```

### Comparison Table

| Aspect | One-Hot | BoW | TF-IDF | Word Embeddings |
|--------|---------|-----|--------|-----------------|
| **Dimensions** | Vocab size | Vocab size | Vocab size | Fixed (50-300) |
| **Sparsity** | Very sparse | Sparse | Sparse | Dense |
| **Semantic** | ❌ | ❌ | ❌ | ✅ |
| **Context** | ❌ | Partial | ❌ | ✅ |
| **Training** | None | None | None | Requires corpus |
| **OOV handling** | ❌ | ❌ | ❌ | ⚠️ (depends on model) |
| **Use case** | Small vocab | Document class | Search | Modern NLP |

---

## 3. Word2Vec: The Game Changer

### Historical Background

**Question: How did Google improve search in its early days?**

Before Word2Vec, search engines relied on exact keyword matching:
```
User searches: "artificial intelligence"
Only found pages with exact phrase "artificial intelligence"
Missed pages with "machine learning" or "deep learning"
```

**The Problem:** They couldn't understand semantic similarity.

**The Solution:** Word embeddings! If "artificial intelligence" and "machine learning" have similar embeddings, they can be matched together.

### What is Word2Vec?

**Core Idea:** "You shall know a word by the company it keeps." - J.R. Firth

**Principle:** Words that appear in similar contexts have similar meanings.

```
Sentence: "The cat sat on the mat"
           The [cat] sat on the mat
           
Context window (size 2):
- Left context: ["The"]
- Right context: ["sat", "on"]

Sentence: "The dog sat on the rug"
           The [dog] sat on the rug

Context window:
- Left context: ["The"]
- Right context: ["sat", "on"]

Because "cat" and "dog" appear in similar contexts,
Word2Vec learns that they should have similar vectors!
```

### How Word2Vec Works: High-Level Overview

**The Learning Process:**

1. **Sliding Window:** Move a window through the text
2. **Extract Context:** For each word, note surrounding words
3. **Train Network:** Use a simple neural network to predict context from word
4. **Learn Embeddings:** The hidden layer weights become word vectors

```
Raw Text: "the quick brown fox jumps over the lazy dog"
Window size: 2

Step 1: [the, quick] brown [fox, jumps]
        → Input: "brown", Predict: ["the", "quick", "fox", "jumps"]

Step 2: [quick, brown] fox [jumps, over]
        → Input: "fox", Predict: ["quick", "brown", "jumps", "over"]

... continue for entire text

Through thousands of these examples, the network learns:
- "quick" and "fast" should be similar
- "brown" and "red" should be similar
- "dog" and "cat" should be similar
```

### Mathematical Framework

**Input:** Word index $w_i$

**Output:** Context words within window (typically 2-5 positions before/after)

**Neural Network:**
- Input layer: One-hot encoded word (vocabulary size)
- Hidden layer: Dense vector (embedding dimension) ← **This is our word vector!**
- Output layer: Softmax over vocabulary (predict context words)

```
Input word: "king"
    ↓
One-hot: [0, 0, ..., 1, ..., 0]  (10,000 dimensions)
    ↓
Hidden layer (weights are learned): [0.15, 0.75, -0.20, ...]  (50 dimensions)
    ↓
Output: Probability distribution over vocabulary
    → Should predict words in context: "the", "ruled", "throne", etc.
```

**Loss Function:** Negative log-likelihood of predicting correct context words

$$L = -\sum_{t=1}^{T} \sum_{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} | w_t)$$

Where:
- $T$ = corpus size
- $m$ = window size
- $w_t$ = target word
- $w_{t+j}$ = context words

---

## 4. Word2Vec Architectures

### Architecture 1: Skip-Gram Model

**Idea:** Predict context words from a target word

```
Input: "cat"
↓
Embedding: [0.25, -0.48, 0.81, -0.12, ...]
↓
Output: Predict ["the", "sat", "on", "mat"]

This is like learning: Given "cat", what words are nearby?
```

**Advantages:**
- Works well with small datasets
- Better for rare words
- Captures more semantic relationships

**Disadvantages:**
- Slower training
- More parameters

**Use Case:** When you want rich semantic relationships

```python
# Skip-Gram in Gensim
model = Word2Vec(
    sentences=tokenized_text,
    sg=1,  # Skip-Gram
    vector_size=100,
    window=5
)
```

### Architecture 2: CBOW (Continuous Bag of Words)

**Idea:** Predict a target word from its context

```
Input: ["the", "sat", "on", "mat"]
↓
Average embeddings
↓
Output: Predict "cat"

This is like learning: Given surrounding words, what's the middle word?
```

**Advantages:**
- Faster training
- Better with small datasets
- Smoother representations

**Disadvantages:**
- Less semantic richness
- Struggles with rare words

**Use Case:** When you need speed or have limited data

```python
# CBOW in Gensim
model = Word2Vec(
    sentences=tokenized_text,
    sg=0,  # CBOW
    vector_size=100,
    window=5
)
```

### Visual Comparison

```
SKIP-GRAM:
word "cat"
    ↓
  [vector]
    ↓
Predict: "sat", "on", "the", "mat"
(1 input → many outputs)

CBOW:
"sat", "on", "the", "mat"
    ↓
Average vectors
    ↓
Predict: "cat"
(Many inputs → 1 output)
```

### Which to Choose?

```python
# Skip-Gram: Better semantics, slower
model_sg = Word2Vec(sentences, sg=1, vector_size=300)

# CBOW: Faster, better for small data
model_cbow = Word2Vec(sentences, sg=0, vector_size=300)

# Performance comparison:
# - Skip-Gram: Better for analogies ("king" - "man" + "woman" = "queen")
# - CBOW: Better for downstream tasks with limited training data
```

---

## 5. Word2Vec Implementation Guide

### Step 1: Prepare Your Data

```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required data
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text = """
Word embeddings are dense numerical vectors. 
They represent words in a continuous vector space. 
Semantically similar words are close to each other.
Word2Vec learns embeddings from text corpora.
It uses neural networks to capture word meaning.
"""

# Step 1: Tokenize into sentences
sentences = sent_tokenize(text)
print(f"Sentences: {sentences[:2]}")

# Step 2: Tokenize each sentence into words
tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]
print(f"Tokenized: {tokenized_sentences[0]}")

# Step 3: Remove stopwords and punctuation (optional)
stop_words = set(stopwords.words('english'))
filtered_sentences = [
    [word for word in sent if word.isalpha() and word not in stop_words]
    for sent in tokenized_sentences
]
print(f"Filtered: {filtered_sentences[0]}")
```

### Step 2: Train Word2Vec Model

```python
from gensim.models import Word2Vec

# Train Word2Vec
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,      # Embedding dimension
    window=5,             # Context window size
    min_count=2,          # Minimum word frequency
    workers=4,            # Number of threads
    sg=1,                 # 1=Skip-Gram, 0=CBOW
    epochs=10             # Training iterations
)

# Check model info
print(f"Vocabulary size: {len(model.wv)}")
print(f"Vector dimension: {model.wv.vector_size}")
```

### Step 3: Explore Learned Embeddings

```python
# Get word vector
word = "embeddings"
if word in model.wv:
    vector = model.wv[word]
    print(f"Vector for '{word}': {vector[:10]}...")  # First 10 dimensions
    print(f"Vector shape: {vector.shape}")

# Find similar words
print("\nMost similar to 'embeddings':")
similar_words = model.wv.most_similar('embeddings', topn=5)
for word, similarity in similar_words:
    print(f"  {word}: {similarity:.4f}")

# Word analogies
print("\nWord Analogies:")
# "king" is to "man" as "queen" is to ?
if all(w in model.wv for w in ['king', 'man', 'queen', 'woman']):
    result = model.wv.most_similar(
        positive=['queen', 'man'],
        negative=['woman'],
        topn=1
    )
    print(f"  queen:woman :: king:? → {result}")

# Calculate similarity
similarity = model.wv.similarity('embeddings', 'vectors')
print(f"\nSimilarity('embeddings', 'vectors'): {similarity:.4f}")

# Find most dissimilar (odd one out)
words = ['king', 'queen', 'man', 'table']
odd_one = model.wv.doesnt_match(words)
print(f"\nOdd one out in {words}: {odd_one}")
```

### Step 4: Use Embeddings in Your Pipeline

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Example: Document classification using Word2Vec

documents = [
    ("this movie is great", "positive"),
    ("i love this film", "positive"),
    ("excellent acting", "positive"),
    ("terrible movie", "negative"),
    ("i hate this", "negative"),
    ("worst film ever", "negative"),
]

# Convert documents to vectors (average word vectors)
X = []
y = []

for doc, label in documents:
    words = word_tokenize(doc.lower())
    
    # Get vectors for words in document
    vectors = [model.wv[w] for w in words if w in model.wv]
    
    if vectors:
        # Average the vectors
        doc_vector = np.mean(vectors, axis=0)
        X.append(doc_vector)
        y.append(label)

X = np.array(X)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Classification accuracy: {accuracy:.2%}")
```

### Step 5: Save and Load Models

```python
# Save model
model.save('word2vec_model.model')
model.wv.save('word2vec_vectors.wordvectors')

# Load model
from gensim.models import Word2Vec
loaded_model = Word2Vec.load('word2vec_model.model')

# Load vectors
from gensim.models import KeyedVectors
loaded_vectors = KeyedVectors.load('word2vec_vectors.wordvectors')
```

---

## 6. Understanding Word2Vec Arguments

### Complete Gensim Word2Vec Parameters

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences,              # List of tokenized sentences
    
    # Core parameters
    vector_size=100,        # Embedding dimension (50-300 typical)
    window=5,              # Context window size (words to look left/right)
    min_count=2,           # Minimum word frequency to include
    workers=4,             # Number of CPU threads
    sg=1,                  # 1=Skip-Gram, 0=CBOW
    epochs=5,              # Number of training iterations
    
    # Advanced parameters
    negative=5,            # Number of negative samples (for efficiency)
    alpha=0.025,          # Initial learning rate
    min_alpha=0.0001,     # Final learning rate
    seed=42,              # Random seed for reproducibility
    max_vocab_size=None,  # Maximum vocabulary size
    sample=1e-3,          # Downsampling threshold for frequent words
    hs=0,                 # 1=hierarchical softmax, 0=negative sampling
    sorted_vocab=1,       # Sort vocabulary by frequency
)
```

### Parameter Explanations

#### 1. **vector_size** (Embedding Dimension)

```python
# Small dimensions (50-100):
# - Pros: Fast training, less memory, good for small datasets
# - Cons: May miss semantic nuances
model_small = Word2Vec(sentences, vector_size=50)

# Medium dimensions (100-300):
# - Balanced for most tasks
model_medium = Word2Vec(sentences, vector_size=200)

# Large dimensions (300+):
# - Pros: Capture fine-grained semantics
# - Cons: Slower, needs more data, risk of overfitting
model_large = Word2Vec(sentences, vector_size=500)

# Practical recommendation: Start with 100, adjust based on:
# - Dataset size (larger dataset → larger vectors)
# - Task complexity
# - Available memory
```

#### 2. **window** (Context Window Size)

```python
# Sentence: "the quick brown fox jumps over the lazy dog"
#           
# window=1: Look 1 word left/right
#           "brown" context: ["quick", "fox"]

# window=2: Look 2 words left/right
#           "brown" context: ["the", "quick", "fox", "jumps"]

# window=5: Look 5 words left/right
#           "brown" context: ["the", "quick", "fox", "jumps", "over", "the"]

model_small_window = Word2Vec(sentences, window=2)
model_large_window = Word2Vec(sentences, window=10)

# Small window (2-3):
# - Captures syntactic relationships ("quick" → "brown")
# - "king" similar to adjectives

# Large window (5-10):
# - Captures topical relationships
# - "king" similar to "throne", "crown", "royal"

# Recommendation: window=5 is a good default
```

#### 3. **min_count** (Minimum Frequency)

```python
# Corpus has 1000 unique words, but some appear only once

# min_count=1: Include all words
# - Pros: Complete vocabulary
# - Cons: Rare words have poor embeddings, noise
model_all = Word2Vec(sentences, min_count=1)

# min_count=2: Include words appearing 2+ times
# - Standard choice, balances coverage and quality
model_standard = Word2Vec(sentences, min_count=2)

# min_count=5: Include only words appearing 5+ times
# - Pros: Quality embeddings, less noise
# - Cons: Misses rare but important words
model_filtered = Word2Vec(sentences, min_count=5)

# Recommendation: min_count=2 or 5 depending on corpus size
```

#### 4. **sg** (Skip-Gram vs CBOW)

```python
# Skip-Gram (sg=1)
# Input: word → Output: context words
model_sg = Word2Vec(
    sentences,
    sg=1,           # Skip-Gram
    vector_size=100
)

# CBOW (sg=0)
# Input: context words → Output: word
model_cbow = Word2Vec(
    sentences,
    sg=0,           # CBOW
    vector_size=100
)

# Benchmark results (from Word2Vec paper):
# Skip-Gram:
#   - Better for semantic tasks (analogies)
#   - Slower to train
#   - Better with limited data
#   - Accuracy: ~75% on Google analogies

# CBOW:
#   - Better for downstream tasks
#   - Faster to train (3-10x)
#   - Needs more data
#   - Smoother embeddings

# Choice rule:
# - Use Skip-Gram if you want semantic richness
# - Use CBOW if you prioritize speed
```

#### 5. **workers** (Parallelization)

```python
# Single-threaded (safe, deterministic)
model = Word2Vec(sentences, workers=1, seed=42)

# Multi-threaded (faster, but less reproducible)
model = Word2Vec(sentences, workers=4)  # 4 CPU cores
model = Word2Vec(sentences, workers=8)  # 8 CPU cores

# Recommendation: Use all available cores
# workers = number of CPU cores on your machine
import os
num_cores = os.cpu_count()
model = Word2Vec(sentences, workers=num_cores)
```

#### 6. **epochs** (Training Iterations)

```python
# Train for 1 epoch (go through data once)
model_1 = Word2Vec(sentences, epochs=1)  # Fast but poor quality

# Train for 5 epochs (standard)
model_5 = Word2Vec(sentences, epochs=5)  # Balanced

# Train for 10+ epochs (slow but better quality)
model_10 = Word2Vec(sentences, epochs=10)  # More time, better embeddings

# Diminishing returns:
# Epoch 1: Large improvement
# Epoch 2-5: Moderate improvement
# Epoch 5+: Minimal improvement

# Recommendation: Start with epochs=5, increase if convergence not reached
```

#### 7. **negative** (Negative Sampling)

```python
# Skip basic explanation; this is advanced
# Optimization trick to speed up training

# negative=5: Default, sample 5 negative examples per positive
model = Word2Vec(sentences, negative=5)

# negative=15: More negative samples, better quality but slower
model = Word2Vec(sentences, negative=15)

# Recommendation: Keep default (5) unless you have specific needs
```

#### 8. **sample** (Downsampling Frequent Words)

```python
# Problem: Common words like "the", "and" dominate
# Solution: Randomly discard them during training

# sample=1e-3 (default): Aggressive downsampling
# Probability of keeping word w:
# P(keep) = (sqrt(z/s) + 1) * (s/z)
# where z = frequency, s = sample parameter

model = Word2Vec(sentences, sample=1e-3)

# sample=0: No downsampling (keep all words)
model = Word2Vec(sentences, sample=0)

# Higher sample value = more downsampling
model = Word2Vec(sentences, sample=1e-2)  # Milder

# Recommendation: Keep default (1e-3)
```

### Complete Example with All Parameters

```python
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Prepare sentences
text = "Your large corpus here..."
sentences = sent_tokenize(text)
tokenized = [word_tokenize(s.lower()) for s in sentences]

# Train with explained parameters
model = Word2Vec(
    sentences=tokenized,
    
    # Architecture
    vector_size=100,      # 100 dimensions (good for most tasks)
    window=5,            # 5 words left/right (captures both syntax and topic)
    sg=1,                # Skip-Gram (better semantics)
    
    # Filtering
    min_count=2,         # Words appearing 2+ times
    sample=1e-3,         # Downsample frequent words
    
    # Training
    epochs=5,            # 5 passes through data
    workers=4,           # 4 CPU threads
    negative=5,          # 5 negative samples
    
    # Reproducibility
    seed=42
)

# Verify training
print(f"Vocabulary size: {len(model.wv)}")
print(f"Training time: {model.epochs} epochs")
print(f"Sample vector shape: {model.wv[list(model.wv.index_to_key)[0]].shape}")
```

---

## 7. Gensim Library

### What is Gensim?

Gensim is a Python library specialized in NLP tasks, particularly:
- Training Word2Vec models efficiently
- Loading pre-trained embeddings
- Topic modeling (LDA)
- Document similarity

**Installation:**
```bash
pip install gensim
```

### Key Gensim Classes

#### 1. Word2Vec

```python
from gensim.models import Word2Vec

# Train from scratch
model = Word2Vec(sentences, vector_size=100, window=5)

# Access word vector
vector = model.wv['word']

# Find similar words
model.wv.most_similar('word', topn=5)

# Word analogies
model.wv.most_similar(positive=['king', 'woman'], negative=['man'])

# Calculate similarity
similarity = model.wv.similarity('word1', 'word2')
```

#### 2. KeyedVectors

```python
from gensim.models import KeyedVectors

# Pre-trained embeddings (e.g., from Google)
model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin',
    binary=True
)

# Access methods
similar = model.most_similar('computer')
vector = model['computer']
```

#### 3. FastText (Handles OOV Words)

```python
from gensim.models import FastText

# FastText extends Word2Vec to handle out-of-vocabulary words
model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1
)

# Works even for unseen words!
# Breaks word into character n-grams
vector = model.wv['unknownword']  # Works! (approximated from char-grams)
```

### Complete Gensim Workflow

```python
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

# 1. Prepare and train
sentences = [['hello', 'world'], ['good', 'morning']]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# 2. Explore
print("Similar to 'hello':", model.wv.most_similar('hello', topn=3))

# 3. Save
model.save('my_word2vec.model')
model.wv.save('my_word2vec.vectors')

# 4. Load
loaded_model = Word2Vec.load('my_word2vec.model')
loaded_vectors = KeyedVectors.load('my_word2vec.vectors')

# 5. Use in pipeline
from sklearn.svm import SVC
import numpy as np

# Convert documents to vectors
def doc_to_vector(doc, model, vector_size=100):
    vectors = [model.wv[word] for word in doc.split() if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(vector_size)

doc_vectors = [doc_to_vector(doc, model) for doc, _ in data]
clf = SVC()
clf.fit(doc_vectors, labels)
```

---

## 8. Word2Vec Limitations and Problems

### Problem 1: The Polysemy Problem

**Definition:** One word has multiple meanings depending on context.

**Example:**
```
"bank" has at least 2 meanings:
1. Financial institution: "I went to the bank to withdraw money"
2. Riverbank: "We sat by the bank of the river"

Word2Vec produces ONE vector for "bank"
This vector is an AVERAGE of both meanings
Not ideal for either context!
```

**Why It's a Problem:**

```python
# When we train Word2Vec:
# Context 1: "withdrew", "money", "account" → vector1
# Context 2: "river", "water", "trees" → vector2
# Final vector = mixture of both

# When predicting similar words:
model.wv.most_similar('bank')
# Result might contain both financial and geography words
# Not ideal!
```

**Solution:** Use contextual embeddings (BERT, GPT)

```python
# With BERT:
# Same word "bank" gets different embeddings based on context
# Solved the polysemy problem!
```

### Problem 2: Word Order is Ignored

**Definition:** Word2Vec only considers co-occurrence, not word order within context window.

**Example:**
```
Sentence 1: "The cat ate the mouse"
            window: ate → [cat, the, mouse, the]

Sentence 2: "The mouse ate the cat"
            window: ate → [mouse, the, cat, the]

Word2Vec sees:
- "cat" appears with "ate", "mouse" ✓
- "mouse" appears with "ate", "cat" ✓

But misses:
- "cat ate" vs "ate cat" (different roles!)
- Word order matters for meaning
```

**Why It's a Problem:**

```
"I didn't like the movie" 
vs
"I like the movie"

Word2Vec might see these as similar because they share many words
But the meanings are opposite!
```

**Limitation:** This is inherent to context window approach

```
Sentence: "The quick brown fox jumped"
Word "fox" context [window=2]:
- Left: ["brown", "quick"]  (what was before?)
- Right: ["jumped"]         (what comes after?)
- No distinction which came before/after

Modern fix: Transformers use attention to track order
```

### Problem 3: Inability to Handle New Words (OOV - Out of Vocabulary)

**Definition:** Word2Vec has fixed vocabulary; can't represent unseen words.

**Example:**
```
Training data words: {"cat", "dog", "bird", "fish"}

New text mentions "dinosaur" (not in training)
↓
Can't look up embedding for "dinosaur"
↓
Two options:
1. Ignore it (lose information)
2. Use <UNK> token (generic representation)
3. Skip it entirely

Not ideal!
```

**Why It's a Problem:**

```
Real world:
- New products, brands, slang emerge constantly
- Scientific terms, proper nouns, typos
- Word2Vec can't handle any of these

Example:
Training: "iPhone X is great"
Test: "iPhone 15 is great"
↓
"iPhone 15" not in vocab → can't represent properly
```

**Solution:** Use FastText or Byte-Pair Encoding

```python
from gensim.models import FastText

# FastText: Break words into character n-grams
# "unknown" = ["<un", "unk", "nkn", "kno", "now", "own", "wn>"]

# Even unseen words can be represented!
model = FastText(sentences, vector_size=100)

# "iPhone15" can be approximated from character n-grams
# even if never seen in training
vector = model.wv['iPhone15']  # Works!

# Trade-off: Less semantically pure but more robust
```

### Problem 4: Static Embeddings Don't Capture Context

**Definition:** Each word gets ONE embedding regardless of context.

**Example:**
```
"I caught the bank of the river with my fishing rod"
"The bank of England announced new policies"
"river bank" (multiple meanings)
"bank robber"

All occurrences of "bank" get the SAME vector
But meanings are different!
```

**Why It's a Problem:**

```
Downstream task: Sentiment analysis
Sentence: "That's sick!" (positive: amazing)
↓
Word2Vec embedding for "sick" was trained on:
- "That sick person needs help" (negative)
- "Sick skills!" (positive)
↓
Static embedding confuses the model
Doesn't know which meaning applies here
```

**Solution:** Contextual embeddings (BERT, GPT, ELMo)

```python
# These models generate embeddings CONDITIONED on context
# Same word gets different representations in different contexts

from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text1 = "I caught the bank of the river"
text2 = "I went to the bank to withdraw money"

# Get BERT embeddings (contextual)
with torch.no_grad():
    inputs1 = tokenizer(text1, return_tensors='pt')
    outputs1 = model(**inputs1)
    embedding1 = outputs1.last_hidden_state
    
    inputs2 = tokenizer(text2, return_tensors='pt')
    outputs2 = model(**inputs2)
    embedding2 = outputs2.last_hidden_state

# Same word "bank" gets DIFFERENT embeddings
# in different contexts!
```

### Problem 5: Dominance of Stopwords

**Definition:** Common words overwhelm important words if not filtered.

**Example:**
```
Text: "The quick brown fox jumps over the lazy dog"

Frequency:
- "the": 2 times
- "quick", "brown", "fox": 1 time each
- "jumps", "over", "lazy", "dog": 1 time each

Word2Vec trains on "the" twice as much
↓
"the" gets a high-quality embedding
↓
"the" is closest to other common words
↓
Not useful for downstream tasks!
```

**Why It's a Problem:**

```
Document similarity:
Doc 1: "The cat is sleeping"
Doc 2: "The dog is playing"

Similarity is HIGH because both contain "the", "is"
But semantically different!

Average embedding heavily influenced by stopwords
```

**Solution:** Remove or downsample

```python
from nltk.corpus import stopwords

# Option 1: Remove stopwords before training
stop_words = set(stopwords.words('english'))
filtered = [[w for w in sent if w not in stop_words] for sent in sentences]
model = Word2Vec(filtered, vector_size=100)

# Option 2: Downsample in Word2Vec
model = Word2Vec(sentences, sample=1e-3)  # Downsample frequent words

# Option 3: Use TF-IDF weighting when creating document vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
# TF-IDF automatically downweights common words
```

### Summary: Word2Vec Problems

| Problem | Impact | Solution |
|---------|--------|----------|
| **Polysemy** | Same word, different meanings confuse model | Use contextual embeddings (BERT) |
| **Word Order** | "cat ate dog" same as "dog ate cat" | Use Transformers with attention |
| **OOV Words** | Can't represent unseen words | Use FastText or subword tokenization |
| **Static** | No context-dependent meanings | Use contextual embeddings |
| **Stopwords** | Common words dominate | Remove or downsample |

---

## 9. Advanced Topics

### PCA for Dimensionality Reduction

**Question: Where is PCA used in Word2Vec?**

**Use Case:** Visualizing high-dimensional embeddings in 2D/3D.

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Word2Vec gives 100-300 dimensional vectors
# Can't visualize directly!

# Get vectors for analysis
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
         'cat', 'dog', 'kitten', 'puppy', 'tiger', 'lion']
vectors = np.array([model.wv[word] for word in words])

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Visualize
plt.figure(figsize=(10, 8))
for word, (x, y) in zip(words, vectors_2d):
    plt.scatter(x, y, s=100)
    plt.annotate(word, (x, y), fontsize=10)

plt.title('Word2Vec Embeddings (PCA Projection)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.grid(True)
plt.show()

# Interpretation:
# Points close together = similar meanings
# PCA chooses dimensions that capture most variance
```

**When to Use PCA:**
- Visualizing embeddings
- Reducing to 2D/3D for plots
- Removing noise (keeping top k dimensions)

```python
# Remove noise
pca = PCA(n_components=50)  # Reduce from 100 to 50
reduced_vectors = pca.fit_transform(vectors)

# Variance captured
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")
# Usually 50 dimensions capture 90%+ variance
```

### Combining Multiple Embeddings

```python
# Train multiple Word2Vec models
model_sg = Word2Vec(sentences, sg=1, vector_size=100)
model_cbow = Word2Vec(sentences, sg=0, vector_size=100)

# Combine for richer representation
word = "learning"
combined_vector = np.concatenate([
    model_sg.wv[word],
    model_cbow.wv[word]
])
# Result: 200-dimensional vector with complementary information

# Use combined vector for downstream task
from sklearn.svm import SVC
clf = SVC()
```

### Domain-Specific Training

```python
# Generic embeddings vs domain-specific

# Generic (large corpus):
generic_model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin'
)

# Domain-specific (medical texts):
medical_sentences = load_medical_corpus()
medical_model = Word2Vec(medical_sentences, vector_size=300, epochs=20)

# Medical model captures relationships like:
# "hypertension" ↔ "blood pressure"
# "MRI" ↔ "CT scan"
# Better for medical NLP tasks!

# In practice: Often combine both
from gensim.models import Word2Vec

class DomainAwareEmbedding:
    def __init__(self, generic_model, domain_model, alpha=0.5):
        self.generic = generic_model
        self.domain = domain_model
        self.alpha = alpha
    
    def get_vector(self, word):
        """Blend generic and domain embeddings"""
        if word in self.domain.wv:
            domain_vec = self.domain.wv[word]
        else:
            domain_vec = np.zeros(self.domain.vector_size)
        
        if word in self.generic.wv:
            generic_vec = self.generic.wv[word]
        else:
            generic_vec = np.zeros(self.generic.vector_size)
        
        # Weighted combination
        return self.alpha * domain_vec + (1 - self.alpha) * generic_vec

# Use for specific domain
blended = DomainAwareEmbedding(generic_model, medical_model, alpha=0.7)
vector = blended.get_vector('hypertension')
```

---

## 10. Transformers and Contextual Embeddings

### The Evolution to Transformers

**Why Transformers?**

Word2Vec and earlier methods are **static** - one embedding per word. But words have multiple meanings!

```
"I love the bank of the river"
"I work at a bank"

Both use "bank" but with different meanings
Word2Vec gives same embedding for both ❌
Transformers give different embeddings ✅
```

### What are Transformers?

**Core Idea:** Use **attention mechanism** to weigh which words are most relevant to each target word.

```python
# Simplified comparison

# Word2Vec:
# "bank" context = ["river", "the", "of"] (static window)
# Embedding = fixed mixture

# Transformer:
# "bank" context = Dynamically weighted:
#   - "river": 0.8 weight (highly relevant)
#   - "the": 0.1 weight (stopword)
#   - "of": 0.1 weight (modifier)
# Embedding = weighted average based on relevance
```

### Popular Transformer Models

```python
# BERT (Bidirectional Encoder Representations from Transformers)
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

text = "bank can mean financial institution or riverbank"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# Get embeddings
embeddings = outputs.last_hidden_state  # (1, seq_len, 768)
pooled = outputs.pooler_output  # (1, 768) - single representation

print(f"Token embeddings shape: {embeddings.shape}")
print(f"Pooled representation shape: {pooled.shape}")

# GPT-2 (Generative, similar idea)
# RoBERTa (Improved BERT)
# ALBERT (Lightweight BERT)
# T5 (Text-to-Text Transfer Transformer)
```

### Contextual vs Static Embeddings

```python
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Same word, different contexts
texts = [
    "I caught the fish by the river bank",
    "I withdraw money from the bank"
]

embeddings = []
for text in texts:
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Find "bank" token
    tokens = tokenizer.tokenize(text)
    bank_idx = tokens.index('bank')
    
    bank_embedding = outputs.last_hidden_state[0, bank_idx]
    embeddings.append(bank_embedding)

# Compare the "bank" embeddings
sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity of 'bank' in different contexts: {sim:.4f}")
# Will be different (not 1.0) because context changes embedding!
```

### When to Use What

```
Task: Select appropriate embedding method

Word2Vec/FastText:
✓ Simple document similarity
✓ Fast inference needed
✓ Limited computational resources
✓ Static relationships OK
✗ Need context-dependent meanings

Transformers:
✓ Production NLP systems
✓ Semantic understanding needed
✓ Handling polysemy important
✓ Have GPU/computational resources
✓ Fine-tuning on domain data
✗ Very slow
✗ Resource-intensive
```

---

## 11. Comparison: BoW vs TF-IDF vs Word2Vec

### Detailed Feature Comparison

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

documents = [
    "python programming language",
    "machine learning with python",
    "deep learning neural networks",
    "python data science"
]

# 1. Bag of Words
print("=" * 50)
print("BAG OF WORDS")
print("=" * 50)

bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(documents)

print(f"Vocabulary: {bow_vectorizer.get_feature_names_out()}")
print(f"Shape: {bow_matrix.shape}")
print(f"Matrix:\n{bow_matrix.toarray()}")
print(f"Sparsity: {1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1]):.1%}")

# 2. TF-IDF
print("\n" + "=" * 50)
print("TF-IDF")
print("=" * 50)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print(f"Shape: {tfidf_matrix.shape}")
print(f"Matrix:\n{tfidf_matrix.toarray()}")
print(f"Sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.1%}")

# 3. Word2Vec
print("\n" + "=" * 50)
print("WORD2VEC")
print("=" * 50)

# Tokenize for Word2Vec
tokenized = [doc.split() for doc in documents]
w2v_model = Word2Vec(tokenized, vector_size=10, window=5, min_count=1)

print(f"Vocabulary: {w2v_model.wv.index_to_key}")
print(f"Vector size: {w2v_model.wv.vector_size}")
print(f"Sample vector for 'python':\n{w2v_model.wv['python']}")

# Create document embeddings (average)
def doc_to_vec(doc, model):
    vectors = [model.wv[word] for word in doc.split() if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.wv.vector_size)

doc_vectors = np.array([doc_to_vec(doc, w2v_model) for doc in documents])
print(f"Document vector shape: {doc_vectors.shape}")
print(f"Sparsity: 0% (all dimensions have values)")

# Compare similarities
print("\n" + "=" * 50)
print("SIMILARITY COMPARISON")
print("=" * 50)

from sklearn.metrics.pairwise import cosine_similarity

# BoW similarity
bow_sim = cosine_similarity(bow_matrix)
print("BoW Similarity (Doc0 vs others):")
for i, sim in enumerate(bow_sim[0]):
    print(f"  Doc{i}: {sim:.3f}")

# TF-IDF similarity
tfidf_sim = cosine_similarity(tfidf_matrix)
print("\nTF-IDF Similarity (Doc0 vs others):")
for i, sim in enumerate(tfidf_sim[0]):
    print(f"  Doc{i}: {sim:.3f}")

# Word2Vec similarity
w2v_sim = cosine_similarity(doc_vectors)
print("\nWord2Vec Similarity (Doc0 vs others):")
for i, sim in enumerate(w2v_sim[0]):
    print(f"  Doc{i}: {sim:.3f}")
```

### Qualitative Comparison Table

| Aspect | BoW | TF-IDF | Word2Vec | Transformer |
|--------|-----|--------|----------|-------------|
| **Dimensionality** | Vocab size | Vocab size | Fixed (50-300) | Fixed (768-3072) |
| **Sparsity** | Sparse | Sparse | Dense | Dense |
| **Semantic** | ❌ No | ❌ No | ✅ Yes | ✅✅ Advanced |
| **Context** | ❌ No | ❌ No | Partial | ✅ Full |
| **Word Order** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Polysemy** | ❌ Single | ❌ Single | ❌ Single | ✅ Multiple |
| **Training Data** | None | None | Large corpus | Pre-trained |
| **Inference Speed** | Fast | Fast | Medium | Slow |
| **OOV Handling** | ❌ No | ❌ No | ⚠️ Limited | ✅ Subword |
| **Use Case** | Baseline | Similarity | Downstream | Production |

### When to Choose Each Method

**Use Bag of Words when:**
```python
# - Baseline/quick implementation needed
# - Resource-constrained
# - Interpretability critical (see exact word counts)
# - Dataset very small

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(documents)
```

**Use TF-IDF when:**
```python
# - Information retrieval task
# - Search engines
# - Document similarity
# - Quick bag-of-words improvement
# - Need interpretability but also weights

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000, max_df=0.8, min_df=2)
X = vectorizer.fit_transform(documents)
```

**Use Word2Vec when:**
```python
# - Semantic similarity important
# - Have large text corpus (>100K documents)
# - Downstream task (classification, clustering)
# - Speed vs accuracy trade-off acceptable
# - Static embeddings sufficient

from gensim.models import Word2Vec
model = Word2Vec(tokenized_sentences, vector_size=100, window=5)
```

**Use Transformers when:**
```python
# - State-of-the-art performance needed
# - Contextual understanding essential
# - Fine-tuning on specific domain
# - Have computational resources
# - Production system

from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained('bert-base-uncased')
```

---

## 12. Interview Questions

### Beginner Level

**Q1: What is the main difference between Word2Vec and Bag of Words?**

A: 
- **BoW:** Represents each document as a vector of word counts. Sparse, high-dimensional, no semantic information.
- **Word2Vec:** Learns dense, low-dimensional vectors for individual words that capture semantic meaning.

```python
# BoW: [2, 1, 0, 3, 0, ...]  (sparse, word counts)
# Word2Vec: [0.25, -0.48, 0.81, ...]  (dense, semantic)
```

**Q2: What does "you shall know a word by the company it keeps" mean?**

A: Words that appear in similar contexts should have similar meanings.

```
"The cat sat on the mat"
"The dog sat on the log"
↓
"cat" and "dog" appear in similar positions
↓
Word2Vec learns they should be similar
```

**Q3: What is vector_size in Word2Vec?**

A: The dimensionality of embedding vectors. Default 100, typical range 50-300.

```python
# Small (50): Fast, less information
# Medium (100-200): Balanced
# Large (300+): More semantic but slower
```

**Q4: What does sg=1 mean?**

A: Skip-Gram architecture (predict context from word). sg=0 means CBOW (predict word from context).

```python
sg=1  # "cat" → predict ["the", "sat", "on", "mat"]
sg=0  # ["the", "sat", "on", "mat"] → predict "cat"
```

**Q5: Why remove stopwords before Word2Vec training?**

A: Stopwords (the, and, a) are too common and don't carry meaning. They can:
- Dominate training
- Lower embedding quality
- Confuse similarity calculations

```python
# Instead of removing, can use sample parameter:
model = Word2Vec(sentences, sample=1e-3)  # Downsample frequent words
```

### Intermediate Level

**Q6: Explain the polysemy problem and its implications.**

A: Polysemy means one word has multiple meanings. Word2Vec produces one static embedding averaging all meanings.

```
"bank" meanings:
1. Financial institution
2. Riverbank
3. Side of a road

Word2Vec embedding ≈ average of all three
Neither meaning is well-represented!

Example consequences:
- "bank" most similar to both "money" and "river"
- Document classification confused
- Machine translation errors
```

**Solution:** Use contextual embeddings (BERT) that generate different vectors based on context.

**Q7: Compare Skip-Gram and CBOW.**

A:

| Aspect | Skip-Gram | CBOW |
|--------|-----------|------|
| Input | Word | Context |
| Output | Context | Word |
| Speed | Slower | Faster (3-10x) |
| Data Needs | Less | More |
| Quality | Better semantics | Better syntax |
| Rare Words | Better | Worse |
| Use | Semantic tasks | Downstream tasks |

```python
# Skip-Gram: Word prediction → semantics
model_sg = Word2Vec(sentences, sg=1)

# CBOW: Context prediction → syntax
model_cbow = Word2Vec(sentences, sg=0)

# Analogy benchmark: Skip-Gram usually wins
# Speed benchmark: CBOW usually wins
```

**Q8: What is the difference between negative sampling and hierarchical softmax?**

A: Both optimize training speed (softmax over 10K words is slow).

- **Negative Sampling:** Sample K negative examples + 1 positive, use binary classification
- **Hierarchical Softmax:** Use binary tree instead of flat softmax

```python
model = Word2Vec(
    sentences,
    negative=5,  # Negative sampling with 5 samples
    hs=0         # Don't use hierarchical softmax (use negative sampling)
)

model = Word2Vec(
    sentences,
    negative=0,  # No negative sampling
    hs=1         # Use hierarchical softmax
)
```

**Recommendation:** Negative sampling (default) is faster and better in most cases.

**Q9: How would you handle Out-of-Vocabulary (OOV) words?**

A: Several approaches:

```python
# Approach 1: Ignore them (lose information)
word_in_vocab = word in model.wv

# Approach 2: Use <UNK> token (generic representation)
vector = model.wv['<UNK>']

# Approach 3: Use FastText (approximate from char n-grams)
from gensim.models import FastText

model = FastText(sentences, vector_size=100)
vector = model.wv['unknownword']  # Works! Approximated

# Approach 4: Use subword tokenization (BPE, WordPiece)
# BERT does this automatically

# Approach 5: Average nearest neighbors
similar_word = find_similar_word_in_vocab(oov_word)
vector = model.wv[similar_word]
```

**FastText is best approach for Word2Vec:**
```python
from gensim.models import FastText

# FastText breaks words into character n-grams
# "learning" = ["<le", "lea", "ear", "arn", "rni", "nin", "ing", "ng>"]
# Unknown word can be approximated from its n-grams!

model = FastText(sentences, vector_size=100)
vector = model.wv['learning']  # Gets n-gram representation
```

**Q10: How would you evaluate Word2Vec embeddings?**

A: Several evaluation methods:

```python
# 1. Word Analogy Task
# Example: "king" - "man" + "woman" = ?
# Should answer "queen"

analogies = [
    ('king', 'man', 'queen', 'woman'),
    ('france', 'paris', 'germany', 'berlin'),
]

correct = 0
for a, b, c, d in analogies:
    try:
        prediction = model.wv.most_similar(
            positive=[c, b],
            negative=[a],
            topn=1
        )[0][0]
        if prediction == d:
            correct += 1
    except:
        pass

accuracy = correct / len(analogies)
print(f"Analogy accuracy: {accuracy:.1%}")

# 2. Word Similarity Correlation
# Compare model similarity to human judgment

from scipy.stats import spearmanr

word_pairs = [
    ('king', 'queen', 0.9),   # (word1, word2, human_similarity)
    ('cat', 'dog', 0.8),
    ('computer', 'keyboard', 0.7),
    ('car', 'bicycle', 0.6),
    ('car', 'cloud', 0.1),
]

model_sims = [model.wv.similarity(w1, w2) for w1, w2, _ in word_pairs]
human_sims = [sim for _, _, sim in word_pairs]

correlation, pvalue = spearmanr(model_sims, human_sims)
print(f"Correlation with human judgment: {correlation:.3f}")

# 3. Downstream Task Performance
# Classify documents using embeddings
from sklearn.svm import SVC

doc_vectors = [average_embeddings(doc, model) for doc in documents]
clf = SVC()
clf.fit(doc_vectors, labels)
accuracy = clf.score(doc_vectors_test, labels_test)
print(f"Classification accuracy: {accuracy:.1%}")

# 4. Visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = ['king', 'queen', 'man', 'woman', 'cat', 'dog']
vectors = np.array([model.wv[w] for w in words])
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
for word, (x, y) in zip(words, vectors_2d):
    plt.annotate(word, (x, y))
plt.show()
```

### Advanced Level

**Q11: What are the limitations of Word2Vec and how do contextual embeddings address them?**

A:

| Limitation | Word2Vec | Contextual (BERT) |
|-----------|----------|------------------|
| Polysemy | Single vector | Multiple (context-aware) |
| Word Order | Ignored | Captured via attention |
| OOV | Can't handle | Subword tokenization |
| Context | Local window | Full document |
| Static | Always same | Changes per context |

```python
# Demonstration
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text1 = "The bank of the river is beautiful"
text2 = "I went to the bank to withdraw money"

# Get embeddings for "bank" in both contexts
for text in [text1, text2]:
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        // filepath: /Users/mayankvashisht/Desktop/AI-ML/AI-ML/NLP/Word_Embeddings_Complete_Guide.md
# Word Embeddings: A Complete Guide from Basics to Advanced

A comprehensive guide covering word embeddings, Word2Vec, contextual embeddings, and practical implementations with detailed explanations, comparisons, and real-world examples.

---

## Table of Contents

1. [Introduction to Word Embeddings](#1-introduction-to-word-embeddings)
2. [Historical Context: From Bag of Words to Embeddings](#2-historical-context-from-bag-of-words-to-embeddings)
3. [Word2Vec: The Game Changer](#3-word2vec-the-game-changer)
4. [Word2Vec Architectures](#4-word2vec-architectures)
5. [Word2Vec Implementation Guide](#5-word2vec-implementation-guide)
6. [Understanding Word2Vec Arguments](#6-understanding-word2vec-arguments)
7. [Gensim Library](#7-gensim-library)
8. [Word2Vec Limitations and Problems](#8-word2vec-limitations-and-problems)
9. [Advanced Topics](#9-advanced-topics)
10. [Transformers and Contextual Embeddings](#10-transformers-and-contextual-embeddings)
11. [Comparison: BoW vs TF-IDF vs Word2Vec](#11-comparison-bow-vs-tfidf-vs-word2vec)
12. [Interview Questions](#12-interview-questions)
13. [Practical Applications](#13-practical-applications)
14. [Summary](#14-summary)

---

## 1. Introduction to Word Embeddings

### What are Word Embeddings?

**Definition:** Word embeddings are dense numerical vectors that represent words in a continuous vector space, where semantically similar words are close to each other.

**Key Characteristics:**
- **Dense:** Typically 50-300 dimensions (vs. sparse BoW with thousands)
- **Semantic:** Captures meaning and relationships between words
- **Learned:** Derived from large text corpora using neural networks
- **Continuous:** Values are real numbers, not just 0s and 1s

### Simple Example

```
One-Hot Encoding (Sparse):
word "king" = [0, 0, 1, 0, 0, ..., 0]  (10,000 dimensions, one 1)
word "queen" = [0, 0, 0, 1, 0, ..., 0]

Word Embeddings (Dense):
word "king" = [0.2, -0.5, 0.8, 0.3, ...]  (50 dimensions, many values)
word "queen" = [0.1, -0.4, 0.9, 0.25, ...]
word "prince" = [0.15, -0.45, 0.75, 0.28, ...]

Notice: "queen" and "prince" are closer to "king" than random words
```

### Why Word Embeddings Matter

```python
# Bag of Words: No semantic understanding
"king" and "queen" are completely unrelated vectors

# Word Embeddings: Capture semantics
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
# This is the famous analogy property!
```

---

## 2. Historical Context: From Bag of Words to Embeddings

### Evolution of Text Representation

#### Stage 1: One-Hot Encoding
```python
vocabulary = ["cat", "dog", "bird", "fish"]

"cat" = [1, 0, 0, 0]
"dog" = [0, 1, 0, 0]
"bird" = [0, 0, 1, 0]

# Problems:
# - Sparse (mostly zeros)
# - No semantic similarity
# - Vocab size = dimensions
# - Can't handle new words
```

#### Stage 2: Bag of Words (BoW)
```python
# Document: "the cat and the dog are playing"
vocabulary = ["the", "cat", "dog", "are", "and", "playing"]

BoW representation = [2, 1, 1, 1, 1, 1]
# Count how many times each word appears

# Improvement: Captures word frequency
# Problems:
# - Still sparse
# - No semantic understanding
# - Order doesn't matter ("cat ate dog" = "dog ate cat")
# - Heavy words dominate
```

#### Stage 3: TF-IDF (Term Frequency-Inverse Document Frequency)

**Formula:**
$$\text{TF-IDF}(word, doc) = TF(word, doc) \times IDF(word)$$

Where:
- $TF(word, doc) = \frac{\text{count of word in doc}}{\text{total words in doc}}$
- $IDF(word) = \log\left(\frac{\text{total documents}}{\text{documents containing word}}\right)$

**Example:**
```
Document 1: "the cat sat on the mat"
Document 2: "the dog sat on the log"
Document 3: "cats and dogs are pets"

Word "the":
- Appears in all 3 documents
- IDF("the") = log(3/3) = 0 (not important)
- TF-IDF is LOW

Word "cat":
- Appears in 1 document
- IDF("cat") = log(3/1) = 1.099 (important)
- TF-IDF is HIGH

# Improvement: Emphasizes important words
# Problems:
# - Still sparse
# - No semantic relationships
# - Fixed vocabulary
```

```python
# Code Example: TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print(f"Shape: {tfidf_matrix.shape}")  # (3, vocab_size)
print(f"Sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.2%}")
# Output: Very sparse (lots of zeros)

# Get feature names
features = vectorizer.get_feature_names_out()
print(f"Features: {features}")
```

#### Stage 4: Word Embeddings (Word2Vec, GloVe, FastText)

```python
# Dense representation - captures semantics!
"cat" = [0.25, -0.48, 0.81, -0.12, 0.56, ...]  (50 dims)
"dog" = [0.28, -0.45, 0.78, -0.10, 0.54, ...]  (50 dims)
"king" = [0.15, 0.75, -0.20, 0.62, -0.30, ...]
"queen" = [0.17, 0.73, -0.22, 0.64, -0.28, ...]
"man" = [0.12, 0.68, -0.18, 0.58, -0.25, ...]
"woman" = [0.19, 0.78, -0.24, 0.66, -0.32, ...]

# Notice the relationships:
# "king" - "man" + "woman" ≈ "queen"
# "cat" is close to "dog"
# "king" is far from "cat"
```

### Comparison Table

| Aspect | One-Hot | BoW | TF-IDF | Word Embeddings |
|--------|---------|-----|--------|-----------------|
| **Dimensions** | Vocab size | Vocab size | Vocab size | Fixed (50-300) |
| **Sparsity** | Very sparse | Sparse | Sparse | Dense |
| **Semantic** | ❌ | ❌ | ❌ | ✅ |
| **Context** | ❌ | Partial | ❌ | ✅ |
| **Training** | None | None | None | Requires corpus |
| **OOV handling** | ❌ | ❌ | ❌ | ⚠️ (depends on model) |
| **Use case** | Small vocab | Document class | Search | Modern NLP |

---

## 3. Word2Vec: The Game Changer

### Historical Background

**Question: How did Google improve search in its early days?**

Before Word2Vec, search engines relied on exact keyword matching:
```
User searches: "artificial intelligence"
Only found pages with exact phrase "artificial intelligence"
Missed pages with "machine learning" or "deep learning"
```

**The Problem:** They couldn't understand semantic similarity.

**The Solution:** Word embeddings! If "artificial intelligence" and "machine learning" have similar embeddings, they can be matched together.

### What is Word2Vec?

**Core Idea:** "You shall know a word by the company it keeps." - J.R. Firth

**Principle:** Words that appear in similar contexts have similar meanings.

```
Sentence: "The cat sat on the mat"
           The [cat] sat on the mat
           
Context window (size 2):
- Left context: ["The"]
- Right context: ["sat", "on"]

Sentence: "The dog sat on the rug"
           The [dog] sat on the rug

Context window:
- Left context: ["The"]
- Right context: ["sat", "on"]

Because "cat" and "dog" appear in similar contexts,
Word2Vec learns that they should have similar vectors!
```

### How Word2Vec Works: High-Level Overview

**The Learning Process:**

1. **Sliding Window:** Move a window through the text
2. **Extract Context:** For each word, note surrounding words
3. **Train Network:** Use a simple neural network to predict context from word
4. **Learn Embeddings:** The hidden layer weights become word vectors

```
Raw Text: "the quick brown fox jumps over the lazy dog"
Window size: 2

Step 1: [the, quick] brown [fox, jumps]
        → Input: "brown", Predict: ["the", "quick", "fox", "jumps"]

Step 2: [quick, brown] fox [jumps, over]
        → Input: "fox", Predict: ["quick", "brown", "jumps", "over"]

... continue for entire text

Through thousands of these examples, the network learns:
- "quick" and "fast" should be similar
- "brown" and "red" should be similar
- "dog" and "cat" should be similar
```

### Mathematical Framework

**Input:** Word index $w_i$

**Output:** Context words within window (typically 2-5 positions before/after)

**Neural Network:**
- Input layer: One-hot encoded word (vocabulary size)
- Hidden layer: Dense vector (embedding dimension) ← **This is our word vector!**
- Output layer: Softmax over vocabulary (predict context words)

```
Input word: "king"
    ↓
One-hot: [0, 0, ..., 1, ..., 0]  (10,000 dimensions)
    ↓
Hidden layer (weights are learned): [0.15, 0.75, -0.20, ...]  (50 dimensions)
    ↓
Output: Probability distribution over vocabulary
    → Should predict words in context: "the", "ruled", "throne", etc.
```

**Loss Function:** Negative log-likelihood of predicting correct context words

$$L = -\sum_{t=1}^{T} \sum_{-m \leq j \leq m, j \neq 0} \log P(w_{t+j} | w_t)$$

Where:
- $T$ = corpus size
- $m$ = window size
- $w_t$ = target word
- $w_{t+j}$ = context words

---

## 4. Word2Vec Architectures

### Architecture 1: Skip-Gram Model

**Idea:** Predict context words from a target word

```
Input: "cat"
↓
Embedding: [0.25, -0.48, 0.81, -0.12, ...]
↓
Output: Predict ["the", "sat", "on", "mat"]

This is like learning: Given "cat", what words are nearby?
```

**Advantages:**
- Works well with small datasets
- Better for rare words
- Captures more semantic relationships

**Disadvantages:**
- Slower training
- More parameters

**Use Case:** When you want rich semantic relationships

```python
# Skip-Gram in Gensim
model = Word2Vec(
    sentences=tokenized_text,
    sg=1,  # Skip-Gram
    vector_size=100,
    window=5
)
```

### Architecture 2: CBOW (Continuous Bag of Words)

**Idea:** Predict a target word from its context

```
Input: ["the", "sat", "on", "mat"]
↓
Average embeddings
↓
Output: Predict "cat"

This is like learning: Given surrounding words, what's the middle word?
```

**Advantages:**
- Faster training
- Better with small datasets
- Smoother representations

**Disadvantages:**
- Less semantic richness
- Struggles with rare words

**Use Case:** When you need speed or have limited data

```python
# CBOW in Gensim
model = Word2Vec(
    sentences=tokenized_text,
    sg=0,  # CBOW
    vector_size=100,
    window=5
)
```

### Visual Comparison

```
SKIP-GRAM:
word "cat"
    ↓
  [vector]
    ↓
Predict: "sat", "on", "the", "mat"
(1 input → many outputs)

CBOW:
"sat", "on", "the", "mat"
    ↓
Average vectors
    ↓
Predict: "cat"
(Many inputs → 1 output)
```

### Which to Choose?

```python
# Skip-Gram: Better semantics, slower
model_sg = Word2Vec(sentences, sg=1, vector_size=300)

# CBOW: Faster, better for small data
model_cbow = Word2Vec(sentences, sg=0, vector_size=300)

# Performance comparison:
# - Skip-Gram: Better for analogies ("king" - "man" + "woman" = "queen")
# - CBOW: Better for downstream tasks with limited training data
```

---

## 5. Word2Vec Implementation Guide

### Step 1: Prepare Your Data

```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required data
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text = """
Word embeddings are dense numerical vectors. 
They represent words in a continuous vector space. 
Semantically similar words are close to each other.
Word2Vec learns embeddings from text corpora.
It uses neural networks to capture word meaning.
"""

# Step 1: Tokenize into sentences
sentences = sent_tokenize(text)
print(f"Sentences: {sentences[:2]}")

# Step 2: Tokenize each sentence into words
tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]
print(f"Tokenized: {tokenized_sentences[0]}")

# Step 3: Remove stopwords and punctuation (optional)
stop_words = set(stopwords.words('english'))
filtered_sentences = [
    [word for word in sent if word.isalpha() and word not in stop_words]
    for sent in tokenized_sentences
]
print(f"Filtered: {filtered_sentences[0]}")
```

### Step 2: Train Word2Vec Model

```python
from gensim.models import Word2Vec

# Train Word2Vec
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,      # Embedding dimension
    window=5,             # Context window size
    min_count=2,          # Minimum word frequency
    workers=4,            # Number of threads
    sg=1,                 # 1=Skip-Gram, 0=CBOW
    epochs=10             # Training iterations
)

# Check model info
print(f"Vocabulary size: {len(model.wv)}")
print(f"Vector dimension: {model.wv.vector_size}")
```

### Step 3: Explore Learned Embeddings

```python
# Get word vector
word = "embeddings"
if word in model.wv:
    vector = model.wv[word]
    print(f"Vector for '{word}': {vector[:10]}...")  # First 10 dimensions
    print(f"Vector shape: {vector.shape}")

# Find similar words
print("\nMost similar to 'embeddings':")
similar_words = model.wv.most_similar('embeddings', topn=5)
for word, similarity in similar_words:
    print(f"  {word}: {similarity:.4f}")

# Word analogies
print("\nWord Analogies:")
# "king" is to "man" as "queen" is to ?
if all(w in model.wv for w in ['king', 'man', 'queen', 'woman']):
    result = model.wv.most_similar(
        positive=['queen', 'man'],
        negative=['woman'],
        topn=1
    )
    print(f"  queen:woman :: king:? → {result}")

# Calculate similarity
similarity = model.wv.similarity('embeddings', 'vectors')
print(f"\nSimilarity('embeddings', 'vectors'): {similarity:.4f}")

# Find most dissimilar (odd one out)
words = ['king', 'queen', 'man', 'table']
odd_one = model.wv.doesnt_match(words)
print(f"\nOdd one out in {words}: {odd_one}")
```

### Step 4: Use Embeddings in Your Pipeline

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Example: Document classification using Word2Vec

documents = [
    ("this movie is great", "positive"),
    ("i love this film", "positive"),
    ("excellent acting", "positive"),
    ("terrible movie", "negative"),
    ("i hate this", "negative"),
    ("worst film ever", "negative"),
]

# Convert documents to vectors (average word vectors)
X = []
y = []

for doc, label in documents:
    words = word_tokenize(doc.lower())
    
    # Get vectors for words in document
    vectors = [model.wv[w] for w in words if w in model.wv]
    
    if vectors:
        # Average the vectors
        doc_vector = np.mean(vectors, axis=0)
        X.append(doc_vector)
        y.append(label)

X = np.array(X)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Classification accuracy: {accuracy:.2%}")
```

### Step 5: Save and Load Models

```python
# Save model
model.save('word2vec_model.model')
model.wv.save('word2vec_vectors.wordvectors')

# Load model
from gensim.models import Word2Vec
loaded_model = Word2Vec.load('word2vec_model.model')

# Load vectors
from gensim.models import KeyedVectors
loaded_vectors = KeyedVectors.load('word2vec_vectors.wordvectors')
```

---

## 6. Understanding Word2Vec Arguments

### Complete Gensim Word2Vec Parameters

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences,              # List of tokenized sentences
    
    # Core parameters
    vector_size=100,        # Embedding dimension (50-300 typical)
    window=5,              # Context window size (words to look left/right)
    min_count=2,           # Minimum word frequency to include
    workers=4,             # Number of CPU threads
    sg=1,                  # 1=Skip-Gram, 0=CBOW
    epochs=5,              # Number of training iterations
    
    # Advanced parameters
    negative=5,            # Number of negative samples (for efficiency)
    alpha=0.025,          # Initial learning rate
    min_alpha=0.0001,     # Final learning rate
    seed=42,              # Random seed for reproducibility
    max_vocab_size=None,  # Maximum vocabulary size
    sample=1e-3,          # Downsampling threshold for frequent words
    hs=0,                 # 1=hierarchical softmax, 0=negative sampling
    sorted_vocab=1,       # Sort vocabulary by frequency
)
```

### Parameter Explanations

#### 1. **vector_size** (Embedding Dimension)

```python
# Small dimensions (50-100):
# - Pros: Fast training, less memory, good for small datasets
# - Cons: May miss semantic nuances
model_small = Word2Vec(sentences, vector_size=50)

# Medium dimensions (100-300):
# - Balanced for most tasks
model_medium = Word2Vec(sentences, vector_size=200)

# Large dimensions (300+):
# - Pros: Capture fine-grained semantics
# - Cons: Slower, needs more data, risk of overfitting
model_large = Word2Vec(sentences, vector_size=500)

# Practical recommendation: Start with 100, adjust based on:
# - Dataset size (larger dataset → larger vectors)
# - Task complexity
# - Available memory
```

#### 2. **window** (Context Window Size)

```python
# Sentence: "the quick brown fox jumps over the lazy dog"
#           
# window=1: Look 1 word left/right
#           "brown" context: ["quick", "fox"]

# window=2: Look 2 words left/right
#           "brown" context: ["the", "quick", "fox", "jumps"]

# window=5: Look 5 words left/right
#           "brown" context: ["the", "quick", "fox", "jumps", "over", "the"]

model_small_window = Word2Vec(sentences, window=2)
model_large_window = Word2Vec(sentences, window=10)

# Small window (2-3):
# - Captures syntactic relationships ("quick" → "brown")
# - "king" similar to adjectives

# Large window (5-10):
# - Captures topical relationships
# - "king" similar to "throne", "crown", "royal"

# Recommendation: window=5 is a good default
```

#### 3. **min_count** (Minimum Frequency)

```python
# Corpus has 1000 unique words, but some appear only once

# min_count=1: Include all words
# - Pros: Complete vocabulary
# - Cons: Rare words have poor embeddings, noise
model_all = Word2Vec(sentences, min_count=1)

# min_count=2: Include words appearing 2+ times
# - Standard choice, balances coverage and quality
model_standard = Word2Vec(sentences, min_count=2)

# min_count=5: Include only words appearing 5+ times
# - Pros: Quality embeddings, less noise
# - Cons: Misses rare but important words
model_filtered = Word2Vec(sentences, min_count=5)

# Recommendation: min_count=2 or 5 depending on corpus size
```

#### 4. **sg** (Skip-Gram vs CBOW)

```python
# Skip-Gram (sg=1)
# Input: word → Output: context words
model_sg = Word2Vec(
    sentences,
    sg=1,           # Skip-Gram
    vector_size=100
)

# CBOW (sg=0)
# Input: context words → Output: word
model_cbow = Word2Vec(
    sentences,
    sg=0,           # CBOW
    vector_size=100
)

# Benchmark results (from Word2Vec paper):
# Skip-Gram:
#   - Better for semantic tasks (analogies)
#   - Slower to train
#   - Better with limited data
#   - Accuracy: ~75% on Google analogies

# CBOW:
#   - Better for downstream tasks
#   - Faster to train (3-10x)
#   - Needs more data
#   - Smoother embeddings

# Choice rule:
# - Use Skip-Gram if you want semantic richness
# - Use CBOW if you prioritize speed
```

#### 5. **workers** (Parallelization)

```python
# Single-threaded (safe, deterministic)
model = Word2Vec(sentences, workers=1, seed=42)

# Multi-threaded (faster, but less reproducible)
model = Word2Vec(sentences, workers=4)  # 4 CPU cores
model = Word2Vec(sentences, workers=8)  # 8 CPU cores

# Recommendation: Use all available cores
# workers = number of CPU cores on your machine
import os
num_cores = os.cpu_count()
model = Word2Vec(sentences, workers=num_cores)
```

#### 6. **epochs** (Training Iterations)

```python
# Train for 1 epoch (go through data once)
model_1 = Word2Vec(sentences, epochs=1)  # Fast but poor quality

# Train for 5 epochs (standard)
model_5 = Word2Vec(sentences, epochs=5)  # Balanced

# Train for 10+ epochs (slow but better quality)
model_10 = Word2Vec(sentences, epochs=10)  # More time, better embeddings

# Diminishing returns:
# Epoch 1: Large improvement
# Epoch 2-5: Moderate improvement
# Epoch 5+: Minimal improvement

# Recommendation: Start with epochs=5, increase if convergence not reached
```

#### 7. **negative** (Negative Sampling)

```python
# Skip basic explanation; this is advanced
# Optimization trick to speed up training

# negative=5: Default, sample 5 negative examples per positive
model = Word2Vec(sentences, negative=5)

# negative=15: More negative samples, better quality but slower
model = Word2Vec(sentences, negative=15)

# Recommendation: Keep default (5) unless you have specific needs
```

#### 8. **sample** (Downsampling Frequent Words)

```python
# Problem: Common words like "the", "and" dominate
# Solution: Randomly discard them during training

# sample=1e-3 (default): Aggressive downsampling
# Probability of keeping word w:
# P(keep) = (sqrt(z/s) + 1) * (s/z)
# where z = frequency, s = sample parameter

model = Word2Vec(sentences, sample=1e-3)

# sample=0: No downsampling (keep all words)
model = Word2Vec(sentences, sample=0)

# Higher sample value = more downsampling
model = Word2Vec(sentences, sample=1e-2)  # Milder

# Recommendation: Keep default (1e-3)
```

### Complete Example with All Parameters

```python
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Prepare sentences
text = "Your large corpus here..."
sentences = sent_tokenize(text)
tokenized = [word_tokenize(s.lower()) for s in sentences]

# Train with explained parameters
model = Word2Vec(
    sentences=tokenized,
    
    # Architecture
    vector_size=100,      # 100 dimensions (good for most tasks)
    window=5,            # 5 words left/right (captures both syntax and topic)
    sg=1,                # Skip-Gram (better semantics)
    
    # Filtering
    min_count=2,         # Words appearing 2+ times
    sample=1e-3,         # Downsample frequent words
    
    # Training
    epochs=5,            # 5 passes through data
    workers=4,           # 4 CPU threads
    negative=5,          # 5 negative samples
    
    # Reproducibility
    seed=42
)

# Verify training
print(f"Vocabulary size: {len(model.wv)}")
print(f"Training time: {model.epochs} epochs")
print(f"Sample vector shape: {model.wv[list(model.wv.index_to_key)[0]].shape}")
```

---

## 7. Gensim Library

### What is Gensim?

Gensim is a Python library specialized in NLP tasks, particularly:
- Training Word2Vec models efficiently
- Loading pre-trained embeddings
- Topic modeling (LDA)
- Document similarity

**Installation:**
```bash
pip install gensim
```

### Key Gensim Classes

#### 1. Word2Vec

```python
from gensim.models import Word2Vec

# Train from scratch
model = Word2Vec(sentences, vector_size=100, window=5)

# Access word vector
vector = model.wv['word']

# Find similar words
model.wv.most_similar('word', topn=5)

# Word analogies
model.wv.most_similar(positive=['king', 'woman'], negative=['man'])

# Calculate similarity
similarity = model.wv.similarity('word1', 'word2')
```

#### 2. KeyedVectors

```python
from gensim.models import KeyedVectors

# Pre-trained embeddings (e.g., from Google)
model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin',
    binary=True
)

# Access methods
similar = model.most_similar('computer')
vector = model['computer']
```

#### 3. FastText (Handles OOV Words)

```python
from gensim.models import FastText

# FastText extends Word2Vec to handle out-of-vocabulary words
model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1
)

# Works even for unseen words!
# Breaks word into character n-grams
vector = model.wv['unknownword']  # Works! (approximated from char-grams)
```

### Complete Gensim Workflow

```python
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

# 1. Prepare and train
sentences = [['hello', 'world'], ['good', 'morning']]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# 2. Explore
print("Similar to 'hello':", model.wv.most_similar('hello', topn=3))

# 3. Save
model.save('my_word2vec.model')
model.wv.save('my_word2vec.vectors')

# 4. Load
loaded_model = Word2Vec.load('my_word2vec.model')
loaded_vectors = KeyedVectors.load('my_word2vec.vectors')

# 5. Use in pipeline
from sklearn.svm import SVC
import numpy as np

# Convert documents to vectors
def doc_to_vector(doc, model, vector_size=100):
    vectors = [model.wv[word] for word in doc.split() if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(vector_size)

doc_vectors = [doc_to_vector(doc, model) for doc, _ in data]
clf = SVC()
clf.fit(doc_vectors, labels)
```

---

## 8. Word2Vec Limitations and Problems

### Problem 1: The Polysemy Problem

**Definition:** One word has multiple meanings depending on context.

**Example:**
```
"bank" has at least 2 meanings:
1. Financial institution: "I went to the bank to withdraw money"
2. Riverbank: "We sat by the bank of the river"

Word2Vec produces ONE vector for "bank"
This vector is an AVERAGE of both meanings
Not ideal for either context!
```

**Why It's a Problem:**

```python
# When we train Word2Vec:
# Context 1: "withdrew", "money", "account" → vector1
# Context 2: "river", "water", "trees" → vector2
# Final vector = mixture of both

# When predicting similar words:
model.wv.most_similar('bank')
# Result might contain both financial and geography words
# Not ideal!
```

**Solution:** Use contextual embeddings (BERT, GPT)

```python
# With BERT:
# Same word "bank" gets different embeddings based on context
# Solved the polysemy problem!
```

### Problem 2: Word Order is Ignored

**Definition:** Word2Vec only considers co-occurrence, not word order within context window.

**Example:**
```
Sentence 1: "The cat ate the mouse"
            window: ate → [cat, the, mouse, the]

Sentence 2: "The mouse ate the cat"
            window: ate → [mouse, the, cat, the]

Word2Vec sees:
- "cat" appears with "ate", "mouse" ✓
- "mouse" appears with "ate", "cat" ✓

But misses:
- "cat ate" vs "ate cat" (different roles!)
- Word order matters for meaning
```

**Why It's a Problem:**

```
"I didn't like the movie" 
vs
"I like the movie"

Word2Vec might see these as similar because they share many words
But the meanings are opposite!
```

**Limitation:** This is inherent to context window approach

```
Sentence: "The quick brown fox jumped"
Word "fox" context [window=2]:
- Left: ["brown", "quick"]  (what was before?)
- Right: ["jumped"]         (what comes after?)
- No distinction which came before/after

Modern fix: Transformers use attention to track order
```

### Problem 3: Inability to Handle New Words (OOV - Out of Vocabulary)

**Definition:** Word2Vec has fixed vocabulary; can't represent unseen words.

**Example:**
```
Training data words: {"cat", "dog", "bird", "fish"}

New text mentions "dinosaur" (not in training)
↓
Can't look up embedding for "dinosaur"
↓
Two options:
1. Ignore it (lose information)
2. Use <UNK> token (generic representation)
3. Skip it entirely

Not ideal!
```

**Why It's a Problem:**

```
Real world:
- New products, brands, slang emerge constantly
- Scientific terms, proper nouns, typos
- Word2Vec can't handle any of these

Example:
Training: "iPhone X is great"
Test: "iPhone 15 is great"
↓
"iPhone 15" not in vocab → can't represent properly
```

**Solution:** Use FastText or Byte-Pair Encoding

```python
from gensim.models import FastText

# FastText: Break words into character n-grams
# "unknown" = ["<un", "unk", "nkn", "kno", "now", "own", "wn>"]

# Even unseen words can be represented!
model = FastText(sentences, vector_size=100)

# "iPhone15" can be approximated from character n-grams
# even if never seen in training
vector = model.wv['iPhone15']  # Works!

# Trade-off: Less semantically pure but more robust
```

### Problem 4: Static Embeddings Don't Capture Context

**Definition:** Each word gets ONE embedding regardless of context.

**Example:**
```
"I caught the bank of the river with my fishing rod"
"The bank of England announced new policies"
"river bank" (multiple meanings)
"bank robber"

All occurrences of "bank" get the SAME vector
But meanings are different!
```

**Why It's a Problem:**

```
Downstream task: Sentiment analysis
Sentence: "That's sick!" (positive: amazing)
↓
Word2Vec embedding for "sick" was trained on:
- "That sick person needs help" (negative)
- "Sick skills!" (positive)
↓
Static embedding confuses the model
Doesn't know which meaning applies here
```

**Solution:** Contextual embeddings (BERT, GPT, ELMo)

```python
# These models generate embeddings CONDITIONED on context
# Same word gets different representations in different contexts

from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text1 = "I caught the bank of the river"
text2 = "I went to the bank to withdraw money"

# Get BERT embeddings (contextual)
with torch.no_grad():
    inputs1 = tokenizer(text1, return_tensors='pt')
    outputs1 = model(**inputs1)
    embedding1 = outputs1.last_hidden_state
    
    inputs2 = tokenizer(text2, return_tensors='pt')
    outputs2 = model(**inputs2)
    embedding2 = outputs2.last_hidden_state

# Same word "bank" gets DIFFERENT embeddings
# in different contexts!
```

### Problem 5: Dominance of Stopwords

**Definition:** Common words overwhelm important words if not filtered.

**Example:**
```
Text: "The quick brown fox jumps over the lazy dog"

Frequency:
- "the": 2 times
- "quick", "brown", "fox": 1 time each
- "jumps", "over", "lazy", "dog": 1 time each

Word2Vec trains on "the" twice as much
↓
"the" gets a high-quality embedding
↓
"the" is closest to other common words
↓
Not useful for downstream tasks!
```

**Why It's a Problem:**

```
Document similarity:
Doc 1: "The cat is sleeping"
Doc 2: "The dog is playing"

Similarity is HIGH because both contain "the", "is"
But semantically different!

Average embedding heavily influenced by stopwords
```

**Solution:** Remove or downsample

```python
from nltk.corpus import stopwords

# Option 1: Remove stopwords before training
stop_words = set(stopwords.words('english'))
filtered = [[w for w in sent if w not in stop_words] for sent in sentences]
model = Word2Vec(filtered, vector_size=100)

# Option 2: Downsample in Word2Vec
model = Word2Vec(sentences, sample=1e-3)  # Downsample frequent words

# Option 3: Use TF-IDF weighting when creating document vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
# TF-IDF automatically downweights common words
```

### Summary: Word2Vec Problems

| Problem | Impact | Solution |
|---------|--------|----------|
| **Polysemy** | Same word, different meanings confuse model | Use contextual embeddings (BERT) |
| **Word Order** | "cat ate dog" same as "dog ate cat" | Use Transformers with attention |
| **OOV Words** | Can't represent unseen words | Use FastText or subword tokenization |
| **Static** | No context-dependent meanings | Use contextual embeddings |
| **Stopwords** | Common words dominate | Remove or downsample |

---

## 9. Advanced Topics

### PCA for Dimensionality Reduction

**Question: Where is PCA used in Word2Vec?**

**Use Case:** Visualizing high-dimensional embeddings in 2D/3D.

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Word2Vec gives 100-300 dimensional vectors
# Can't visualize directly!

# Get vectors for analysis
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
         'cat', 'dog', 'kitten', 'puppy', 'tiger', 'lion']
vectors = np.array([model.wv[word] for word in words])

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Visualize
plt.figure(figsize=(10, 8))
for word, (x, y) in zip(words, vectors_2d):
    plt.scatter(x, y, s=100)
    plt.annotate(word, (x, y), fontsize=10)

plt.title('Word2Vec Embeddings (PCA Projection)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.grid(True)
plt.show()

# Interpretation:
# Points close together = similar meanings
# PCA chooses dimensions that capture most variance
```

**When to Use PCA:**
- Visualizing embeddings
- Reducing to 2D/3D for plots
- Removing noise (keeping top k dimensions)

```python
# Remove noise
pca = PCA(n_components=50)  # Reduce from 100 to 50
reduced_vectors = pca.fit_transform(vectors)

# Variance captured
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")
# Usually 50 dimensions capture 90%+ variance
```

### Combining Multiple Embeddings

```python
# Train multiple Word2Vec models
model_sg = Word2Vec(sentences, sg=1, vector_size=100)
model_cbow = Word2Vec(sentences, sg=0, vector_size=100)

# Combine for richer representation
word = "learning"
combined_vector = np.concatenate([
    model_sg.wv[word],
    model_cbow.wv[word]
])
# Result: 200-dimensional vector with complementary information

# Use combined vector for downstream task
from sklearn.svm import SVC
clf = SVC()
```

### Domain-Specific Training

```python
# Generic embeddings vs domain-specific

# Generic (large corpus):
generic_model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin'
)

# Domain-specific (medical texts):
medical_sentences = load_medical_corpus()
medical_model = Word2Vec(medical_sentences, vector_size=300, epochs=20)

# Medical model captures relationships like:
# "hypertension" ↔ "blood pressure"
# "MRI" ↔ "CT scan"
# Better for medical NLP tasks!

# In practice: Often combine both
from gensim.models import Word2Vec

class DomainAwareEmbedding:
    def __init__(self, generic_model, domain_model, alpha=0.5):
        self.generic = generic_model
        self.domain = domain_model
        self.alpha = alpha
    
    def get_vector(self, word):
        """Blend generic and domain embeddings"""
        if word in self.domain.wv:
            domain_vec = self.domain.wv[word]
        else:
            domain_vec = np.zeros(self.domain.vector_size)
        
        if word in self.generic.wv:
            generic_vec = self.generic.wv[word]
        else:
            generic_vec = np.zeros(self.generic.vector_size)
        
        # Weighted combination
        return self.alpha * domain_vec + (1 - self.alpha) * generic_vec

# Use for specific domain
blended = DomainAwareEmbedding(generic_model, medical_model, alpha=0.7)
vector = blended.get_vector('hypertension')
```

---

## 10. Transformers and Contextual Embeddings

### The Evolution to Transformers

**Why Transformers?**

Word2Vec and earlier methods are **static** - one embedding per word. But words have multiple meanings!

```
"I love the bank of the river"
"I work at a bank"

Both use "bank" but with different meanings
Word2Vec gives same embedding for both ❌
Transformers give different embeddings ✅
```

### What are Transformers?

**Core Idea:** Use **attention mechanism** to weigh which words are most relevant to each target word.

```python
# Simplified comparison

# Word2Vec:
# "bank" context = ["river", "the", "of"] (static window)
# Embedding = fixed mixture

# Transformer:
# "bank" context = Dynamically weighted:
#   - "river": 0.8 weight (highly relevant)
#   - "the": 0.1 weight (stopword)
#   - "of": 0.1 weight (modifier)
# Embedding = weighted average based on relevance
```

### Popular Transformer Models

```python
# BERT (Bidirectional Encoder Representations from Transformers)
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

text = "bank can mean financial institution or riverbank"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# Get embeddings
embeddings = outputs.last_hidden_state  # (1, seq_len, 768)
pooled = outputs.pooler_output  # (1, 768) - single representation

print(f"Token embeddings shape: {embeddings.shape}")
print(f"Pooled representation shape: {pooled.shape}")

# GPT-2 (Generative, similar idea)
# RoBERTa (Improved BERT)
# ALBERT (Lightweight BERT)
# T5 (Text-to-Text Transfer Transformer)
```

### Contextual vs Static Embeddings

```python
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Same word, different contexts
texts = [
    "I caught the fish by the river bank",
    "I withdraw money from the bank"
]

embeddings = []
for text in texts:
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Find "bank" token
    tokens = tokenizer.tokenize(text)
    bank_idx = tokens.index('bank')
    
    bank_embedding = outputs.last_hidden_state[0, bank_idx]
    embeddings.append(bank_embedding)

# Compare the "bank" embeddings
sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity of 'bank' in different contexts: {sim:.4f}")
# Will be different (not 1.0) because context changes embedding!
```

### When to Use What

```
Task: Select appropriate embedding method

Word2Vec/FastText:
✓ Simple document similarity
✓ Fast inference needed
✓ Limited computational resources
✓ Static relationships OK
✗ Need context-dependent meanings

Transformers:
✓ Production NLP systems
✓ Semantic understanding needed
✓ Handling polysemy important
✓ Have GPU/computational resources
✓ Fine-tuning on domain data
✗ Very slow
✗ Resource-intensive
```

---

## 11. Comparison: BoW vs TF-IDF vs Word2Vec

### Detailed Feature Comparison

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

documents = [
    "python programming language",
    "machine learning with python",
    "deep learning neural networks",
    "python data science"
]

# 1. Bag of Words
print("=" * 50)
print("BAG OF WORDS")
print("=" * 50)

bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(documents)

print(f"Vocabulary: {bow_vectorizer.get_feature_names_out()}")
print(f"Shape: {bow_matrix.shape}")
print(f"Matrix:\n{bow_matrix.toarray()}")
print(f"Sparsity: {1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1]):.1%}")

# 2. TF-IDF
print("\n" + "=" * 50)
print("TF-IDF")
print("=" * 50)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print(f"Shape: {tfidf_matrix.shape}")
print(f"Matrix:\n{tfidf_matrix.toarray()}")
print(f"Sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.1%}")

# 3. Word2Vec
print("\n" + "=" * 50)
print("WORD2VEC")
print("=" * 50)

# Tokenize for Word2Vec
tokenized = [doc.split() for doc in documents]
w2v_model = Word2Vec(tokenized, vector_size=10, window=5, min_count=1)

print(f"Vocabulary: {w2v_model.wv.index_to_key}")
print(f"Vector size: {w2v_model.wv.vector_size}")
print(f"Sample vector for 'python':\n{w2v_model.wv['python']}")

# Create document embeddings (average)
def doc_to_vec(doc, model):
    vectors = [model.wv[word] for word in doc.split() if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.wv.vector_size)

doc_vectors = np.array([doc_to_vec(doc, w2v_model) for doc in documents])
print(f"Document vector shape: {doc_vectors.shape}")
print(f"Sparsity: 0% (all dimensions have values)")

# Compare similarities
print("\n" + "=" * 50)
print("SIMILARITY COMPARISON")
print("=" * 50)

from sklearn.metrics.pairwise import cosine_similarity

# BoW similarity
bow_sim = cosine_similarity(bow_matrix)
print("BoW Similarity (Doc0 vs others):")
for i, sim in enumerate(bow_sim[0]):
    print(f"  Doc{i}: {sim:.3f}")

# TF-IDF similarity
tfidf_sim = cosine_similarity(tfidf_matrix)
print("\nTF-IDF Similarity (Doc0 vs others):")
for i, sim in enumerate(tfidf_sim[0]):
    print(f"  Doc{i}: {sim:.3f}")

# Word2Vec similarity
w2v_sim = cosine_similarity(doc_vectors)
print("\nWord2Vec Similarity (Doc0 vs others):")
for i, sim in enumerate(w2v_sim[0]):
    print(f"  Doc{i}: {sim:.3f}")
```

### Qualitative Comparison Table

| Aspect | BoW | TF-IDF | Word2Vec | Transformer |
|--------|-----|--------|----------|-------------|
| **Dimensionality** | Vocab size | Vocab size | Fixed (50-300) | Fixed (768-3072) |
| **Sparsity** | Sparse | Sparse | Dense | Dense |
| **Semantic** | ❌ No | ❌ No | ✅ Yes | ✅✅ Advanced |
| **Context** | ❌ No | ❌ No | Partial | ✅ Full |
| **Word Order** | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Polysemy** | ❌ Single | ❌ Single | ❌ Single | ✅ Multiple |
| **Training Data** | None | None | Large corpus | Pre-trained |
| **Inference Speed** | Fast | Fast | Medium | Slow |
| **OOV Handling** | ❌ No | ❌ No | ⚠️ Limited | ✅ Subword |
| **Use Case** | Baseline | Similarity | Downstream | Production |

### When to Choose Each Method

**Use Bag of Words when:**
```python
# - Baseline/quick implementation needed
# - Resource-constrained
# - Interpretability critical (see exact word counts)
# - Dataset very small

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(documents)
```

**Use TF-IDF when:**
```python
# - Information retrieval task
# - Search engines
# - Document similarity
# - Quick bag-of-words improvement
# - Need interpretability but also weights

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000, max_df=0.8, min_df=2)
X = vectorizer.fit_transform(documents)
```

**Use Word2Vec when:**
```python
# - Semantic similarity important
# - Have large text corpus (>100K documents)
# - Downstream task (classification, clustering)
# - Speed vs accuracy trade-off acceptable
# - Static embeddings sufficient

from gensim.models import Word2Vec
model = Word2Vec(tokenized_sentences, vector_size=100, window=5)
```

**Use Transformers when:**
```python
# - State-of-the-art performance needed
# - Contextual understanding essential
# - Fine-tuning on specific domain
# - Have computational resources
# - Production system

from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained('bert-base-uncased')
```

---

## 12. Interview Questions

### Beginner Level

**Q1: What is the main difference between Word2Vec and Bag of Words?**

A: 
- **BoW:** Represents each document as a vector of word counts. Sparse, high-dimensional, no semantic information.
- **Word2Vec:** Learns dense, low-dimensional vectors for individual words that capture semantic meaning.

```python
# BoW: [2, 1, 0, 3, 0, ...]  (sparse, word counts)
# Word2Vec: [0.25, -0.48, 0.81, ...]  (dense, semantic)
```

**Q2: What does "you shall know a word by the company it keeps" mean?**

A: Words that appear in similar contexts should have similar meanings.

```
"The cat sat on the mat"
"The dog sat on the log"
↓
"cat" and "dog" appear in similar positions
↓
Word2Vec learns they should be similar
```

**Q3: What is vector_size in Word2Vec?**

A: The dimensionality of embedding vectors. Default 100, typical range 50-300.

```python
# Small (50): Fast, less information
# Medium (100-200): Balanced
# Large (300+): More semantic but slower
```

**Q4: What does sg=1 mean?**

A: Skip-Gram architecture (predict context from word). sg=0 means CBOW (predict word from context).

```python
sg=1  # "cat" → predict ["the", "sat", "on", "mat"]
sg=0  # ["the", "sat", "on", "mat"] → predict "cat"
```

**Q5: Why remove stopwords before Word2Vec training?**

A: Stopwords (the, and, a) are too common and don't carry meaning. They can:
- Dominate training
- Lower embedding quality
- Confuse similarity calculations

```python
# Instead of removing, can use sample parameter:
model = Word2Vec(sentences, sample=1e-3)  # Downsample frequent words
```

### Intermediate Level

**Q6: Explain the polysemy problem and its implications.**

A: Polysemy means one word has multiple meanings. Word2Vec produces one static embedding averaging all meanings.

```
"bank" meanings:
1. Financial institution
2. Riverbank
3. Side of a road

Word2Vec embedding ≈ average of all three
Neither meaning is well-represented!

Example consequences:
- "bank" most similar to both "money" and "river"
- Document classification confused
- Machine translation errors
```

**Solution:** Use contextual embeddings (BERT) that generate different vectors based on context.

**Q7: Compare Skip-Gram and CBOW.**

A:

| Aspect | Skip-Gram | CBOW |
|--------|-----------|------|
| Input | Word | Context |
| Output | Context | Word |
| Speed | Slower | Faster (3-10x) |
| Data Needs | Less | More |
| Quality | Better semantics | Better syntax |
| Rare Words | Better | Worse |
| Use | Semantic tasks | Downstream tasks |

```python
# Skip-Gram: Word prediction → semantics
model_sg = Word2Vec(sentences, sg=1)

# CBOW: Context prediction → syntax
model_cbow = Word2Vec(sentences, sg=0)

# Analogy benchmark: Skip-Gram usually wins
# Speed benchmark: CBOW usually wins
```

**Q8: What is the difference between negative sampling and hierarchical softmax?**

A: Both optimize training speed (softmax over 10K words is slow).

- **Negative Sampling:** Sample K negative examples + 1 positive, use binary classification
- **Hierarchical Softmax:** Use binary tree instead of flat softmax

```python
model = Word2Vec(
    sentences,
    negative=5,  # Negative sampling with 5 samples
    hs=0         # Don't use hierarchical softmax (use negative sampling)
)

model = Word2Vec(
    sentences,
    negative=0,  # No negative sampling
    hs=1         # Use hierarchical softmax
)
```

**Recommendation:** Negative sampling (default) is faster and better in most cases.

**Q9: How would you handle Out-of-Vocabulary (OOV) words?**

A: Several approaches:

```python
# Approach 1: Ignore them (lose information)
word_in_vocab = word in model.wv

# Approach 2: Use <UNK> token (generic representation)
vector = model.wv['<UNK>']

# Approach 3: Use FastText (approximate from char n-grams)
from gensim.models import FastText

model = FastText(sentences, vector_size=100)
vector = model.wv['unknownword']  # Works! Approximated

# Approach 4: Use subword tokenization (BPE, WordPiece)
# BERT does this automatically

# Approach 5: Average nearest neighbors
similar_word = find_similar_word_in_vocab(oov_word)
vector = model.wv[similar_word]
```

**FastText is best approach for Word2Vec:**
```python
from gensim.models import FastText

# FastText breaks words into character n-grams
# "learning" = ["<le", "lea", "ear", "arn", "rni", "nin", "ing", "ng>"]
# Unknown word can be approximated from its n-grams!

model = FastText(sentences, vector_size=100)
vector = model.wv['learning']  # Gets n-gram representation
```

**Q10: How would you evaluate Word2Vec embeddings?**

A: Several evaluation methods:

```python
# 1. Word Analogy Task
# Example: "king" - "man" + "woman" = ?
# Should answer "queen"

analogies = [
    ('king', 'man', 'queen', 'woman'),
    ('france', 'paris', 'germany', 'berlin'),
]

correct = 0
for a, b, c, d in analogies:
    try:
        prediction = model.wv.most_similar(
            positive=[c, b],
            negative=[a],
            topn=1
        )[0][0]
        if prediction == d:
            correct += 1
    except:
        pass

accuracy = correct / len(analogies)
print(f"Analogy accuracy: {accuracy:.1%}")

# 2. Word Similarity Correlation
# Compare model similarity to human judgment

from scipy.stats import spearmanr

word_pairs = [
    ('king', 'queen', 0.9),   # (word1, word2, human_similarity)
    ('cat', 'dog', 0.8),
    ('computer', 'keyboard', 0.7),
    ('car', 'bicycle', 0.6),
    ('car', 'cloud', 0.1),
]

model_sims = [model.wv.similarity(w1, w2) for w1, w2, _ in word_pairs]
human_sims = [sim for _, _, sim in word_pairs]

correlation, pvalue = spearmanr(model_sims, human_sims)
print(f"Correlation with human judgment: {correlation:.3f}")

# 3. Downstream Task Performance
# Classify documents using embeddings
from sklearn.svm import SVC

doc_vectors = [average_embeddings(doc, model) for doc in documents]
clf = SVC()
clf.fit(doc_vectors, labels)
accuracy = clf.score(doc_vectors_test, labels_test)
print(f"Classification accuracy: {accuracy:.1%}")

# 4. Visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = ['king', 'queen', 'man', 'woman', 'cat', 'dog']
vectors = np.array([model.wv[w] for w in words])
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
for word, (x, y) in zip(words, vectors_2d):
    plt.annotate(word, (x, y))
plt.show()
```

### Advanced Level

**Q11: What are the limitations of Word2Vec and how do contextual embeddings address them?**

A:

| Limitation | Word2Vec | Contextual (BERT) |
|-----------|----------|------------------|
| Polysemy | Single vector | Multiple (context-aware) |
| Word Order | Ignored | Captured via attention |
| OOV | Can't handle | Subword tokenization |
| Context | Local window | Full document |
| Static | Always same | Changes per context |

**Q11 (continued): Complete Demonstration of Contextual Embeddings vs Static**

```python
# Full demonstration comparing Word2Vec and BERT
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from gensim.models import Word2Vec

# ============================================
# SETUP: Word2Vec Model
# ============================================

sentences = [
    ["I", "caught", "the", "fish", "by", "the", "river", "bank"],
    ["I", "withdraw", "money", "from", "the", "bank"],
    ["The", "river", "has", "a", "steep", "bank"],
]

# Train Word2Vec
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)

# Get "bank" vector (static - always the same)
bank_w2v = w2v_model.wv['bank']
print("=" * 60)
print("WORD2VEC: Static Embeddings")
print("=" * 60)
print(f"Vector shape: {bank_w2v.shape}")
print(f"First 10 dimensions: {bank_w2v[:10]}")

# ============================================
# SETUP: BERT Model (Contextual)
# ============================================

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

texts = [
    "I caught the fish by the river bank",
    "I withdraw money from the bank",
]

embeddings_bert = []

print("\n" + "=" * 60)
print("BERT: Contextual Embeddings")
print("=" * 60)

for text in texts:
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get all token embeddings
    token_embeddings = outputs.last_hidden_state[0]
    
    # Find "bank" token
    tokens = tokenizer.tokenize(text)
    if 'bank' in tokens:
        bank_idx = tokens.index('bank')
        bank_embedding = token_embeddings[bank_idx]
        embeddings_bert.append(bank_embedding.numpy())
        
        print(f"\nText: {text}")
        print(f"Token index of 'bank': {bank_idx}")
        print(f"Vector shape: {bank_embedding.shape}")
        print(f"First 10 dimensions: {bank_embedding[:10].numpy()}")

# ============================================
# COMPARISON
# ============================================

print("\n" + "=" * 60)
print("COMPARISON: Word2Vec vs BERT")
print("=" * 60)

# Word2Vec: Same vector always
print(f"\nWord2Vec 'bank' vector (always the same):")
print(f"  Dimensions: {bank_w2v.shape}")
print(f"  Type: Static (one vector for all contexts)")

# BERT: Different vectors per context
print(f"\nBERT 'bank' embeddings (context-dependent):")
print(f"  Context 1 (river bank): {embeddings_bert[0].shape}")
print(f"  Context 2 (financial bank): {embeddings_bert[1].shape}")
print(f"  Type: Contextual (different vector per context)")

# Calculate similarity between the two BERT embeddings
if len(embeddings_bert) == 2:
    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    sim = cosine_similarity(
        embeddings_bert[0].reshape(1, -1),
        embeddings_bert[1].reshape(1, -1)
    )[0, 0]
    
    print(f"\n✓ Cosine similarity between two 'bank' contexts (BERT): {sim:.4f}")
    print(f"  → < 1.0 means different meanings recognized!")
    print(f"  → This is what Word2Vec CANNOT do")
```

**Q12: Explain the Attention Mechanism in Transformers**

A: **Attention** is the core innovation that makes transformers work.

```python
# Simplified attention visualization

"""
Basic Idea: Instead of fixed context window, 
use LEARNED weights to focus on relevant words

Word2Vec (fixed window):
"bank" looks at: ["river", "the", "bank", "of", "the"]
All get equal importance in processing

Transformer (attention):
"bank" LEARNS which words matter:
- "river": 0.8 (very relevant - semantic context)
- "the": 0.05 (common word - less relevant)  
- "of": 0.1 (structural - somewhat relevant)
- "money": 0.05 (from another context)

So "bank" embedding is weighted average:
bank_embedding = 0.8 * river_emb + 0.05 * the_emb + ...
"""

# Visual representation
import numpy as np
import matplotlib.pyplot as plt

# Simulated attention weights for "bank"
words = ["I", "caught", "the", "fish", "by", "the", "river", "bank"]
attention_weights = [0.02, 0.05, 0.03, 0.08, 0.05, 0.03, 0.7, 0.2]

plt.figure(figsize=(10, 6))
colors = ['red' if w == 'bank' else 'blue' for w in words]
plt.bar(words, attention_weights, color=colors)
plt.ylabel('Attention Weight')
plt.title('Attention Weights for "bank" in Context')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Interpretation:")
print("- Red bar (bank itself): 0.2 weight")
print("- Blue bars (other words):")
print("  - 'river': 0.7 weight (MOST important!)")
print("  - Others: low weights")
print("\nThis is learned during training - no manual engineering!")
```

**Q13: BERT vs GPT - What's the difference?**

A: Both are transformer-based but designed differently:

```python
# BERT: Bidirectional Encoder Representations from Transformers
# - Reads text LEFT-TO-RIGHT AND RIGHT-TO-LEFT simultaneously
# - Masked Language Model training (predict missing words)
# - Best for understanding/classification tasks
# - ENCODER ONLY

from transformers import AutoTokenizer, AutoModel

bert_model = AutoModel.from_pretrained("bert-base-uncased")

text = "The [MASK] sat on the mat"
# BERT can look both ways to predict [MASK]
# Uses context from both directions

# Usage: Sentence classification, token classification, etc.
```

```python
# GPT: Generative Pre-trained Transformer
# - Reads text LEFT-TO-RIGHT only (autoregressive)
# - Next token prediction training (predict next word)
# - Best for text generation tasks
# - DECODER ONLY

from transformers import AutoTokenizer, AutoModelForCausalLM

gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")

text = "The cat sat on"
# GPT predicts next token(s)
# Can only look to the LEFT (already generated text)

# Usage: Text generation, story writing, code completion
```

```python
# Comparison Table

| Aspect | BERT | GPT |
|--------|------|-----|
| Direction | Bidirectional | Unidirectional (left-to-right) |
| Training | Masked Language Model | Next Token Prediction |
| Output | Embeddings (understanding) | Sequences (generation) |
| Best For | Classification, NER, Q&A | Text generation, completion |
| Speed | Faster inference | Variable (generates token by token) |
| Pre-trained | Yes (ready to use) | Yes (ready to use) |

# Practical example:

# BERT: "Classify sentiment of this review"
text = "I love this movie! It's amazing."
# BERT understands entire sentence bidirectionally
# → Outputs: [positive, confidence=0.95]

# GPT: "Complete this sentence"
prompt = "I love this"
# GPT generates one token at a time
# → Outputs: " movie! It's amazing and I would recommend it to everyone."
```

---

## 13. Practical Applications & Real-World Examples

### Application 1: Semantic Search Engine

```python
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticSearchEngine:
    def __init__(self, documents, model=None):
        """
        Initialize semantic search engine
        
        Args:
            documents: List of documents (strings)
            model: Pre-trained Word2Vec model (optional)
        """
        self.documents = documents
        
        # Train or use provided model
        if model is None:
            tokenized = [doc.lower().split() for doc in documents]
            self.model = Word2Vec(tokenized, vector_size=100, window=5, min_count=1)
        else:
            self.model = model
        
        # Create document vectors
        self.doc_vectors = np.array([
            self._doc_to_vector(doc) for doc in documents
        ])
    
    def _doc_to_vector(self, doc):
        """Convert document to vector (average word embeddings)"""
        words = doc.lower().split()
        vectors = [self.model.wv[w] for w in words if w in self.model.wv]
        
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.model.wv.vector_size)
    
    def search(self, query, top_k=3):
        """
        Search for similar documents
        
        Args:
            query: Search query (string)
            top_k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        query_vector = self._doc_to_vector(query).reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (self.documents[i], similarities[i])
            for i in top_indices
        ]
        
        return results

# Example usage
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Python is a popular programming language",
    "Data science involves statistics and programming",
    "Natural language processing deals with text analysis",
]

search_engine = SemanticSearchEngine(documents)

# Search
query = "neural networks and deep learning"
results = search_engine.search(query, top_k=3)

print(f"Query: {query}\n")
print("Top results:")
for doc, score in results:
    print(f"  [{score:.3f}] {doc}")

# Output:
# Top results:
#   [0.856] Deep learning uses neural networks with multiple layers
#   [0.612] Machine learning is a subset of artificial intelligence
#   [0.445] Data science involves statistics and programming
```

### Application 2: Duplicate Document Detection

```python
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DuplicateDetector:
    def __init__(self, threshold=0.85):
        """
        Initialize duplicate detector
        
        Args:
            threshold: Similarity threshold (0-1) for marking as duplicate
        """
        self.threshold = threshold
        self.model = None
        self.documents = []
        self.doc_vectors = None
    
    def train(self, documents):
        """Train on documents"""
        self.documents = documents
        
        # Train Word2Vec
        tokenized = [doc.lower().split() for doc in documents]
        self.model = Word2Vec(tokenized, vector_size=100, window=5, min_count=1)
        
        # Create vectors
        self.doc_vectors = np.array([
            self._doc_to_vector(doc) for doc in documents
        ])
    
    def _doc_to_vector(self, doc):
        """Convert to vector"""
        words = doc.lower().split()
        vectors = [self.model.wv[w] for w in words if w in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.wv.vector_size)
    
    def find_duplicates(self):
        """Find all duplicate pairs"""
        duplicates = []
        
        # Compare all pairs
        n = len(self.documents)
        similarities = cosine_similarity(self.doc_vectors)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarities[i, j]
                
                if sim >= self.threshold:
                    duplicates.append({
                        'doc1_idx': i,
                        'doc2_idx': j,
                        'doc1': self.documents[i],
                        'doc2': self.documents[j],
                        'similarity': sim
                    })
        
        return duplicates

# Example
documents = [
    "Machine learning is powerful",
    "Machine learning is very powerful",
    "Deep learning uses neural networks",
    "Python is great",
    "Python programming language",
]

detector = DuplicateDetector(threshold=0.80)
detector.train(documents)
duplicates = detector.find_duplicates()

print("Found duplicates:\n")
for dup in duplicates:
    print(f"Similarity: {dup['similarity']:.3f}")
    print(f"  Doc1: {dup['doc1']}")
    print(f"  Doc2: {dup['doc2']}")
    print()
```

### Application 3: Recommendation System

```python
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentBasedRecommender:
    def __init__(self, items, descriptions):
        """
        Initialize recommender
        
        Args:
            items: List of item names
            descriptions: List of item descriptions
        """
        self.items = items
        self.descriptions = descriptions
        
        # Train Word2Vec on descriptions
        tokenized = [desc.lower().split() for desc in descriptions]
        self.model = Word2Vec(tokenized, vector_size=100, window=5, min_count=1)
        
        # Create item vectors
        self.item_vectors = np.array([
            self._desc_to_vector(desc) for desc in descriptions
        ])
    
    def _desc_to_vector(self, desc):
        """Convert description to vector"""
        words = desc.lower().split()
        vectors = [self.model.wv[w] for w in words if w in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.wv.vector_size)
    
    def recommend(self, item_idx, top_k=3):
        """Get recommendations based on item"""
        item_vector = self.item_vectors[item_idx].reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(item_vector, self.item_vectors)[0]
        
        # Get top-k (excluding the item itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        recommendations = [
            (self.items[i], similarities[i])
            for i in top_indices
        ]
        
        return recommendations

# Example
movies = [
    "Inception",
    "The Matrix",
    "Interstellar",
    "Titanic",
    "Avatar",
]

descriptions = [
    "mind bending sci-fi thriller about dreams",
    "hacker discovers reality is simulation sci-fi action",
    "epic space exploration time concepts",
    "romantic disaster epic film",
    "sci-fi fantasy adventure on alien planet",
]

recommender = ContentBasedRecommender(movies, descriptions)

# Get recommendations for "Inception"
recommendations = recommender.recommend(item_idx=0, top_k=3)

print("Movies similar to 'Inception':\n")
for movie, score in recommendations:
    print(f"  [{score:.3f}] {movie}")

# Output:
# Movies similar to 'Inception':
#   [0.821] The Matrix
#   [0.754] Interstellar
#   [0.312] Avatar
```

---

## 14. Transformer Fine-tuning for Specific Tasks

### Fine-tuning BERT for Sentiment Analysis

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from datasets import Dataset
import numpy as np

class SentimentAnalyzerBERT:
    def __init__(self, model_name="bert-base-uncased"):
        """Initialize BERT for sentiment analysis"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2  # binary: positive/negative
        )
    
    def prepare_data(self, texts, labels):
        """Prepare data for training"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'label': labels
        })
        
        return dataset
    
    def fine_tune(self, train_texts, train_labels, epochs=3):
        """Fine-tune model on task"""
        train_dataset = self.prepare_data(train_texts, train_labels)
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            logging_steps=10,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
    
    def predict(self, text):
        """Predict sentiment"""
        inputs = self.tokenizer(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0, prediction].item()
        
        sentiment = "positive" if prediction == 1 else "negative"
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': torch.softmax(logits, dim=1)[0, 0].item(),
                'positive': torch.softmax(logits, dim=1)[0, 1].item(),
            }
        }

# Usage
analyzer = SentimentAnalyzerBERT()

# Training data
train_texts = [
    "This movie is amazing!",
    "I love this product",
    "Terrible experience",
    "Worst movie ever",
    "Great quality and fast shipping",
]

train_labels = [1, 1, 0, 0, 1]  # 1=positive, 0=negative

# Fine-tune
analyzer.fine_tune(train_texts, train_labels, epochs=3)

# Predict
test_texts = [
    "Excellent service!",
    "Horrible quality",
    "Not bad, could be better"
]

print("Predictions:\n")
for text in test_texts:
    result = analyzer.predict(text)
    print(f"Text: {text}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Probabilities: {result['probabilities']}")
    print()
```

---

## Summary and Decision Tree

### Quick Decision Guide

```
START: Need to represent text?
│
├─→ Speed critical?
│  ├─→ YES → Use Bag of Words or TF-IDF
│  └─→ NO → Continue
│
├─→ Need semantic similarity?
│  ├─→ NO → Use BoW or TF-IDF
│  ├─→ YES, limited resources → Use Word2Vec
│  └─→ YES, resources available → Continue
│
├─→ Need context-dependent meanings (polysemy)?
│  ├─→ NO → Use Word2Vec/GloVe
│  ├─→ YES, limited GPU → Use ELMo
│  └─→ YES, GPU available → Use BERT/GPT
│
├─→ Task type?
│  ├─→ Classification/Understanding → Use BERT
│  ├─→ Generation/Completion → Use GPT
│  ├─→ Translation → Use mT5/mBART
│  └─→ Q&A → Use ALBERT/RoBERTa
│
└─→ Final Choice: ✓
```

### Performance Comparison Chart

```python
import matplotlib.pyplot as plt
import numpy as np

methods = ['BoW', 'TF-IDF', 'Word2Vec', 'GloVe', 'BERT', 'GPT-3']
semantic_understanding = [1, 2, 7, 7.5, 9, 9.5]
computational_cost = [1, 1.5, 3, 3.5, 8, 10]
inference_speed = [10, 9, 8, 8, 3, 2]
training_data_needed = [1, 1.5, 6, 6, 7, 10]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Semantic Understanding
axes[0, 0].barh(methods, semantic_understanding, color='steelblue')
axes[0, 0].set_xlabel('Semantic Understanding (0-10)')
axes[0, 0].set_title('Semantic Understanding Capability')
axes[0, 0].set_xlim(0, 10)

# Plot 2: Computational Cost
axes[0, 1].barh(methods, computational_cost, color='coral')
axes[0, 1].set_xlabel('Computational Cost (0-10)')
axes[0, 1].set_title('Computational Requirements')
axes[0, 1].set_xlim(0, 10)

# Plot 3: Inference Speed
axes[1, 0].barh(methods, inference_speed, color='lightgreen')
axes[1, 0].set_xlabel('Inference Speed (0-10)')
axes[1, 0].set_title('Inference Speed')
axes[1, 0].set_xlim(0, 10)

# Plot 4: Training Data Needed
axes[1, 1].barh(methods, training_data_needed, color='plum')
axes[1, 1].set_xlabel('Training Data Needed (0-10)')
axes[1, 1].set_title('Data Requirements')
axes[1, 1].set_xlim(0, 10)

plt.tight_layout()
plt.show()

print("""
Summary:
--------
BoW:       Fast baseline (no training needed)
TF-IDF:    Weighted BoW (good for search)
Word2Vec:  Good balance (semantic + fast)
GloVe:     Similar to Word2Vec (different training)
BERT:      Best for understanding (slow, needs GPU)
GPT-3:     Best for generation (requires API)
""")
```

---

## Final Checklist: Getting Started

```python
# ✓ Step 1: Install requirements
pip install gensim nltk scikit-learn transformers torch datasets

# ✓ Step 2: Choose your method
# - Simple task → Word2Vec
# - Complex task → BERT
# - Generation task → GPT

# ✓ Step 3: Prepare data
from nltk.tokenize import sent_tokenize, word_tokenize
sentences = [word_tokenize(s.lower()) for s in sent_tokenize(text)]

# ✓ Step 4: Train/Load model
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5)

# OR

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# ✓ Step 5: Use embeddings
embeddings = [model.wv[word] for word in words]  # Word2Vec
# OR
embeddings = model(tokenizer(text, return_tensors='pt'))  # BERT

# ✓ Step 6: Downstream task
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(embeddings, labels)
```

---

## Recommended Learning Path

1. **Week 1-2:** Understand fundamentals
   - [ ] Read this guide thoroughly
   - [ ] Understand BoW → TF-IDF → Word2Vec progression
   - [ ] Grasp Word2Vec math (skip-gram, context windows)

2. **Week 3:** Hands-on Word2Vec
   - [ ] Implement Word2Vec from scratch (optional)
   - [ ] Train on real dataset
   - [ ] Explore word analogies
   - [ ] Visualize with PCA

3. **Week 4:** Advanced embeddings
   - [ ] Learn about FastText (handles OOV)
   - [ ] Study GloVe algorithm
   - [ ] Understand why contextual embeddings matter

4. **Week 5-6:** Transformers
   - [ ] Understand attention mechanism
   - [ ] Learn BERT architecture
   - [ ] Fine-tune BERT on custom task

5. **Week 7:** Production deployment
   - [ ] Build embedding service
   - [ ] Handle edge cases
   - [ ] Optimize for latency

---

**End of Complete Word Embeddings Guide**

This comprehensive guide covers everything from basic concepts to production deployment. 
Practice with real datasets and you'll become proficient in NLP embeddings!

Key Takeaway: Choose simplest method that solves your problem. Start with Word2Vec, 
upgrade to BERT/Transformers only if needed. 90% of tasks can be solved with Word2Vec or TF-IDF!