# Text Representation in NLP: Complete Guide from Scratch to Advanced

A comprehensive guide covering all text representation techniques, from basic preprocessing to advanced embeddings, with examples, comparisons, and interview questions.

---

## Table of Contents

1. [Text Preprocessing Fundamentals](#1-text-preprocessing-fundamentals)
2. [Text Representation Types Overview](#2-text-representation-types-overview)
3. [One-Hot Encoding for Text](#3-one-hot-encoding-for-text)
4. [Bag of Words (BoW)](#4-bag-of-words-bow)
5. [TF-IDF (Term Frequency-Inverse Document Frequency)](#5-tf-idf)
6. [N-grams (Bigrams, Trigrams)](#6-n-grams)
7. [Word Embeddings](#7-word-embeddings)
8. [Similarity Measures](#8-similarity-measures)
9. [Complete Comparison](#9-complete-comparison)
10. [Advanced Topics](#10-advanced-topics)
11. [Interview Questions](#11-interview-questions)
12. [Practical Applications](#12-practical-applications)

---

## 1. Text Preprocessing Fundamentals

### Why Preprocessing?

Raw text contains noise, inconsistencies, and variations that make it difficult for machines to understand. Preprocessing standardizes text for better model performance.

### Contractions Handling

**What are contractions?**
- Shortened forms of words: "don't" → "do not", "I'm" → "I am"

**Why expand them?**
- Reduces vocabulary size
- Maintains semantic meaning
- Improves model consistency

```python
import contractions

text = "I don't think we're ready for this"
expanded = contractions.fix(text)
print(expanded)
# Output: "I do not think we are ready for this"
```

**Using contractions library:**

```python
# Installation
# pip install contractions

import contractions

# Basic usage
text = "I'll be there, won't you?"
expanded = contractions.fix(text)
print(expanded)  # "I will be there, will not you?"

# Custom handling
def expand_contractions(text):
    """Expand contractions in text"""
    return contractions.fix(text)

# Example
sentences = [
    "I'm learning NLP",
    "They've completed the course",
    "She'd rather stay home"
]

for sent in sentences:
    print(f"Original: {sent}")
    print(f"Expanded: {expand_contractions(sent)}\n")
```

### Complete Preprocessing Pipeline

```python
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import contractions

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Complete preprocessing pipeline"""
        # 1. Lowercase
        text = text.lower()
        
        # 2. Expand contractions
        text = contractions.fix(text)
        
        # 3. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # 4. Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # 5. Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # 6. Remove numbers (optional)
        text = re.sub(r'\d+', '', text)
        
        # 7. Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 8. Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """Tokenize text"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem(self, tokens):
        """Apply stemming"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize(self, tokens):
        """Apply lemmatization"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text, remove_stopwords=True, use_lemmatization=True):
        """Full preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize or stem
        if use_lemmatization:
            tokens = self.lemmatize(tokens)
        else:
            tokens = self.stem(tokens)
        
        return tokens

# Usage
preprocessor = TextPreprocessor()

text = """
I'm learning NLP! Check out https://example.com 
#MachineLearning @AIExpert - it's amazing!!!
Contact: test@email.com for info. 
"""

# Clean only
cleaned = preprocessor.clean_text(text)
print(f"Cleaned: {cleaned}")

# Full preprocessing
tokens = preprocessor.preprocess(text)
print(f"Tokens: {tokens}")
```

---

## 2. Text Representation Types Overview

### Two Main Categories

#### 1. **Sparse Representation**
- Most values are zero
- High dimensionality
- Examples: One-Hot, BoW, TF-IDF

#### 2. **Dense Representation (Semantic)**
- Low dimensionality (typically 50-300 dimensions)
- Captures semantic meaning
- Examples: Word2Vec, GloVe, FastText, BERT

### Comparison Table

| Type | Dimensionality | Semantic Info | Sparsity | Use Case |
|------|---------------|---------------|----------|----------|
| One-Hot | Vocabulary size | No | Very High | Small vocab |
| BoW | Vocabulary size | No | High | Document classification |
| TF-IDF | Vocabulary size | No | High | Information retrieval |
| Word2Vec | 100-300 | Yes | Low | Similarity tasks |
| BERT | 768 | Yes (contextual) | Low | Advanced NLP |

---

## 3. One-Hot Encoding for Text

### What is One-Hot Encoding?

Each word is represented as a vector with:
- Length = vocabulary size
- 1 at the word's index position
- 0 everywhere else

### Visual Example

```
Vocabulary: ["cat", "dog", "bird", "fish"]

"cat"  → [1, 0, 0, 0]
"dog"  → [0, 1, 0, 0]
"bird" → [0, 0, 1, 0]
"fish" → [0, 0, 0, 1]
```

### Why is it Sparse?

**Example:**
- Vocabulary size: 10,000 words
- Each word vector: 10,000 dimensions
- Only 1 value is 1, 9,999 are 0
- Sparsity = 9,999/10,000 = 99.99%

### Implementation

```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Method 1: Manual implementation
def manual_one_hot(words):
    """Create one-hot encoding manually"""
    # Create vocabulary
    vocab = sorted(list(set(words)))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Create one-hot vectors
    one_hot = np.zeros((len(words), len(vocab)))
    for i, word in enumerate(words):
        one_hot[i, word_to_idx[word]] = 1
    
    return one_hot, vocab, word_to_idx

# Example
words = ["cat", "dog", "bird", "cat", "fish"]
vectors, vocab, word_to_idx = manual_one_hot(words)

print(f"Vocabulary: {vocab}")
print(f"\nOne-hot encoding for '{words[0]}':")
print(vectors[0])

# Method 2: Using sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

words = ["cat", "dog", "bird", "cat", "fish"]

# Label encode first
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(words)

# One-hot encode
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print("\nUsing sklearn:")
print(onehot_encoded)
```

### For Sentences

```python
def sentence_to_onehot(sentence, vocab):
    """Convert sentence to one-hot matrix"""
    words = sentence.lower().split()
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    
    # Create matrix
    matrix = np.zeros((len(words), len(vocab)))
    
    for i, word in enumerate(words):
        if word in vocab_dict:
            matrix[i, vocab_dict[word]] = 1
    
    return matrix

# Example
vocab = ["the", "cat", "sat", "on", "mat", "dog"]
sentence = "the cat sat on the mat"

onehot_matrix = sentence_to_onehot(sentence, vocab)
print(f"Sentence: {sentence}")
print(f"Shape: {onehot_matrix.shape}")
print(f"Matrix:\n{onehot_matrix}")
```

### Drawbacks of One-Hot Encoding

1. **No semantic information**
   - "king" and "queen" are equally different as "king" and "apple"
   - Cannot capture word relationships

2. **Extremely sparse**
   - 99%+ zeros for large vocabularies
   - Memory inefficient

3. **High dimensionality**
   - Vector size = vocabulary size
   - Can be 10,000+ dimensions

4. **No word order**
   - "dog bites man" vs "man bites dog" look identical

5. **Out-of-vocabulary (OOV) problem**
   - Cannot handle new words

```python
# Demonstrating no semantic similarity
def cosine_similarity(v1, v2):
    """Calculate cosine similarity"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# One-hot vectors
vocab = ["king", "queen", "man", "woman", "apple", "orange"]
king = [1, 0, 0, 0, 0, 0]
queen = [0, 1, 0, 0, 0, 0]
apple = [0, 0, 0, 0, 1, 0]

print(f"Similarity(king, queen): {cosine_similarity(king, queen)}")  # 0.0
print(f"Similarity(king, apple): {cosine_similarity(king, apple)}")  # 0.0
# Both are 0! No semantic understanding
```

---

## 4. Bag of Words (BoW)

### What is Bag of Words?

A document is represented as a vector where each position corresponds to a word in the vocabulary, and the value is the **count** of that word in the document.

### Key Characteristics

- **Ignores word order** ("bag" of words)
- **Counts word frequency**
- **Fixed vocabulary**

### Visual Example

```
Vocabulary: ["the", "cat", "sat", "on", "mat", "dog"]

Document 1: "the cat sat on the mat"
BoW: [2, 1, 1, 1, 1, 0]
      ↑  ↑  ↑  ↑  ↑  ↑
     the cat sat on mat dog

Document 2: "the dog sat on the mat"
BoW: [2, 0, 1, 1, 1, 1]
```

### Implementation

```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Sample documents
documents = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat and the dog",
    "cat sat on mat"
]

# Create BoW using sklearn
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

# View vocabulary
vocab = vectorizer.get_feature_names_out()
print(f"Vocabulary: {vocab}")

# View BoW matrix
print(f"\nBoW Matrix (sparse):\n{bow_matrix}")
print(f"\nBoW Matrix (dense):\n{bow_matrix.toarray()}")

# Analyze specific document
doc_idx = 0
print(f"\nDocument: '{documents[doc_idx]}'")
print(f"BoW vector: {bow_matrix[doc_idx].toarray()[0]}")

# Word counts
for word, count in zip(vocab, bow_matrix[doc_idx].toarray()[0]):
    if count > 0:
        print(f"  {word}: {count}")
```

### Manual Implementation

```python
from collections import Counter

class SimpleBagOfWords:
    def __init__(self):
        self.vocabulary = []
        self.vocab_dict = {}
    
    def fit(self, documents):
        """Build vocabulary from documents"""
        # Tokenize and collect all words
        all_words = []
        for doc in documents:
            words = doc.lower().split()
            all_words.extend(words)
        
        # Create vocabulary
        self.vocabulary = sorted(list(set(all_words)))
        self.vocab_dict = {word: idx for idx, word in enumerate(self.vocabulary)}
        
        return self
    
    def transform(self, documents):
        """Transform documents to BoW vectors"""
        bow_matrix = np.zeros((len(documents), len(self.vocabulary)))
        
        for doc_idx, doc in enumerate(documents):
            words = doc.lower().split()
            word_counts = Counter(words)
            
            for word, count in word_counts.items():
                if word in self.vocab_dict:
                    bow_matrix[doc_idx, self.vocab_dict[word]] = count
        
        return bow_matrix
    
    def fit_transform(self, documents):
        """Fit and transform in one step"""
        return self.fit(documents).transform(documents)

# Usage
bow = SimpleBagOfWords()
documents = [
    "the cat sat on the mat",
    "the dog sat on the mat"
]

bow_matrix = bow.fit_transform(documents)
print(f"Vocabulary: {bow.vocabulary}")
print(f"BoW Matrix:\n{bow_matrix}")
```

### Advanced CountVectorizer Options

```python
from sklearn.feature_extraction.text import CountVectorizer

# With various parameters
vectorizer = CountVectorizer(
    max_features=1000,        # Limit vocabulary size
    min_df=2,                 # Ignore words appearing in < 2 documents
    max_df=0.8,               # Ignore words appearing in > 80% documents
    stop_words='english',     # Remove English stop words
    ngram_range=(1, 2),       # Include unigrams and bigrams
    lowercase=True,           # Convert to lowercase
    token_pattern=r'\b\w+\b'  # Token extraction pattern
)

documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are pets",
    "The cat is a pet"
]

bow_matrix = vectorizer.fit_transform(documents)
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Features: {vectorizer.get_feature_names_out()}")
print(f"Shape: {bow_matrix.shape}")
```

### Drawbacks of BoW

1. **Loss of word order**
   ```python
   # These two sentences produce same BoW
   "dog bites man" → [1, 1, 1]
   "man bites dog" → [1, 1, 1]
   ```

2. **High dimensionality**
   - Vocabulary can be 10,000+ words

3. **Sparse representation**
   - Most values are zero

4. **No semantic meaning**
   - Cannot capture synonyms or related words

5. **Common words dominate**
   - "the", "is", "and" have high counts but low importance

---

## 5. TF-IDF

### What is TF-IDF?

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a numerical statistic that reflects how important a word is to a document in a collection of documents.

### Components

#### 1. **TF (Term Frequency)**

**What:** How often a word appears in a document.

**Formula:**
```
TF(word, document) = (Number of times word appears in document) / (Total words in document)
```

**Example:**
```
Document: "the cat sat on the mat"
Total words: 6

TF("the") = 2/6 = 0.333
TF("cat") = 1/6 = 0.167
TF("sat") = 1/6 = 0.167
```

#### 2. **IDF (Inverse Document Frequency)**

**What:** How rare or important a word is across all documents.

**Intuition:**
- Common words (like "the") appear in many documents → low IDF → less important
- Rare words appear in few documents → high IDF → more important

**Formula:**
```
IDF(word) = log((Total number of documents) / (Number of documents containing the word))
```

**Example:**
```
Corpus: 100 documents

Word "the" appears in 100 documents:
IDF("the") = log(100/100) = log(1) = 0

Word "python" appears in 5 documents:
IDF("python") = log(100/5) = log(20) = 2.996

Word "machine" appears in 20 documents:
IDF("machine") = log(100/20) = log(5) = 1.609
```

#### 3. **TF-IDF (Together)**

**Formula:**
```
TF-IDF(word, document) = TF(word, document) × IDF(word)
```

**Interpretation:**
- High TF-IDF: Word is frequent in this document but rare overall → **important**
- Low TF-IDF: Word is either rare in this document or common everywhere → **less important**

### Step-by-Step Example

```
Corpus of 3 documents:

Doc 1: "the cat sat on the mat"
Doc 2: "the dog sat on the log"
Doc 3: "cats and dogs are pets"

Calculate TF-IDF for word "cat" in Doc 1:

Step 1: Calculate TF
TF("cat", Doc1) = 1/6 = 0.167

Step 2: Calculate IDF
Documents containing "cat": Doc 1 only = 1
Total documents: 3
IDF("cat") = log(3/1) = log(3) = 1.099

Step 3: Calculate TF-IDF
TF-IDF("cat", Doc1) = 0.167 × 1.099 = 0.183
```

### Implementation from Scratch

```python
import numpy as np
import math
from collections import Counter

class TFIDFFromScratch:
    def __init__(self):
        self.vocabulary = []
        self.idf_values = {}
        self.documents = []
    
    def fit(self, documents):
        """Calculate IDF values"""
        self.documents = documents
        n_documents = len(documents)
        
        # Build vocabulary
        all_words = set()
        for doc in documents:
            words = doc.lower().split()
            all_words.update(words)
        
        self.vocabulary = sorted(list(all_words))
        
        # Calculate IDF for each word
        for word in self.vocabulary:
            # Count documents containing this word
            doc_count = sum(1 for doc in documents if word in doc.lower().split())
            
            # Calculate IDF
            self.idf_values[word] = math.log(n_documents / doc_count)
        
        return self
    
    def transform(self, documents):
        """Calculate TF-IDF vectors"""
        tfidf_matrix = []
        
        for doc in documents:
            words = doc.lower().split()
            word_counts = Counter(words)
            total_words = len(words)
            
            # Calculate TF-IDF for each word in vocabulary
            tfidf_vector = []
            for word in self.vocabulary:
                # TF
                tf = word_counts.get(word, 0) / total_words
                
                # IDF
                idf = self.idf_values.get(word, 0)
                
                # TF-IDF
                tfidf = tf * idf
                tfidf_vector.append(tfidf)
            
            tfidf_matrix.append(tfidf_vector)
        
        return np.array(tfidf_matrix)
    
    def fit_transform(self, documents):
        """Fit and transform in one step"""
        return self.fit(documents).transform(documents)

# Example usage
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

tfidf = TFIDFFromScratch()
tfidf_matrix = tfidf.fit_transform(documents)

print("Vocabulary:", tfidf.vocabulary)
print("\nIDF values:")
for word, idf in sorted(tfidf.idf_values.items(), key=lambda x: -x[1])[:5]:
    print(f"  {word}: {idf:.4f}")

print(f"\nTF-IDF Matrix shape: {tfidf_matrix.shape}")
print(f"TF-IDF Matrix:\n{tfidf_matrix}")
```

### Using sklearn's TfidfVectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets",
    "the cat is a pet"
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=100,
    stop_words='english',
    ngram_range=(1, 2)
)

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(documents)

# View results
vocab = vectorizer.get_feature_names_out()
print(f"Vocabulary: {vocab}")
print(f"\nTF-IDF Matrix:\n{tfidf_matrix.toarray()}")

# Analyze specific document
doc_idx = 0
feature_scores = list(zip(vocab, tfidf_matrix[doc_idx].toarray()[0]))
feature_scores = sorted(feature_scores, key=lambda x: -x[1])

print(f"\nTop features for document {doc_idx}:")
for word, score in feature_scores[:5]:
    print(f"  {word}: {score:.4f}")
```

### TF-IDF Variants

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Different TF-IDF configurations

# 1. Binary TF (presence/absence)
vectorizer_binary = TfidfVectorizer(binary=True)

# 2. Sublinear TF (use log scale for TF)
vectorizer_sublinear = TfidfVectorizer(sublinear_tf=True)

# 3. Custom norm
vectorizer_norm = TfidfVectorizer(norm='l1')  # L1 normalization

# 4. Smooth IDF
vectorizer_smooth = TfidfVectorizer(smooth_idf=True)  # Add 1 to document frequencies
```

### Advantages of TF-IDF over BoW

1. **Reduces importance of common words**
   ```python
   # "the" appears everywhere → low TF-IDF
   # "python" appears rarely → high TF-IDF
   ```

2. **Highlights important words**
   - Words unique to a document get higher scores

3. **Better for information retrieval**
   - Search engines use TF-IDF-like measures

4. **Normalized representation**
   - Documents of different lengths comparable

### Drawbacks of TF-IDF

1. **Still sparse**
   - Same dimensionality as BoW

2. **No semantic understanding**
   - "happy" and "joyful" treated as completely different

3. **Loses word order**
   - "not good" vs "good not" identical

4. **Fixed vocabulary**
   - Cannot handle new words

```python
# Demonstrating limitations
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "The movie was not good",
    "The movie was good"
]

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(docs)

# These should have opposite meanings but might look similar
print("TF-IDF vectors:")
print(tfidf.toarray())
```

---

## 6. N-grams

### What are N-grams?

**N-gram:** A contiguous sequence of N items (words) from a text.

- **Unigram (1-gram):** Single words → "cat", "sat", "mat"
- **Bigram (2-gram):** Two consecutive words → "the cat", "cat sat"
- **Trigram (3-gram):** Three consecutive words → "the cat sat"

### Why N-grams?

**Problem with Bag of Words:**
```
"not good" and "good not" → same representation
```

**Solution: Use bigrams**
```
"not good" → contains bigram "not good"
"good not" → contains bigram "good not"
→ Different representations!
```

### Bigram Examples

```
Sentence: "the cat sat on the mat"

Bigrams:
1. "the cat"
2. "cat sat"
3. "sat on"
4. "on the"
5. "the mat"
```

### Visual Example

```
Sentence: "I love machine learning"

Unigrams: ["I", "love", "machine", "learning"]

Bigrams: ["I love", "love machine", "machine learning"]

Trigrams: ["I love machine", "love machine learning"]
```

### Implementation

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "machine learning is great",
    "deep learning is powerful",
    "machine learning and deep learning"
]

# Unigrams only
vectorizer_unigram = CountVectorizer(ngram_range=(1, 1))
unigram_matrix = vectorizer_unigram.fit_transform(documents)
print("Unigrams:", vectorizer_unigram.get_feature_names_out())

# Bigrams only
vectorizer_bigram = CountVectorizer(ngram_range=(2, 2))
bigram_matrix = vectorizer_bigram.fit_transform(documents)
print("\nBigrams:", vectorizer_bigram.get_feature_names_out())

# Both unigrams and bigrams
vectorizer_both = CountVectorizer(ngram_range=(1, 2))
both_matrix = vectorizer_both.fit_transform(documents)
print("\nUnigrams + Bigrams:", vectorizer_both.get_feature_names_out())
```

### Manual N-gram Generation

```python
def generate_ngrams(text, n):
    """Generate n-grams from text"""
    words = text.split()
    ngrams = []
    
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

# Example
text = "the quick brown fox jumps over the lazy dog"

print("Unigrams:", generate_ngrams(text, 1))
print("Bigrams:", generate_ngrams(text, 2))
print("Trigrams:", generate_ngrams(text, 3))
```

### Character N-grams

```python
def char_ngrams(text, n):
    """Generate character n-grams"""
    return [text[i:i+n] for i in range(len(text) - n + 1)]

# Useful for misspellings and unknown words
word = "learning"
print(f"Character bigrams of '{word}':")
print(char_ngrams(word, 2))

# sklearn implementation
vectorizer_char = CountVectorizer(
    analyzer='char',
    ngram_range=(2, 3)
)

docs = ["learning", "machine", "deep"]
char_matrix = vectorizer_char.fit_transform(docs)
print("\nCharacter n-grams:")
print(vectorizer_char.get_feature_names_out())
```

### TF-IDF with N-grams

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "natural language processing is amazing",
    "machine learning is powerful",
    "deep learning revolutionizes AI"
]

# TF-IDF with bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(documents)

# Show top features
feature_names = vectorizer.get_feature_names_out()
doc_idx = 0

feature_scores = list(zip(feature_names, tfidf_matrix[doc_idx].toarray()[0]))
feature_scores = sorted(feature_scores, key=lambda x: -x[1])

print(f"Top features (including bigrams) for document {doc_idx}:")
for word, score in feature_scores[:10]:
    print(f"  {word}: {score:.4f}")
```

### Advantages of N-grams

1. **Captures word order** (partially)
   ```python
   "not good" ≠ "good not"
   ```

2. **Better context understanding**
   ```python
   "New York" treated as single unit
   "machine learning" recognized as phrase
   ```

3. **Improves classification**
   - Better features for models

### Disadvantages of N-grams

1. **Vocabulary explosion**
   ```python
   # 1,000 words
   # Unigrams: 1,000 features
   # Bigrams: up to 1,000,000 combinations
   # Trigrams: up to 1,000,000,000 combinations
   ```

2. **Increased sparsity**
   - Most bigrams appear only once

3. **Computational cost**
   - More features = more memory and processing

4. **Data sparsity**
   - Need more data to learn meaningful patterns

```python
# Demonstrating vocabulary explosion
from sklearn.feature_extraction.text import CountVectorizer

documents = ["the cat sat on the mat"] * 10

for n in [(1,1), (2,2), (3,3), (1,2), (1,3)]:
    vectorizer = CountVectorizer(ngram_range=n)
    matrix = vectorizer.fit_transform(documents)
    print(f"N-gram range {n}: {len(vectorizer.vocabulary_)} features")
```

---

## 7. Word Embeddings

### What are Word Embeddings?

**Dense vector representations** of words that capture semantic meaning in a low-dimensional space (typically 50-300 dimensions).

### Key Difference from Previous Methods

| Feature | One-Hot/BoW/TF-IDF | Word Embeddings |
|---------|-------------------|-----------------|
| **Dimensionality** | Vocabulary size (10,000+) | Fixed (50-300) |
| **Sparsity** | Very sparse (99%+ zeros) | Dense (all non-zero) |
| **Semantics** | No meaning | Captures meaning |
| **Similarity** | Cannot measure | Can measure |

### Core Idea

```
Words with similar meanings should have similar vectors

"king" ≈ "queen" ≈ "monarch"
"dog" ≈ "cat" ≈ "pet"

"king" ≠ "apple"
```

### Types of Word Embeddings

#### 1. **Word2Vec**

Two architectures:
- **CBOW (Continuous Bag of Words):** Predict target word from context
- **Skip-gram:** Predict context words from target word

#### 2. **GloVe (Global Vectors)**

- Based on word co-occurrence statistics
- Trained on global corpus statistics

#### 3. **FastText**

- Extension of Word2Vec
- Uses character n-grams
- Can handle out-of-vocabulary words

#### 4. **BERT/Transformers**

- Contextual embeddings
- Same word has different vectors in different contexts

### Word2Vec Example

```python
# Installation: pip install gensim

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample corpus
sentences = [
    "king is a male ruler",
    "queen is a female ruler",
    "man is to woman as king is to queen",
    "prince is the son of a king",
    "princess is the daughter of a queen",
    "dog and cat are pets",
    "puppy is a young dog",
    "kitten is a young cat"
]

# Tokenize
tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]

# Train Word2Vec
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,      # Dimension of embeddings
    window=5,             # Context window size
    min_count=1,          # Minimum word frequency
    workers=4,            # Parallel processing
    sg=0                  # 0=CBOW, 1=Skip-gram
)

# Get word vector
king_vector = model.wv['king']
print(f"'king' vector (first 10 dims): {king_vector[:10]}")

# Find similar words
similar_to_king = model.wv.most_similar('king', topn=5)
print(f"\nWords similar to 'king': {similar_to_king}")

# Word arithmetic
# king - man + woman ≈ queen
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
print(f"\nking - man + woman = {result[0][0]}")

# Similarity score
similarity = model.wv.similarity('king', 'queen')
print(f"\nSimilarity(king, queen): {similarity:.4f}")
```

### Using Pre-trained Embeddings

```python
import gensim.downloader as api

# Load pre-trained GloVe embeddings
# Options: 'glove-wiki-gigaword-100', 'word2vec-google-news-300', etc.
embeddings = api.load('glove-wiki-gigaword-100')

# Get vector
word = 'computer'
vector = embeddings[word]
print(f"Vector for '{word}': {vector[:10]}")

# Find similar words
similar_words = embeddings.most_similar('computer', topn=10)
print(f"\nSimilar to 'computer':")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

# Word analogies
result = embeddings.most_similar(
    positive=['woman', 'king'],
    negative=['man'],
    topn=3
)
print(f"\nwoman + king - man:")
for word, score in result:
    print(f"  {word}: {score:.4f}")
```

### Document Embeddings from Word Embeddings

```python
import numpy as np

def document_vector(doc, model):
    """Average word vectors in document"""
    words = doc.lower().split()
    word_vectors = []
    
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Example
doc1 = "king and queen rule the kingdom"
doc2 = "dog and cat are common pets"

doc1_vec = document_vector(doc1, model)
doc2_vec = document_vector(doc2, model)

# Compare documents
from scipy.spatial.distance import cosine
similarity = 1 - cosine(doc1_vec, doc2_vec)
print(f"Document similarity: {similarity:.4f}")
```

### Advantages of Word Embeddings

1. **Semantic similarity**
   ```python
   similarity('happy', 'joyful') > similarity('happy', 'sad')
   ```

2. **Low dimensionality**
   - 100-300 dimensions vs 10,000+ for one-hot

3. **Word relationships**
   ```python
   king - man + woman ≈ queen
   ```

4. **Transfer learning**
   - Use pre-trained embeddings

5. **Dense representation**
   - No sparsity issues

### Drawbacks of Word Embeddings

1. **Fixed vocabulary**
   - Cannot handle new words (except FastText)

2. **No context**
   - "bank" (river) vs "bank" (money) same vector

3. **Training time**
   - Need large corpus

4. **Memory**
   - Pre-trained models can be large (1-2 GB)

---

## 8. Similarity Measures

### Why Similarity Measures?

To compare:
- Word vectors
- Document vectors
- Sentences
- User queries with documents

### 1. Euclidean Distance

**Formula:**
```
distance = √[(x₁ - x₂)² + (y₁ - y₂)² + ...]
```

**Intuition:** Straight-line distance in n-dimensional space

**Implementation:**

```python
import numpy as np

def euclidean_distance(v1, v2):
    """Calculate Euclidean distance"""
    return np.linalg.norm(v1 - v2)

# Example
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

distance = euclidean_distance(v1, v2)
print(f"Euclidean distance: {distance:.4f}")

# Using scipy
from scipy.spatial.distance import euclidean
distance = euclidean(v1, v2)
print(f"Using scipy: {distance:.4f}")
```

**When to use:**
- When magnitude matters
- In coordinate systems
- For clustering (K-means)

**Drawback:**
- Sensitive to vector magnitude
- Not scale-invariant

```python
# Problem: Different magnitudes
v1 = np.array([1, 1])      # Length ≈ 1.41
v2 = np.array([10, 10])    # Length ≈ 14.1, same direction!

print(f"Distance: {euclidean_distance(v1, v2):.4f}")  # Large!
# But they point in the same direction
```

### 2. Cosine Similarity

**Formula:**
```
cosine_similarity = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product
- ||A|| = magnitude of A
```

**Range:** -1 to 1
- 1: Identical direction
- 0: Perpendicular
- -1: Opposite direction

**Intuition:** Measures angle between vectors, ignoring magnitude

**Implementation:**

```python
import numpy as np

def cosine_similarity(v1, v2):
    """Calculate cosine similarity"""
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    
    return dot_product / (magnitude_v1 * magnitude_v2)

# Example
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # Same direction, different magnitude

cos_sim = cosine_similarity(v1, v2)
print(f"Cosine similarity: {cos_sim:.4f}")  # Should be 1.0

# Using sklearn
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

v1 = v1.reshape(1, -1)
v2 = v2.reshape(1, -1)
similarity = sk_cosine(v1, v2)[0][0]
print(f"Using sklearn: {similarity:.4f}")
```

**Visual Example:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Create vectors
v1 = np.array([3, 4])
v2 = np.array([6, 8])  # Same direction
v3 = np.array([4, 3])  # Different direction

# Calculate similarities
sim_v1_v2 = cosine_similarity(v1, v2)
sim_v1_v3 = cosine_similarity(v1, v3)

print(f"Similarity(v1, v2): {sim_v1_v2:.4f}")  # High (same direction)
print(f"Similarity(v1, v3): {sim_v1_v3:.4f}")  # Lower (different direction)

# Plot
plt.figure(figsize=(8, 8))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
plt.quiver(0, 0, v3[0], v3[1], angles='xy', scale_units='xy', scale=1, color='g', label='v3')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
plt.title('Vector Comparison')
plt.grid(True)
plt.show()
```

**When to use Cosine:**
- Text similarity (TF-IDF vectors)
- Recommendation systems
- Word embeddings
- When direction matters more than magnitude

### Euclidean vs Cosine: Comparison

```python
# Demonstrating difference

# Case 1: Same direction, different magnitude
v1 = np.array([1, 1])
v2 = np.array([10, 10])

print("Case 1: Same direction, different magnitude")
print(f"Euclidean distance: {euclidean_distance(v1, v2):.4f}")  # Large
print(f"Cosine similarity: {cosine_similarity(v1, v2):.4f}")    # 1.0 (perfect)

# Case 2: Different direction, similar magnitude
v1 = np.array([5, 5])
v2 = np.array([5, -5])

print("\nCase 2: Different direction, similar magnitude")
print(f"Euclidean distance: {euclidean_distance(v1, v2):.4f}")  # Moderate
print(f"Cosine similarity: {cosine_similarity(v1, v2):.4f}")    # 0.0 (perpendicular)
```

### Document Similarity Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "machine learning is amazing",
    "deep learning is a subset of machine learning",
    "natural language processing uses machine learning",
    "I love eating pizza"
]

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate similarities
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Document Similarity Matrix:")
print(similarity_matrix)

# Find most similar to document 0
doc_idx = 0
similarities = list(enumerate(similarity_matrix[doc_idx]))
similarities = sorted(similarities, key=lambda x: -x[1])

print(f"\nDocuments similar to: '{documents[doc_idx]}'")
for idx, score in similarities[1:]:  # Skip itself
    print(f"  {documents[idx]}: {score:.4f}")
```

### Other Similarity Measures

```python
from scipy.spatial.distance import *

v1 = np.array([1, 2, 3, 4])
v2 = np.array([2, 3, 4, 5])

print("Different similarity measures:")
print(f"Euclidean: {euclidean(v1, v2):.4f}")
print(f"Cosine: {1 - cosine(v1, v2):.4f}")  # scipy returns distance
print(f"Manhattan: {cityblock(v1, v2):.4f}")
print(f"Chebyshev: {chebyshev(v1, v2):.4f}")
print(f"Jaccard: {jaccard(v1 > 0, v2 > 0):.4f}")  # For binary
```

---

## 9. Complete Comparison

### Comparison Table

| Method | Dimensionality | Semantic | Sparsity | Context | OOV Handling | Use Case |
|--------|---------------|----------|----------|---------|--------------|----------|
| **One-Hot** | Vocab size | ❌ | Very High | ❌ | ❌ | Small vocabs only |
| **BoW** | Vocab size | ❌ | High | ❌ | ❌ | Document classification |
| **TF-IDF** | Vocab size | ❌ | High | ❌ | ❌ | Information retrieval |
| **N-grams** | Vocab size × N | Partial | Very High | Partial | ❌ | Phrase detection |
| **Word2Vec** | 100-300 | ✅ | Low | ❌ | ⚠️ (FastText) | Similarity tasks |
| **GloVe** | 100-300 | ✅ | Low | ❌ | ❌ | General embeddings |
| **BERT** | 768 | ✅ | Low | ✅ | ✅ | State-of-the-art NLP |

### Detailed Comparison

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Sample documents
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

# 1. One-Hot / BoW
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(documents)
print(f"BoW shape: {bow_matrix.shape}")
print(f"BoW sparsity: {1 - (bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])):.2%}")

# 2. TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print(f"\nTF-IDF shape: {tfidf_matrix.shape}")
print(f"TF-IDF sparsity: {1 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.2%}")

# 3. Word Embeddings (average)
tokenized = [doc.split() for doc in documents]
w2v_model = Word2Vec(tokenized, vector_size=50, min_count=1)

def doc_to_vec(doc, model):
    words = doc.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

w2v_matrix = np.array([doc_to_vec(doc, w2v_model) for doc in documents])
print(f"\nWord2Vec shape: {w2v_matrix.shape}")
print(f"Word2Vec sparsity: {(w2v_matrix == 0).sum() / w2v_matrix.size:.2%}")
```

### Drawbacks Summary

#### One-Hot Encoding
1. ❌ No semantic similarity
2. ❌ Extremely sparse (99%+ zeros)
3. ❌ High dimensionality = vocabulary size
4. ❌ Cannot handle OOV words
5. ❌ All words equally different

#### Bag of Words
1. ❌ Loses word order
2. ❌ High dimensionality
3. ❌ Sparse representation
4. ❌ No semantic meaning
5. ❌ Common words dominate

#### TF-IDF
1. ❌ Still sparse
2. ❌ No semantic understanding
3. ❌ Loses word order
4. ❌ Fixed vocabulary (OOV problem)
5. ⚠️ Better than BoW but not semantic

#### N-grams
1. ❌ Vocabulary explosion (exponential growth)
2. ❌ Increased sparsity
3. ❌ Computational cost
4. ❌ Needs more training data
5. ⚠️ Partial context only

#### Word Embeddings (Word2Vec/GloVe)
1. ❌ Fixed vocabulary (except FastText)
2. ❌ No context (same vector for "bank" river vs money)
3. ❌ Training requires large corpus
4. ❌ Memory intensive for pre-trained models
5. ⚠️ Best among traditional methods

### Decision Tree: Which to Use?

```
Start
  |
  ├─ Need semantic similarity?
  │   ├─ YES → Word Embeddings (Word2Vec, GloVe)
  │   │         or BERT (for contextual)
  │   └─ NO ↓
  │
  ├─ Need to handle phrases?
  │   ├─ YES → N-grams (with BoW or TF-IDF)
  │   └─ NO ↓
  │
  ├─ Information retrieval / search?
  │   ├─ YES → TF-IDF
  │   └─ NO ↓
  │
  ├─ Simple document classification?
  │   ├─ YES → BoW or TF-IDF
  │   └─ NO ↓
  │
  └─ Very small vocabulary (<100 words)?
      ├─ YES → One-Hot Encoding
      └─ NO → Word Embeddings
```

---

## 10. Advanced Topics

### 1. Document Embeddings: Doc2Vec

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Prepare documents
documents = [
    "machine learning is amazing",
    "deep learning is powerful",
    "natural language processing is useful"
]

# Tag documents
tagged_docs = [TaggedDocument(words=doc.split(), tags=[str(i)]) 
               for i, doc in enumerate(documents)]

# Train Doc2Vec
model = Doc2Vec(
    tagged_docs,
    vector_size=50,
    window=2,
    min_count=1,
    workers=4,
    epochs=100
)

# Get document vector
doc_vector = model.dv['0']
print(f"Document 0 vector: {doc_vector[:10]}")

# Find similar documents
similar_docs = model.dv.most_similar('0')
print(f"Similar to document 0: {similar_docs}")

# Infer vector for new document
new_doc = "machine learning applications"
new_vector = model.infer_vector(new_doc.split())
print(f"New document vector: {new_vector[:10]}")
```

### 2. Sentence Transformers (BERT-based)

```python
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
sentences = [
    "The cat sits on the mat",
    "A feline rests on a rug",
    "Dogs are playing in the park"
]

embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")

# Compute similarities
similarities = util.cos_sim(embeddings, embeddings)
print("\nSentence Similarities:")
for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        if i < j:
            print(f"{i}-{j}: {similarities[i][j]:.4f}")
            print(f"  '{sent1}'")
            print(f"  '{sent2}'")
```

### 3. Contextual Embeddings (BERT)

```python
# pip install transformers

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    """Get BERT embedding for text"""
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', 
                      padding=True, truncation=True)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding
    return outputs.last_hidden_state[:, 0, :].numpy()

# Example: Same word, different contexts
text1 = "I went to the bank to deposit money"
text2 = "I sat on the river bank"

emb1 = get_bert_embedding(text1)
emb2 = get_bert_embedding(text2)

# "bank" has different meanings → different embeddings
from scipy.spatial.distance import cosine
similarity = 1 - cosine(emb1[0], emb2[0])
print(f"Contextual similarity: {similarity:.4f}")
```

### 4. tqdm: Progress Bars

**What is tqdm?**
A library for showing progress bars during loops/iterations.

```python
# pip install tqdm

from tqdm import tqdm
import time

# Basic usage
for i in tqdm(range(100)):
    time.sleep(0.01)

# With description
for i in tqdm(range(100), desc="Processing"):
    time.sleep(0.01)

# For lists
items = ['apple', 'banana', 'cherry'] * 10
for item in tqdm(items, desc="Processing fruits"):
    time.sleep(0.1)

# Manual control
with tqdm(total=100) as pbar:
    for i in range(10):
        time.sleep(0.1)
        pbar.update(10)  # Update by 10

# For training models
from tqdm import tqdm

epochs = 10
for epoch in tqdm(range(epochs), desc="Training"):
    # Training code
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        # Process batch
        pass
```

### 5. Building a Recommendation System

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Sample articles
articles = {
    'id': [1, 2, 3, 4, 5],
    'title': [
        'Introduction to Machine Learning',
        'Deep Learning Basics',
        'Machine Learning Algorithms',
        'Cooking Italian Pasta',
        'Neural Networks Explained'
    ],
    'content': [
        'Machine learning is a subset of AI that learns from data',
        'Deep learning uses neural networks with multiple layers',
        'Various algorithms exist for machine learning tasks',
        'How to cook perfect Italian pasta at home',
        'Neural networks are inspired by the human brain'
    ]
}

df = pd.DataFrame(articles)

# Create TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(article_id, top_n=3):
    """Get similar articles"""
    idx = df[df['id'] == article_id].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N (excluding itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get article indices
    article_indices = [i[0] for i in sim_scores]
    
    # Return recommendations
    return df.iloc[article_indices][['id', 'title']]

# Test
print("Recommendations for 'Introduction to Machine Learning':")
print(get_recommendations(1))

print("\nRecommendations for 'Cooking Italian Pasta':")
print(get_recommendations(4))
```

### 6. Deep-ML Website

**What is deep-ml.com?**
- A platform for practicing ML/DL interview questions
- Focus on coding implementations
- Problems covering:
  - Linear algebra
  - Neural networks
  - Optimization algorithms
  - ML algorithms from scratch
  - Data preprocessing
  - Model evaluation

**Topics Covered:**
1. Matrix operations
2. Activation functions
3. Loss functions
4. Backpropagation
5. Optimization algorithms (SGD, Adam, etc.)
6. Data transformations
7. Feature engineering
8. Model architectures

---

## 11. Interview Questions

### Beginner Level

**Q1: What is the difference between One-Hot encoding and Word Embeddings?**

**A:** 
- **One-Hot:** 
  - Vector size = vocabulary size
  - Very sparse (99%+ zeros)
  - No semantic meaning
  - Each word is equally different from others
  
- **Word Embeddings:**
  - Fixed size (50-300 dimensions)
  - Dense (all non-zero values)
  - Captures semantic similarity
  - Similar words have similar vectors

```python
# One-Hot: "cat" and "dog" are equally different
cat = [1, 0, 0, 0]
dog = [0, 1, 0, 0]
apple = [0, 0, 1, 0]
# similarity(cat, dog) = 0
# similarity(cat, apple) = 0

# Word Embeddings: "cat" and "dog" are similar
cat = [0.5, 0.3, 0.8, ...]
dog = [0.6, 0.4, 0.7, ...]
apple = [-0.1, -0.5, 0.2, ...]
# similarity(cat, dog) = 0.9
# similarity(cat, apple) = 0.1
```

---

**Q2: Why is TF-IDF better than Bag of Words?**

**A:** TF-IDF reduces the importance of common words and highlights important words.

Example:
```
Document: "the cat sat on the mat"

BoW: [2, 1, 1, 1, 1]  # "the" has high count
TF-IDF: [0.1, 0.5, 0.5, 0.5, 0.5]  # "the" downweighted
```

Benefits:
1. Common words (like "the") get low scores
2. Rare, important words get high scores
3. Better for information retrieval
4. Normalizes document length

---

**Q3: What are bigrams and why use them?**

**A:** Bigrams are pairs of consecutive words. They capture local context.

Example:
```
"not good" → bigram: "not good"
"good not" → bigram: "good not"

Without bigrams: Both look identical
With bigrams: Different representations!
```

Use cases:
- Phrase detection ("New York", "machine learning")
- Sentiment analysis ("not good" vs "very good")
- Better context understanding

---

**Q4: When should I use Euclidean distance vs Cosine similarity?**

**A:** 

**Use Euclidean when:**
- Magnitude matters (e.g., coordinates, measurements)
- Clustering similar-sized objects
- Distance in physical space

**Use Cosine when:**
- Direction/angle matters more than magnitude
- Text similarity (TF-IDF, word embeddings)
- Recommendation systems
- Document comparison

Example:
```python
v1 = [1, 1]      # Small magnitude
v2 = [10, 10]    # Large magnitude, same direction

Euclidean: Large distance (magnitude difference)
Cosine: Similarity = 1.0 (same direction)
```

---

**Q5: What is the main drawback of One-Hot encoding?**

**A:** 

1. **No semantic similarity:**
   - "king" and "queen" are treated as equally different as "king" and "apple"

2. **Extremely sparse:**
   - For 10,000-word vocabulary, 99.99% of values are zero

3. **High dimensionality:**
   - Vector size = vocabulary size (can be 10,000+)

4. **Memory inefficient:**
   - Storing mostly zeros wastes memory

5. **Cannot handle OOV:**
   - New words cannot be encoded

---

### Intermediate Level

**Q6: Compare BoW, TF-IDF, and Word2Vec for a classification task.**

**A:**

| Aspect | BoW | TF-IDF | Word2Vec |
|--------|-----|--------|----------|
| **Semantic** | ❌ | ❌ | ✅ |
| **Dimensionality** | High | High | Low |
| **Sparsity** | High | High | Low |
| **Performance** | Baseline | Better | Best |
| **Training** | Fast | Fast | Slower |
| **Interpretability** | High | Medium | Low |

**When to use:**
- **BoW:** Simple baseline, interpretable features
- **TF-IDF:** Information retrieval, search
- **Word2Vec:** When semantic similarity matters, advanced NLP

---

**Q7: How do you handle Out-of-// filepath: NLP/Text_Representation_Complete_Guide.md
# Text Representation in NLP: Complete Guide from Scratch to Advanced

A comprehensive guide covering all text representation techniques, from basic preprocessing to advanced embeddings, with examples, comparisons, and interview questions.

---

## Table of Contents

1. [Text Preprocessing Fundamentals](#1-text-preprocessing-fundamentals)
2. [Text Representation Types Overview](#2-text-representation-types-overview)
3. [One-Hot Encoding for Text](#3-one-hot-encoding-for-text)
4. [Bag of Words (BoW)](#4-bag-of-words-bow)
5. [TF-IDF (Term Frequency-Inverse Document Frequency)](#5-tf-idf)
6. [N-grams (Bigrams, Trigrams)](#6-n-grams)
7. [Word Embeddings](#7-word-embeddings)
8. [Similarity Measures](#8-similarity-measures)
9. [Complete Comparison](#9-complete-comparison)
10. [Advanced Topics](#10-advanced-topics)
11. [Interview Questions](#11-interview-questions)
12. [Practical Applications](#12-practical-applications)

---

## 1. Text Preprocessing Fundamentals

### Why Preprocessing?

Raw text contains noise, inconsistencies, and variations that make it difficult for machines to understand. Preprocessing standardizes text for better model performance.

### Contractions Handling

**What are contractions?**
- Shortened forms of words: "don't" → "do not", "I'm" → "I am"

**Why expand them?**
- Reduces vocabulary size
- Maintains semantic meaning
- Improves model consistency

```python
import contractions

text = "I don't think we're ready for this"
expanded = contractions.fix(text)
print(expanded)
# Output: "I do not think we are ready for this"
```

**Using contractions library:**

```python
# Installation
# pip install contractions

import contractions

# Basic usage
text = "I'll be there, won't you?"
expanded = contractions.fix(text)
print(expanded)  # "I will be there, will not you?"

# Custom handling
def expand_contractions(text):
    """Expand contractions in text"""
    return contractions.fix(text)

# Example
sentences = [
    "I'm learning NLP",
    "They've completed the course",
    "She'd rather stay home"
]

for sent in sentences:
    print(f"Original: {sent}")
    print(f"Expanded: {expand_contractions(sent)}\n")
```

### Complete Preprocessing Pipeline

```python
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import contractions

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Complete preprocessing pipeline"""
        # 1. Lowercase
        text = text.lower()
        
        # 2. Expand contractions
        text = contractions.fix(text)
        
        # 3. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # 4. Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # 5. Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # 6. Remove numbers (optional)
        text = re.sub(r'\d+', '', text)
        
        # 7. Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 8. Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text):
        """Tokenize text"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem(self, tokens):
        """Apply stemming"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize(self, tokens):
        """Apply lemmatization"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text, remove_stopwords=True, use_lemmatization=True):
        """Full preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize or stem
        if use_lemmatization:
            tokens = self.lemmatize(tokens)
        else:
            tokens = self.stem(tokens)
        
        return tokens

# Usage
preprocessor = TextPreprocessor()

text = """
I'm learning NLP! Check out https://example.com 
#MachineLearning @AIExpert - it's amazing!!!
Contact: test@email.com for info. 
"""

# Clean only
cleaned = preprocessor.clean_text(text)
print(f"Cleaned: {cleaned}")

# Full preprocessing
tokens = preprocessor.preprocess(text)
print(f"Tokens: {tokens}")
```

---

## 2. Text Representation Types Overview

### Two Main Categories

#### 1. **Sparse Representation**
- Most values are zero
- High dimensionality
- Examples: One-Hot, BoW, TF-IDF

#### 2. **Dense Representation (Semantic)**
- Low dimensionality (typically 50-300 dimensions)
- Captures semantic meaning
- Examples: Word2Vec, GloVe, FastText, BERT

### Comparison Table

| Type | Dimensionality | Semantic Info | Sparsity | Use Case |
|------|---------------|---------------|----------|----------|
| One-Hot | Vocabulary size | No | Very High | Small vocab |
| BoW | Vocabulary size | No | High | Document classification |
| TF-IDF | Vocabulary size | No | High | Information retrieval |
| Word2Vec | 100-300 | Yes | Low | Similarity tasks |
| BERT | 768 | Yes (contextual) | Low | Advanced NLP |

---

## 3. One-Hot Encoding for Text

### What is One-Hot Encoding?

Each word is represented as a vector with:
- Length = vocabulary size
- 1 at the word's index position
- 0 everywhere else

### Visual Example

```
Vocabulary: ["cat", "dog", "bird", "fish"]

"cat"  → [1, 0, 0, 0]
"dog"  → [0, 1, 0, 0]
"bird" → [0, 0, 1, 0]
"fish" → [0, 0, 0, 1]
```

### Why is it Sparse?

**Example:**
- Vocabulary size: 10,000 words
- Each word vector: 10,000 dimensions
- Only 1 value is 1, 9,999 are 0
- Sparsity = 9,999/10,000 = 99.99%

### Implementation

```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Method 1: Manual implementation
def manual_one_hot(words):
    """Create one-hot encoding manually"""
    # Create vocabulary
    vocab = sorted(list(set(words)))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Create one-hot vectors
    one_hot = np.zeros((len(words), len(vocab)))
    for i, word in enumerate(words):
        one_hot[i, word_to_idx[word]] = 1
    
    return one_hot, vocab, word_to_idx

# Example
words = ["cat", "dog", "bird", "cat", "fish"]
vectors, vocab, word_to_idx = manual_one_hot(words)

print(f"Vocabulary: {vocab}")
print(f"\nOne-hot encoding for '{words[0]}':")
print(vectors[0])

# Method 2: Using sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

words = ["cat", "dog", "bird", "cat", "fish"]

# Label encode first
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(words)

# One-hot encode
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print("\nUsing sklearn:")
print(onehot_encoded)
```

### For Sentences

```python
def sentence_to_onehot(sentence, vocab):
    """Convert sentence to one-hot matrix"""
    words = sentence.lower().split()
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}
    
    # Create matrix
    matrix = np.zeros((len(words), len(vocab)))
    
    for i, word in enumerate(words):
        if word in vocab_dict:
            matrix[i, vocab_dict[word]] = 1
    
    return matrix

# Example
vocab = ["the", "cat", "sat", "on", "mat", "dog"]
sentence = "the cat sat on the mat"

onehot_matrix = sentence_to_onehot(sentence, vocab)
print(f"Sentence: {sentence}")
print(f"Shape: {onehot_matrix.shape}")
print(f"Matrix:\n{onehot_matrix}")
```

### Drawbacks of One-Hot Encoding

1. **No semantic information**
   - "king" and "queen" are equally different as "king" and "apple"
   - Cannot capture word relationships

2. **Extremely sparse**
   - 99%+ zeros for large vocabularies
   - Memory inefficient

3. **High dimensionality**
   - Vector size = vocabulary size
   - Can be 10,000+ dimensions

4. **No word order**
   - "dog bites man" vs "man bites dog" look identical

5. **Out-of-vocabulary (OOV) problem**
   - Cannot handle new words

```python
# Demonstrating no semantic similarity
def cosine_similarity(v1, v2):
    """Calculate cosine similarity"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# One-hot vectors
vocab = ["king", "queen", "man", "woman", "apple", "orange"]
king = [1, 0, 0, 0, 0, 0]
queen = [0, 1, 0, 0, 0, 0]
apple = [0, 0, 0, 0, 1, 0]

print(f"Similarity(king, queen): {cosine_similarity(king, queen)}")  # 0.0
print(f"Similarity(king, apple): {cosine_similarity(king, apple)}")  # 0.0
# Both are 0! No semantic understanding
```

---

## 4. Bag of Words (BoW)

### What is Bag of Words?

A document is represented as a vector where each position corresponds to a word in the vocabulary, and the value is the **count** of that word in the document.

### Key Characteristics

- **Ignores word order** ("bag" of words)
- **Counts word frequency**
- **Fixed vocabulary**

### Visual Example

```
Vocabulary: ["the", "cat", "sat", "on", "mat", "dog"]

Document 1: "the cat sat on the mat"
BoW: [2, 1, 1, 1, 1, 0]
      ↑  ↑  ↑  ↑  ↑  ↑
     the cat sat on mat dog

Document 2: "the dog sat on the mat"
BoW: [2, 0, 1, 1, 1, 1]
```

### Implementation

```python
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Sample documents
documents = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat and the dog",
    "cat sat on mat"
]

# Create BoW using sklearn
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

# View vocabulary
vocab = vectorizer.get_feature_names_out()
print(f"Vocabulary: {vocab}")

# View BoW matrix
print(f"\nBoW Matrix (sparse):\n{bow_matrix}")
print(f"\nBoW Matrix (dense):\n{bow_matrix.toarray()}")

# Analyze specific document
doc_idx = 0
print(f"\nDocument: '{documents[doc_idx]}'")
print(f"BoW vector: {bow_matrix[doc_idx].toarray()[0]}")

# Word counts
for word, count in zip(vocab, bow_matrix[doc_idx].toarray()[0]):
    if count > 0:
        print(f"  {word}: {count}")
```

### Manual Implementation

```python
from collections import Counter

class SimpleBagOfWords:
    def __init__(self):
        self.vocabulary = []
        self.vocab_dict = {}
    
    def fit(self, documents):
        """Build vocabulary from documents"""
        # Tokenize and collect all words
        all_words = []
        for doc in documents:
            words = doc.lower().split()
            all_words.extend(words)
        
        # Create vocabulary
        self.vocabulary = sorted(list(set(all_words)))
        self.vocab_dict = {word: idx for idx, word in enumerate(self.vocabulary)}
        
        return self
    
    def transform(self, documents):
        """Transform documents to BoW vectors"""
        bow_matrix = np.zeros((len(documents), len(self.vocabulary)))
        
        for doc_idx, doc in enumerate(documents):
            words = doc.lower().split()
            word_counts = Counter(words)
            
            for word, count in word_counts.items():
                if word in self.vocab_dict:
                    bow_matrix[doc_idx, self.vocab_dict[word]] = count
        
        return bow_matrix
    
    def fit_transform(self, documents):
        """Fit and transform in one step"""
        return self.fit(documents).transform(documents)

# Usage
bow = SimpleBagOfWords()
documents = [
    "the cat sat on the mat",
    "the dog sat on the mat"
]

bow_matrix = bow.fit_transform(documents)
print(f"Vocabulary: {bow.vocabulary}")
print(f"BoW Matrix:\n{bow_matrix}")
```

### Advanced CountVectorizer Options

```python
from sklearn.feature_extraction.text import CountVectorizer

# With various parameters
vectorizer = CountVectorizer(
    max_features=1000,        # Limit vocabulary size
    min_df=2,                 # Ignore words appearing in < 2 documents
    max_df=0.8,               # Ignore words appearing in > 80% documents
    stop_words='english',     # Remove English stop words
    ngram_range=(1, 2),       # Include unigrams and bigrams
    lowercase=True,           # Convert to lowercase
    token_pattern=r'\b\w+\b'  # Token extraction pattern
)

documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are pets",
    "The cat is a pet"
]

bow_matrix = vectorizer.fit_transform(documents)
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Features: {vectorizer.get_feature_names_out()}")
print(f"Shape: {bow_matrix.shape}")
```

### Drawbacks of BoW

1. **Loss of word order**
   ```python
   # These two sentences produce same BoW
   "dog bites man" → [1, 1, 1]
   "man bites dog" → [1, 1, 1]
   ```

2. **High dimensionality**
   - Vocabulary can be 10,000+ words

3. **Sparse representation**
   - Most values are zero

4. **No semantic meaning**
   - Cannot capture synonyms or related words

5. **Common words dominate**
   - "the", "is", "and" have high counts but low importance

---

## 5. TF-IDF

### What is TF-IDF?

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a numerical statistic that reflects how important a word is to a document in a collection of documents.

### Components

#### 1. **TF (Term Frequency)**

**What:** How often a word appears in a document.

**Formula:**
```
TF(word, document) = (Number of times word appears in document) / (Total words in document)
```

**Example:**
```
Document: "the cat sat on the mat"
Total words: 6

TF("the") = 2/6 = 0.333
TF("cat") = 1/6 = 0.167
TF("sat") = 1/6 = 0.167
```

#### 2. **IDF (Inverse Document Frequency)**

**What:** How rare or important a word is across all documents.

**Intuition:**
- Common words (like "the") appear in many documents → low IDF → less important
- Rare words appear in few documents → high IDF → more important

**Formula:**
```
IDF(word) = log((Total number of documents) / (Number of documents containing the word))
```

**Example:**
```
Corpus: 100 documents

Word "the" appears in 100 documents:
IDF("the") = log(100/100) = log(1) = 0

Word "python" appears in 5 documents:
IDF("python") = log(100/5) = log(20) = 2.996

Word "machine" appears in 20 documents:
IDF("machine") = log(100/20) = log(5) = 1.609
```

#### 3. **TF-IDF (Together)**

**Formula:**
```
TF-IDF(word, document) = TF(word, document) × IDF(word)
```

**Interpretation:**
- High TF-IDF: Word is frequent in this document but rare overall → **important**
- Low TF-IDF: Word is either rare in this document or common everywhere → **less important**

### Step-by-Step Example

```
Corpus of 3 documents:

Doc 1: "the cat sat on the mat"
Doc 2: "the dog sat on the log"
Doc 3: "cats and dogs are pets"

Calculate TF-IDF for word "cat" in Doc 1:

Step 1: Calculate TF
TF("cat", Doc1) = 1/6 = 0.167

Step 2: Calculate IDF
Documents containing "cat": Doc 1 only = 1
Total documents: 3
IDF("cat") = log(3/1) = log(3) = 1.099

Step 3: Calculate TF-IDF
TF-IDF("cat", Doc1) = 0.167 × 1.099 = 0.183
```

### Implementation from Scratch

```python
import numpy as np
import math
from collections import Counter

class TFIDFFromScratch:
    def __init__(self):
        self.vocabulary = []
        self.idf_values = {}
        self.documents = []
    
    def fit(self, documents):
        """Calculate IDF values"""
        self.documents = documents
        n_documents = len(documents)
        
        # Build vocabulary
        all_words = set()
        for doc in documents:
            words = doc.lower().split()
            all_words.update(words)
        
        self.vocabulary = sorted(list(all_words))
        
        # Calculate IDF for each word
        for word in self.vocabulary:
            # Count documents containing this word
            doc_count = sum(1 for doc in documents if word in doc.lower().split())
            
            # Calculate IDF
            self.idf_values[word] = math.log(n_documents / doc_count)
        
        return self
    
    def transform(self, documents):
        """Calculate TF-IDF vectors"""
        tfidf_matrix = []
        
        for doc in documents:
            words = doc.lower().split()
            word_counts = Counter(words)
            total_words = len(words)
            
            # Calculate TF-IDF for each word in vocabulary
            tfidf_vector = []
            for word in self.vocabulary:
                # TF
                tf = word_counts.get(word, 0) / total_words
                
                # IDF
                idf = self.idf_values.get(word, 0)
                
                # TF-IDF
                tfidf = tf * idf
                tfidf_vector.append(tfidf)
            
            tfidf_matrix.append(tfidf_vector)
        
        return np.array(tfidf_matrix)
    
    def fit_transform(self, documents):
        """Fit and transform in one step"""
        return self.fit(documents).transform(documents)

# Example usage
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

tfidf = TFIDFFromScratch()
tfidf_matrix = tfidf.fit_transform(documents)

print("Vocabulary:", tfidf.vocabulary)
print("\nIDF values:")
for word, idf in sorted(tfidf.idf_values.items(), key=lambda x: -x[1])[:5]:
    print(f"  {word}: {idf:.4f}")

print(f"\nTF-IDF Matrix shape: {tfidf_matrix.shape}")
print(f"TF-IDF Matrix:\n{tfidf_matrix}")
```

### Using sklearn's TfidfVectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets",
    "the cat is a pet"
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=100,
    stop_words='english',
    ngram_range=(1, 2)
)

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(documents)

# View results
vocab = vectorizer.get_feature_names_out()
print(f"Vocabulary: {vocab}")
print(f"\nTF-IDF Matrix:\n{tfidf_matrix.toarray()}")

# Analyze specific document
doc_idx = 0
feature_scores = list(zip(vocab, tfidf_matrix[doc_idx].toarray()[0]))
feature_scores = sorted(feature_scores, key=lambda x: -x[1])

print(f"\nTop features for document {doc_idx}:")
for word, score in feature_scores[:5]:
    print(f"  {word}: {score:.4f}")
```

### TF-IDF Variants

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Different TF-IDF configurations

# 1. Binary TF (presence/absence)
vectorizer_binary = TfidfVectorizer(binary=True)

# 2. Sublinear TF (use log scale for TF)
vectorizer_sublinear = TfidfVectorizer(sublinear_tf=True)

# 3. Custom norm
vectorizer_norm = TfidfVectorizer(norm='l1')  # L1 normalization

# 4. Smooth IDF
vectorizer_smooth = TfidfVectorizer(smooth_idf=True)  # Add 1 to document frequencies
```

### Advantages of TF-IDF over BoW

1. **Reduces importance of common words**
   ```python
   # "the" appears everywhere → low TF-IDF
   # "python" appears rarely → high TF-IDF
   ```

2. **Highlights important words**
   - Words unique to a document get higher scores

3. **Better for information retrieval**
   - Search engines use TF-IDF-like measures

4. **Normalized representation**
   - Documents of different lengths comparable

### Drawbacks of TF-IDF

1. **Still sparse**
   - Same dimensionality as BoW

2. **No semantic understanding**
   - "happy" and "joyful" treated as completely different

3. **Loses word order**
   - "not good" vs "good not" identical

4. **Fixed vocabulary**
   - Cannot handle new words

```python
# Demonstrating limitations
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "The movie was not good",
    "The movie was good"
]

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(docs)

# These should have opposite meanings but might look similar
print("TF-IDF vectors:")
print(tfidf.toarray())
```

---

## 6. N-grams

### What are N-grams?

**N-gram:** A contiguous sequence of N items (words) from a text.

- **Unigram (1-gram):** Single words → "cat", "sat", "mat"
- **Bigram (2-gram):** Two consecutive words → "the cat", "cat sat"
- **Trigram (3-gram):** Three consecutive words → "the cat sat"

### Why N-grams?

**Problem with Bag of Words:**
```
"not good" and "good not" → same representation
```

**Solution: Use bigrams**
```
"not good" → contains bigram "not good"
"good not" → contains bigram "good not"
→ Different representations!
```

### Bigram Examples

```
Sentence: "the cat sat on the mat"

Bigrams:
1. "the cat"
2. "cat sat"
3. "sat on"
4. "on the"
5. "the mat"
```

### Visual Example

```
Sentence: "I love machine learning"

Unigrams: ["I", "love", "machine", "learning"]

Bigrams: ["I love", "love machine", "machine learning"]

Trigrams: ["I love machine", "love machine learning"]
```

### Implementation

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "machine learning is great",
    "deep learning is powerful",
    "machine learning and deep learning"
]

# Unigrams only
vectorizer_unigram = CountVectorizer(ngram_range=(1, 1))
unigram_matrix = vectorizer_unigram.fit_transform(documents)
print("Unigrams:", vectorizer_unigram.get_feature_names_out())

# Bigrams only
vectorizer_bigram = CountVectorizer(ngram_range=(2, 2))
bigram_matrix = vectorizer_bigram.fit_transform(documents)
print("\nBigrams:", vectorizer_bigram.get_feature_names_out())

# Both unigrams and bigrams
vectorizer_both = CountVectorizer(ngram_range=(1, 2))
both_matrix = vectorizer_both.fit_transform(documents)
print("\nUnigrams + Bigrams:", vectorizer_both.get_feature_names_out())
```

### Manual N-gram Generation

```python
def generate_ngrams(text, n):
    """Generate n-grams from text"""
    words = text.split()
    ngrams = []
    
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

# Example
text = "the quick brown fox jumps over the lazy dog"

print("Unigrams:", generate_ngrams(text, 1))
print("Bigrams:", generate_ngrams(text, 2))
print("Trigrams:", generate_ngrams(text, 3))
```

### Character N-grams

```python
def char_ngrams(text, n):
    """Generate character n-grams"""
    return [text[i:i+n] for i in range(len(text) - n + 1)]

# Useful for misspellings and unknown words
word = "learning"
print(f"Character bigrams of '{word}':")
print(char_ngrams(word, 2))

# sklearn implementation
vectorizer_char = CountVectorizer(
    analyzer='char',
    ngram_range=(2, 3)
)

docs = ["learning", "machine", "deep"]
char_matrix = vectorizer_char.fit_transform(docs)
print("\nCharacter n-grams:")
print(vectorizer_char.get_feature_names_out())
```

### TF-IDF with N-grams

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "natural language processing is amazing",
    "machine learning is powerful",
    "deep learning revolutionizes AI"
]

# TF-IDF with bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(documents)

# Show top features
feature_names = vectorizer.get_feature_names_out()
doc_idx = 0

feature_scores = list(zip(feature_names, tfidf_matrix[doc_idx].toarray()[0]))
feature_scores = sorted(feature_scores, key=lambda x: -x[1])

print(f"Top features (including bigrams) for document {doc_idx}:")
for word, score in feature_scores[:10]:
    print(f"  {word}: {score:.4f}")
```

### Advantages of N-grams

1. **Captures word order** (partially)
   ```python
   "not good" ≠ "good not"
   ```

2. **Better context understanding**
   ```python
   "New York" treated as single unit
   "machine learning" recognized as phrase
   ```

3. **Improves classification**
   - Better features for models

### Disadvantages of N-grams

1. **Vocabulary explosion**
   ```python
   # 1,000 words
   # Unigrams: 1,000 features
   # Bigrams: up to 1,000,000 combinations
   # Trigrams: up to 1,000,000,000 combinations
   ```

2. **Increased sparsity**
   - Most bigrams appear only once

3. **Computational cost**
   - More features = more memory and processing

4. **Data sparsity**
   - Need more data to learn meaningful patterns

```python
# Demonstrating vocabulary explosion
from sklearn.feature_extraction.text import CountVectorizer

documents = ["the cat sat on the mat"] * 10

for n in [(1,1), (2,2), (3,3), (1,2), (1,3)]:
    vectorizer = CountVectorizer(ngram_range=n)
    matrix = vectorizer.fit_transform(documents)
    print(f"N-gram range {n}: {len(vectorizer.vocabulary_)} features")
```

---

## 7. Word Embeddings

### What are Word Embeddings?

**Dense vector representations** of words that capture semantic meaning in a low-dimensional space (typically 50-300 dimensions).

### Key Difference from Previous Methods

| Feature | One-Hot/BoW/TF-IDF | Word Embeddings |
|---------|-------------------|-----------------|
| **Dimensionality** | Vocabulary size (10,000+) | Fixed (50-300) |
| **Sparsity** | Very sparse (99%+ zeros) | Dense (all non-zero) |
| **Semantics** | No meaning | Captures meaning |
| **Similarity** | Cannot measure | Can measure |

### Core Idea

```
Words with similar meanings should have similar vectors

"king" ≈ "queen" ≈ "monarch"
"dog" ≈ "cat" ≈ "pet"

"king" ≠ "apple"
```

### Types of Word Embeddings

#### 1. **Word2Vec**

Two architectures:
- **CBOW (Continuous Bag of Words):** Predict target word from context
- **Skip-gram:** Predict context words from target word

#### 2. **GloVe (Global Vectors)**

- Based on word co-occurrence statistics
- Trained on global corpus statistics

#### 3. **FastText**

- Extension of Word2Vec
- Uses character n-grams
- Can handle out-of-vocabulary words

#### 4. **BERT/Transformers**

- Contextual embeddings
- Same word has different vectors in different contexts

### Word2Vec Example

```python
# Installation: pip install gensim

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample corpus
sentences = [
    "king is a male ruler",
    "queen is a female ruler",
    "man is to woman as king is to queen",
    "prince is the son of a king",
    "princess is the daughter of a queen",
    "dog and cat are pets",
    "puppy is a young dog",
    "kitten is a young cat"
]

# Tokenize
tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]

# Train Word2Vec
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,      # Dimension of embeddings
    window=5,             # Context window size
    min_count=1,          # Minimum word frequency
    workers=4,            # Parallel processing
    sg=0                  # 0=CBOW, 1=Skip-gram
)

# Get word vector
king_vector = model.wv['king']
print(f"'king' vector (first 10 dims): {king_vector[:10]}")

# Find similar words
similar_to_king = model.wv.most_similar('king', topn=5)
print(f"\nWords similar to 'king': {similar_to_king}")

# Word arithmetic
# king - man + woman ≈ queen
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
print(f"\nking - man + woman = {result[0][0]}")

# Similarity score
similarity = model.wv.similarity('king', 'queen')
print(f"\nSimilarity(king, queen): {similarity:.4f}")
```

### Using Pre-trained Embeddings

```python
import gensim.downloader as api

# Load pre-trained GloVe embeddings
# Options: 'glove-wiki-gigaword-100', 'word2vec-google-news-300', etc.
embeddings = api.load('glove-wiki-gigaword-100')

# Get vector
word = 'computer'
vector = embeddings[word]
print(f"Vector for '{word}': {vector[:10]}")

# Find similar words
similar_words = embeddings.most_similar('computer', topn=10)
print(f"\nSimilar to 'computer':")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

# Word analogies
result = embeddings.most_similar(
    positive=['woman', 'king'],
    negative=['man'],
    topn=3
)
print(f"\nwoman + king - man:")
for word, score in result:
    print(f"  {word}: {score:.4f}")
```

### Document Embeddings from Word Embeddings

```python
import numpy as np

def document_vector(doc, model):
    """Average word vectors in document"""
    words = doc.lower().split()
    word_vectors = []
    
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])
    
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Example
doc1 = "king and queen rule the kingdom"
doc2 = "dog and cat are common pets"

doc1_vec = document_vector(doc1, model)
doc2_vec = document_vector(doc2, model)

# Compare documents
from scipy.spatial.distance import cosine
similarity = 1 - cosine(doc1_vec, doc2_vec)
print(f"Document similarity: {similarity:.4f}")
```

### Advantages of Word Embeddings

1. **Semantic similarity**
   ```python
   similarity('happy', 'joyful') > similarity('happy', 'sad')
   ```

2. **Low dimensionality**
   - 100-300 dimensions vs 10,000+ for one-hot

3. **Word relationships**
   ```python
   king - man + woman ≈ queen
   ```

4. **Transfer learning**
   - Use pre-trained embeddings

5. **Dense representation**
   - No sparsity issues

### Drawbacks of Word Embeddings

1. **Fixed vocabulary**
   - Cannot handle new words (except FastText)

2. **No context**
   - "bank" (river) vs "bank" (money) same vector

3. **Training time**
   - Need large corpus

4. **Memory**
   - Pre-trained models can be large (1-2 GB)

---

## 8. Similarity Measures

### Why Similarity Measures?

To compare:
- Word vectors
- Document vectors
- Sentences
- User queries with documents

### 1. Euclidean Distance

**Formula:**
```
distance = √[(x₁ - x₂)² + (y₁ - y₂)² + ...]
```

**Intuition:** Straight-line distance in n-dimensional space

**Implementation:**

```python
import numpy as np

def euclidean_distance(v1, v2):
    """Calculate Euclidean distance"""
    return np.linalg.norm(v1 - v2)

# Example
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

distance = euclidean_distance(v1, v2)
print(f"Euclidean distance: {distance:.4f}")

# Using scipy
from scipy.spatial.distance import euclidean
distance = euclidean(v1, v2)
print(f"Using scipy: {distance:.4f}")
```

**When to use:**
- When magnitude matters
- In coordinate systems
- For clustering (K-means)

**Drawback:**
- Sensitive to vector magnitude
- Not scale-invariant

```python
# Problem: Different magnitudes
v1 = np.array([1, 1])      # Length ≈ 1.41
v2 = np.array([10, 10])    # Length ≈ 14.1, same direction!

print(f"Distance: {euclidean_distance(v1, v2):.4f}")  # Large!
# But they point in the same direction
```

### 2. Cosine Similarity

**Formula:**
```
cosine_similarity = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product
- ||A|| = magnitude of A
```

**Range:** -1 to 1
- 1: Identical direction
- 0: Perpendicular
- -1: Opposite direction

**Intuition:** Measures angle between vectors, ignoring magnitude

**Implementation:**

```python
import numpy as np

def cosine_similarity(v1, v2):
    """Calculate cosine similarity"""
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0
    
    return dot_product / (magnitude_v1 * magnitude_v2)

# Example
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # Same direction, different magnitude

cos_sim = cosine_similarity(v1, v2)
print(f"Cosine similarity: {cos_sim:.4f}")  # Should be 1.0

# Using sklearn
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

v1 = v1.reshape(1, -1)
v2 = v2.reshape(1, -1)
similarity = sk_cosine(v1, v2)[0][0]
print(f"Using sklearn: {similarity:.4f}")
```

**Visual Example:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Create vectors
v1 = np.array([3, 4])
v2 = np.array([6, 8])  # Same direction
v3 = np.array([4, 3])  # Different direction

# Calculate similarities
sim_v1_v2 = cosine_similarity(v1, v2)
sim_v1_v3 = cosine_similarity(v1, v3)

print(f"Similarity(v1, v2): {sim_v1_v2:.4f}")  # High (same direction)
print(f"Similarity(v1, v3): {sim_v1_v3:.4f}")  # Lower (different direction)

# Plot
plt.figure(figsize=(8, 8))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
plt.quiver(0, 0, v3[0], v3[1], angles='xy', scale_units='xy', scale=1, color='g', label='v3')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
plt.title('Vector Comparison')
plt.grid(True)
plt.show()
```

**When to use Cosine:**
- Text similarity (TF-IDF vectors)
- Recommendation systems
- Word embeddings
- When direction matters more than magnitude

### Euclidean vs Cosine: Comparison

```python
# Demonstrating difference

# Case 1: Same direction, different magnitude
v1 = np.array([1, 1])
v2 = np.array([10, 10])

print("Case 1: Same direction, different magnitude")
print(f"Euclidean distance: {euclidean_distance(v1, v2):.4f}")  # Large
print(f"Cosine similarity: {cosine_similarity(v1, v2):.4f}")    # 1.0 (perfect)

# Case 2: Different direction, similar magnitude
v1 = np.array([5, 5])
v2 = np.array([5, -5])

print("\nCase 2: Different direction, similar magnitude")
print(f"Euclidean distance: {euclidean_distance(v1, v2):.4f}")  # Moderate
print(f"Cosine similarity: {cosine_similarity(v1, v2):.4f}")    # 0.0 (perpendicular)
```

### Document Similarity Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "machine learning is amazing",
    "deep learning is a subset of machine learning",
    "natural language processing uses machine learning",
    "I love eating pizza"
]

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate similarities
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Document Similarity Matrix:")
print(similarity_matrix)

# Find most similar to document 0
doc_idx = 0
similarities = list(enumerate(similarity_matrix[doc_idx]))
similarities = sorted(similarities, key=lambda x: -x[1])

print(f"\nDocuments similar to: '{documents[doc_idx]}'")
for idx, score in similarities[1:]:  # Skip itself
    print(f"  {documents[idx]}: {score:.4f}")
```

### Other Similarity Measures

```python
from scipy.spatial.distance import *

v1 = np.array([1, 2, 3, 4])
v2 = np.array([2, 3, 4, 5])

print("Different similarity measures:")
print(f"Euclidean: {euclidean(v1, v2):.4f}")
print(f"Cosine: {1 - cosine(v1, v2):.4f}")  # scipy returns distance
print(f"Manhattan: {cityblock(v1, v2):.4f}")
print(f"Chebyshev: {chebyshev(v1, v2):.4f}")
print(f"Jaccard: {jaccard(v1 > 0, v2 > 0):.4f}")  # For binary
```

---

## 9. Complete Comparison

### Comparison Table

| Method | Dimensionality | Semantic | Sparsity | Context | OOV Handling | Use Case |
|--------|---------------|----------|----------|---------|--------------|----------|
| **One-Hot** | Vocab size | ❌ | Very High | ❌ | ❌ | Small vocabs only |
| **BoW** | Vocab size | ❌ | High | ❌ | ❌ | Document classification |
| **TF-IDF** | Vocab size | ❌ | High | ❌ | ❌ | Information retrieval |
| **N-grams** | Vocab size × N | Partial | Very High | Partial | ❌ | Phrase detection |
| **Word2Vec** | 100-300 | ✅ | Low | ❌ | ⚠️ (FastText) | Similarity tasks |
| **GloVe** | 100-300 | ✅ | Low | ❌ | ❌ | General embeddings |
| **BERT** | 768 | ✅ | Low | ✅ | ✅ | State-of-the-art NLP |

### Detailed Comparison

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Sample documents
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]

# 1. One-Hot / BoW
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(documents)
print(f"BoW shape: {bow_matrix.shape}")
print(f"BoW sparsity: {1 - (bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])):.2%}")

# 2. TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print(f"\nTF-IDF shape: {tfidf_matrix.shape}")
print(f"TF-IDF sparsity: {1 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.2%}")

# 3. Word Embeddings (average)
tokenized = [doc.split() for doc in documents]
w2v_model = Word2Vec(tokenized, vector_size=50, min_count=1)

def doc_to_vec(doc, model):
    words = doc.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

w2v_matrix = np.array([doc_to_vec(doc, w2v_model) for doc in documents])
print(f"\nWord2Vec shape: {w2v_matrix.shape}")
print(f"Word2Vec sparsity: {(w2v_matrix == 0).sum() / w2v_matrix.size:.2%}")
```

### Drawbacks Summary

#### One-Hot Encoding
1. ❌ No semantic similarity
2. ❌ Extremely sparse (99%+ zeros)
3. ❌ High dimensionality = vocabulary size
4. ❌ Cannot handle OOV words
5. ❌ All words equally different

#### Bag of Words
1. ❌ Loses word order
2. ❌ High dimensionality
3. ❌ Sparse representation
4. ❌ No semantic meaning
5. ❌ Common words dominate

#### TF-IDF
1. ❌ Still sparse
2. ❌ No semantic understanding
3. ❌ Loses word order
4. ❌ Fixed vocabulary (OOV problem)
5. ⚠️ Better than BoW but not semantic

#### N-grams
1. ❌ Vocabulary explosion (exponential growth)
2. ❌ Increased sparsity
3. ❌ Computational cost
4. ❌ Needs more training data
5. ⚠️ Partial context only

#### Word Embeddings (Word2Vec/GloVe)
1. ❌ Fixed vocabulary (except FastText)
2. ❌ No context (same vector for "bank" river vs money)
3. ❌ Training requires large corpus
4. ❌ Memory intensive for pre-trained models
5. ⚠️ Best among traditional methods

### Decision Tree: Which to Use?

```
Start
  |
  ├─ Need semantic similarity?
  │   ├─ YES → Word Embeddings (Word2Vec, GloVe)
  │   │         or BERT (for contextual)
  │   └─ NO ↓
  │
  ├─ Need to handle phrases?
  │   ├─ YES → N-grams (with BoW or TF-IDF)
  │   └─ NO ↓
  │
  ├─ Information retrieval / search?
  │   ├─ YES → TF-IDF
  │   └─ NO ↓
  │
  ├─ Simple document classification?
  │   ├─ YES → BoW or TF-IDF
  │   └─ NO ↓
  │
  └─ Very small vocabulary (<100 words)?
      ├─ YES → One-Hot Encoding
      └─ NO → Word Embeddings
```

---

## 10. Advanced Topics

### 1. Document Embeddings: Doc2Vec

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Prepare documents
documents = [
    "machine learning is amazing",
    "deep learning is powerful",
    "natural language processing is useful"
]

# Tag documents
tagged_docs = [TaggedDocument(words=doc.split(), tags=[str(i)]) 
               for i, doc in enumerate(documents)]

# Train Doc2Vec
model = Doc2Vec(
    tagged_docs,
    vector_size=50,
    window=2,
    min_count=1,
    workers=4,
    epochs=100
)

# Get document vector
doc_vector = model.dv['0']
print(f"Document 0 vector: {doc_vector[:10]}")

# Find similar documents
similar_docs = model.dv.most_similar('0')
print(f"Similar to document 0: {similar_docs}")

# Infer vector for new document
new_doc = "machine learning applications"
new_vector = model.infer_vector(new_doc.split())
print(f"New document vector: {new_vector[:10]}")
```

### 2. Sentence Transformers (BERT-based)

```python
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
sentences = [
    "The cat sits on the mat",
    "A feline rests on a rug",
    "Dogs are playing in the park"
]

embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")

# Compute similarities
similarities = util.cos_sim(embeddings, embeddings)
print("\nSentence Similarities:")
for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        if i < j:
            print(f"{i}-{j}: {similarities[i][j]:.4f}")
            print(f"  '{sent1}'")
            print(f"  '{sent2}'")
```

### 3. Contextual Embeddings (BERT)

```python
# pip install transformers

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    """Get BERT embedding for text"""
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', 
                      padding=True, truncation=True)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding
    return outputs.last_hidden_state[:, 0, :].numpy()

# Example: Same word, different contexts
text1 = "I went to the bank to deposit money"
text2 = "I sat on the river bank"

emb1 = get_bert_embedding(text1)
emb2 = get_bert_embedding(text2)

# "bank" has different meanings → different embeddings
from scipy.spatial.distance import cosine
similarity = 1 - cosine(emb1[0], emb2[0])
print(f"Contextual similarity: {similarity:.4f}")
```

### 4. tqdm: Progress Bars

**What is tqdm?**
A library for showing progress bars during loops/iterations.

```python
# pip install tqdm

from tqdm import tqdm
import time

# Basic usage
for i in tqdm(range(100)):
    time.sleep(0.01)

# With description
for i in tqdm(range(100), desc="Processing"):
    time.sleep(0.01)

# For lists
items = ['apple', 'banana', 'cherry'] * 10
for item in tqdm(items, desc="Processing fruits"):
    time.sleep(0.1)

# Manual control
with tqdm(total=100) as pbar:
    for i in range(10):
        time.sleep(0.1)
        pbar.update(10)  # Update by 10

# For training models
from tqdm import tqdm

epochs = 10
for epoch in tqdm(range(epochs), desc="Training"):
    # Training code
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        # Process batch
        pass
```

### 5. Building a Recommendation System

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Sample articles
articles = {
    'id': [1, 2, 3, 4, 5],
    'title': [
        'Introduction to Machine Learning',
        'Deep Learning Basics',
        'Machine Learning Algorithms',
        'Cooking Italian Pasta',
        'Neural Networks Explained'
    ],
    'content': [
        'Machine learning is a subset of AI that learns from data',
        'Deep learning uses neural networks with multiple layers',
        'Various algorithms exist for machine learning tasks',
        'How to cook perfect Italian pasta at home',
        'Neural networks are inspired by the human brain'
    ]
}

df = pd.DataFrame(articles)

# Create TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(article_id, top_n=3):
    """Get similar articles"""
    idx = df[df['id'] == article_id].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N (excluding itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get article indices
    article_indices = [i[0] for i in sim_scores]
    
    # Return recommendations
    return df.iloc[article_indices][['id', 'title']]

# Test
print("Recommendations for 'Introduction to Machine Learning':")
print(get_recommendations(1))

print("\nRecommendations for 'Cooking Italian Pasta':")
print(get_recommendations(4))
```

### 6. Deep-ML Website

**What is deep-ml.com?**
- A platform for practicing ML/DL interview questions
- Focus on coding implementations
- Problems covering:
  - Linear algebra
  - Neural networks
  - Optimization algorithms
  - ML algorithms from scratch
  - Data preprocessing
  - Model evaluation

**Topics Covered:**
1. Matrix operations
2. Activation functions
3. Loss functions
4. Backpropagation
5. Optimization algorithms (SGD, Adam, etc.)
6. Data transformations
7. Feature engineering
8. Model architectures

---

## 11. Interview Questions

### Beginner Level

**Q1: What is the difference between One-Hot encoding and Word Embeddings?**

**A:** 
- **One-Hot:** 
  - Vector size = vocabulary size
  - Very sparse (99%+ zeros)
  - No semantic meaning
  - Each word is equally different from others
  
- **Word Embeddings:**
  - Fixed size (50-300 dimensions)
  - Dense (all non-zero values)
  - Captures semantic similarity
  - Similar words have similar vectors

```python
# One-Hot: "cat" and "dog" are equally different
cat = [1, 0, 0, 0]
dog = [0, 1, 0, 0]
apple = [0, 0, 1, 0]
# similarity(cat, dog) = 0
# similarity(cat, apple) = 0

# Word Embeddings: "cat" and "dog" are similar
cat = [0.5, 0.3, 0.8, ...]
dog = [0.6, 0.4, 0.7, ...]
apple = [-0.1, -0.5, 0.2, ...]
# similarity(cat, dog) = 0.9
# similarity(cat, apple) = 0.1
```

---

**Q2: Why is TF-IDF better than Bag of Words?**

**A:** TF-IDF reduces the importance of common words and highlights important words.

Example:
```
Document: "the cat sat on the mat"

BoW: [2, 1, 1, 1, 1]  # "the" has high count
TF-IDF: [0.1, 0.5, 0.5, 0.5, 0.5]  # "the" downweighted
```

Benefits:
1. Common words (like "the") get low scores
2. Rare, important words get high scores
3. Better for information retrieval
4. Normalizes document length

---

**Q3: What are bigrams and why use them?**

**A:** Bigrams are pairs of consecutive words. They capture local context.

Example:
```
"not good" → bigram: "not good"
"good not" → bigram: "good not"

Without bigrams: Both look identical
With bigrams: Different representations!
```

Use cases:
- Phrase detection ("New York", "machine learning")
- Sentiment analysis ("not good" vs "very good")
- Better context understanding

---

**Q4: When should I use Euclidean distance vs Cosine similarity?**

**A:** 

**Use Euclidean when:**
- Magnitude matters (e.g., coordinates, measurements)
- Clustering similar-sized objects
- Distance in physical space

**Use Cosine when:**
- Direction/angle matters more than magnitude
- Text similarity (TF-IDF, word embeddings)
- Recommendation systems
- Document comparison

Example:
```python
v1 = [1, 1]      # Small magnitude
v2 = [10, 10]    # Large magnitude, same direction

Euclidean: Large distance (magnitude difference)
Cosine: Similarity = 1.0 (same direction)
```

---

**Q5: What is the main drawback of One-Hot encoding?**

**A:** 

1. **No semantic similarity:**
   - "king" and "queen" are treated as equally different as "king" and "apple"

2. **Extremely sparse:**
   - For 10,000-word vocabulary, 99.99% of values are zero

3. **High dimensionality:**
   - Vector size = vocabulary size (can be 10,000+)

4. **Memory inefficient:**
   - Storing mostly zeros wastes memory

5. **Cannot handle OOV:**
   - New words cannot be encoded

---

### Intermediate Level

**Q6: Compare BoW, TF-IDF, and Word2Vec for a classification task.**

**A:**

| Aspect | BoW | TF-IDF | Word2Vec |
|--------|-----|--------|----------|
| **Semantic** | ❌ | ❌ | ✅ |
| **Dimensionality** | High | High | Low |
| **Sparsity** | High | High | Low |
| **Performance** | Baseline | Better | Best |
| **Training** | Fast | Fast | Slower |
| **Interpretability** | High | Medium | Low |

**When to use:**
- **BoW:** Simple baseline, interpretable features
- **TF-IDF:** Information retrieval, search
- **Word2Vec:** When semantic similarity matters, advanced NLP

---
**Q7: How do you handle Out-of-Vocabulary (OOV) words in different text representation methods?**

**A:**

| Method | OOV Handling | Approach |
|--------|--------------|----------|
| **One-Hot** | ❌ Cannot handle | Ignore or use <UNK> token |
| **BoW/TF-IDF** | ❌ Cannot handle | Ignore unknown words |
| **Word2Vec** | ❌ Cannot handle | Use average of known words or <UNK> |
| **FastText** | ✅ Can handle | Uses character n-grams |
| **BERT** | ✅ Can handle | Subword tokenization (WordPiece) |

**Solutions:**

```python
# 1. Using FastText (handles OOV)
from gensim.models import FastText

sentences = [["machine", "learning"], ["deep", "learning"]]
model = FastText(sentences, vector_size=100, window=3, min_count=1)

# Can get vectors for unseen words!
vector = model.wv['machinelearning']  # Even if not in training
print(f"OOV word vector: {vector[:5]}")

# 2. Fallback strategies for Word2Vec
def get_word_vector_with_fallback(word, model):
    """Handle OOV with fallback"""
    if word in model.wv:
        return model.wv[word]
    
    # Strategy 1: Return zero vector
    # return np.zeros(model.vector_size)
    
    # Strategy 2: Return random vector
    # return np.random.randn(model.vector_size)
    
    # Strategy 3: Use most similar known word
    # Find closest word by edit distance
    import difflib
    known_words = list(model.wv.key_to_index.keys())
    similar = difflib.get_close_matches(word, known_words, n=1)
    if similar:
        return model.wv[similar[0]]
    
    return np.zeros(model.vector_size)

# 3. Using BERT tokenizer (handles OOV automatically)
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# BERT breaks unknown words into subwords
text = "machinelearningisawesome"  # Made-up word
tokens = tokenizer.tokenize(text)
print(f"Subword tokens: {tokens}")
# Output: ['machine', '##learning', '##is', '##awesome']
```

---

**Q8: Explain the difference between CBOW and Skip-gram in Word2Vec.**

**A:**

**CBOW (Continuous Bag of Words):**
- Predicts target word from context words
- Faster to train
- Better for frequent words
- Smooths over distributional information

**Skip-gram:**
- Predicts context words from target word
- Slower to train
- Better for rare words
- Better at capturing rare word relationships

**Visual Example:**
```
Sentence: "The cat sat on the mat"
Window size: 2

CBOW:
Input: [the, cat, on, the] → Output: "sat"
(context) → (target)

Skip-gram:
Input: "sat" → Output: [the, cat, on, the]
(target) → (context)
```

**Implementation:**

```python
from gensim.models import Word2Vec

sentences = [
    "the cat sat on the mat".split(),
    "the dog sat on the log".split()
]

# CBOW
cbow_model = Word2Vec(
    sentences,
    vector_size=100,
    window=2,
    min_count=1,
    sg=0  # CBOW
)

# Skip-gram
skipgram_model = Word2Vec(
    sentences,
    vector_size=100,
    window=2,
    min_count=1,
    sg=1  # Skip-gram
)

# Compare
word = "sat"
print(f"CBOW similar: {cbow_model.wv.most_similar(word, topn=3)}")
print(f"Skip-gram similar: {skipgram_model.wv.most_similar(word, topn=3)}")
```

**When to use:**
- **CBOW:** Large dataset, frequent words, faster training needed
- **Skip-gram:** Small dataset, rare words important, semantic quality priority

---

### Advanced Level

**Q9: How would you implement a document similarity search system from scratch?**

**A:** Complete implementation:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class DocumentSearchEngine:
    def __init__(self, use_tfidf=True):
        """Initialize search engine"""
        self.use_tfidf = use_tfidf
        self.vectorizer = None
        self.doc_vectors = None
        self.documents = []
        self.doc_ids = []
    
    def fit(self, documents, doc_ids=None):
        """Index documents"""
        self.documents = documents
        self.doc_ids = doc_ids if doc_ids else list(range(len(documents)))
        
        # Create vectorizer
        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
        else:
            from sklearn.feature_extraction.text import CountVectorizer
            self.vectorizer = CountVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        # Vectorize documents
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        
        print(f"Indexed {len(documents)} documents")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return self
    
    def search(self, query, top_k=5):
        """Search for similar documents"""
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        
        # Get top K results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'doc_id': self.doc_ids[idx],
                'document': self.documents[idx],
                'similarity': similarities[idx]
            })
        
        return results
    
    def find_similar_docs(self, doc_id, top_k=5):
        """Find documents similar to a given document"""
        doc_idx = self.doc_ids.index(doc_id)
        doc_vector = self.doc_vectors[doc_idx]
        
        # Calculate similarities
        similarities = cosine_similarity(doc_vector, self.doc_vectors)[0]
        
        # Get top K (excluding itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for idx in top_indices:
            results.append({
                'doc_id': self.doc_ids[idx],
                'document': self.documents[idx],
                'similarity': similarities[idx]
            })
        
        return results
    
    def save(self, filepath):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'doc_vectors': self.doc_vectors,
                'documents': self.documents,
                'doc_ids': self.doc_ids,
                'use_tfidf': self.use_tfidf
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        engine = cls(use_tfidf=data['use_tfidf'])
        engine.vectorizer = data['vectorizer']
        engine.doc_vectors = data['doc_vectors']
        engine.documents = data['documents']
        engine.doc_ids = data['doc_ids']
        
        return engine

# Example usage
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Natural language processing helps computers understand text",
    "Computer vision enables machines to interpret images",
    "Reinforcement learning trains agents through rewards"
]

doc_ids = ['ML001', 'DL001', 'NLP001', 'CV001', 'RL001']

# Create and train engine
engine = DocumentSearchEngine(use_tfidf=True)
engine.fit(documents, doc_ids)

# Search
query = "neural networks and deep learning"
results = engine.search(query, top_k=3)

print(f"\nSearch results for: '{query}'")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Doc ID: {result['doc_id']}")
    print(f"   Similarity: {result['similarity']:.4f}")
    print(f"   Text: {result['document'][:100]}...")

# Find similar documents
similar = engine.find_similar_docs('ML001', top_k=2)
print(f"\nDocuments similar to 'ML001':")
for result in similar:
    print(f"- {result['doc_id']}: {result['similarity']:.4f}")

# Save model
engine.save('search_engine.pkl')

# Load model
loaded_engine = DocumentSearchEngine.load('search_engine.pkl')
```

---

**Q10: Explain the intuition behind TF-IDF. Why multiply TF by IDF?**

**A:**

**Intuition:**

1. **TF (Term Frequency):** "How important is this word in THIS document?"
   - High TF = word appears often in document = likely important to document

2. **IDF (Inverse Document Frequency):** "How unique/rare is this word ACROSS ALL documents?"
   - High IDF = word appears in few documents = more discriminative
   - Low IDF = word appears in many documents = less useful for distinguishing

**Why multiply?**

```
TF × IDF balances both factors:

Example 1: Common word "the"
- High TF (appears often in document)
- Low IDF (appears in all documents)
- Result: High × Low = Moderate/Low score ✓
  → Not useful for distinguishing documents

Example 2: Discriminative word "tensorflow"
- High TF (appears often in document)
- High IDF (appears in few documents)
- Result: High × High = Very High score ✓
  → Very useful for identifying relevant documents

Example 3: Rare but infrequent word "quantum"
- Low TF (appears once in document)
- High IDF (appears in few documents)
- Result: Low × High = Moderate score ✓
  → Somewhat useful
```

**Mathematical Intuition:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize TF-IDF components
tf_values = np.linspace(0, 1, 100)
idf_values = [0, 0.5, 1, 2, 3]  # Different IDF values

plt.figure(figsize=(10, 6))
for idf in idf_values:
    tfidf_values = tf_values * idf
    plt.plot(tf_values, tfidf_values, label=f'IDF={idf}')

plt.xlabel('Term Frequency (TF)')
plt.ylabel('TF-IDF Score')
plt.title('TF-IDF: How IDF Scales TF')
plt.legend()
plt.grid(True)
plt.show()

# Key insight: IDF acts as a scaling factor
# - Common words (low IDF) get scaled down
# - Rare words (high IDF) get scaled up
```

**Real Example:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the bird flew over the tree"
]

# Without IDF (just TF)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
tf_matrix = cv.fit_transform(docs)

# With TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(docs)

# Compare
vocab = cv.get_feature_names_out()
df_comparison = pd.DataFrame({
    'Word': vocab,
    'TF_Doc1': tf_matrix[0].toarray()[0],
    'TFIDF_Doc1': tfidf_matrix[0].toarray()[0]
})

print(df_comparison.sort_values('TF_Doc1', ascending=False))

# Notice: "the" has high TF but lower TF-IDF
#         "cat", "mat" have lower TF but higher TF-IDF (relative to their TF)
```

---

**Q11: How would you handle imbalanced text classification with different representation methods?**

**A:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

class ImbalancedTextClassifier:
    def __init__(self, method='class_weight'):
        """
        Methods:
        - 'class_weight': Weight classes inversely proportional to frequencies
        - 'oversample': SMOTE oversampling
        - 'undersample': Random undersampling
        - 'combined': SMOTE + undersampling
        """
        self.method = method
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = None
        
    def fit(self, X, y):
        """Train classifier"""
        # Vectorize
        X_vec = self.vectorizer.fit_transform(X)
        
        if self.method == 'class_weight':
            # Use class weights
            weights = class_weight.compute_class_weight(
                'balanced',
                classes=np.unique(y),
                y=y
            )
            class_weights = dict(enumerate(weights))
            
            self.classifier = LogisticRegression(
                class_weight=class_weights,
                max_iter=1000
            )
            self.classifier.fit(X_vec, y)
            
        elif self.method == 'oversample':
            # SMOTE oversampling
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_vec, y)
            
            self.classifier = LogisticRegression(max_iter=1000)
            self.classifier.fit(X_resampled, y_resampled)
            
        elif self.method == 'undersample':
            # Random undersampling
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X_vec, y)
            
            self.classifier = LogisticRegression(max_iter=1000)
            self.classifier.fit(X_resampled, y_resampled)
            
        elif self.method == 'combined':
            # SMOTE + undersampling
            from imblearn.pipeline import Pipeline
            pipeline = Pipeline([
                ('oversample', SMOTE(random_state=42)),
                ('undersample', RandomUnderSampler(random_state=42))
            ])
            X_resampled, y_resampled = pipeline.fit_resample(X_vec, y)
            
            self.classifier = LogisticRegression(max_iter=1000)
            self.classifier.fit(X_resampled, y_resampled)
        
        return self
    
    def predict(self, X):
        """Predict classes"""
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict(X_vec)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_vec)

# Example with imbalanced data
texts = [
    "This product is amazing",
    "Great quality, highly recommend",
    "Excellent service",
    # ... 97 more positive reviews
] + [
    "Terrible product",
    "Waste of money",
    "Do not buy"
    # ... only 3 negative reviews
]

labels = [1] * 100 + [0] * 3  # Highly imbalanced!

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Compare methods
from sklearn.metrics import classification_report, f1_score

methods = ['class_weight', 'oversample', 'undersample', 'combined']

for method in methods:
    clf = ImbalancedTextClassifier(method=method)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n{method.upper()} Results:")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
```

**Additional Strategies:**

```python
# 1. Ensemble methods
from sklearn.ensemble import RandomForestClassifier

class EnsembleTextClassifier:
    def __init__(self):
        self.vectorizers = {
            'tfidf_unigram': TfidfVectorizer(ngram_range=(1,1)),
            'tfidf_bigram': TfidfVectorizer(ngram_range=(1,2)),
            'bow': CountVectorizer()
        }
        self.classifiers = {}
    
    def fit(self, X, y):
        for name, vectorizer in self.vectorizers.items():
            X_vec = vectorizer.fit_transform(X)
            clf = RandomForestClassifier(
                class_weight='balanced',
                n_estimators=100
            )
            clf.fit(X_vec, y)
            self.classifiers[name] = (vectorizer, clf)
        return self
    
    def predict(self, X):
        predictions = []
        for vectorizer, clf in self.classifiers.values():
            X_vec = vectorizer.transform(X)
            predictions.append(clf.predict_proba(X_vec))
        
        # Average predictions
        avg_pred = np.mean(predictions, axis=0)
        return np.argmax(avg_pred, axis=1)

# 2. Threshold adjustment
def adjust_threshold(y_true, y_pred_proba, target_recall=0.9):
    """Find threshold that achieves target recall"""
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_pred_proba[:, 1]
    )
    
    # Find threshold closest to target recall
    idx = np.argmin(np.abs(recall - target_recall))
    best_threshold = thresholds[idx]
    
    return best_threshold

# 3. Cost-sensitive learning
from sklearn.svm import SVC

clf = SVC(
    class_weight={0: 10, 1: 1},  # Penalize minority class errors more
    probability=True
)
```

---

## 12. Practical Applications

### 1. Spam Detection System

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class SpamDetector:
    def __init__(self):
        """Initialize spam detector pipeline"""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
    
    def train(self, emails, labels):
        """Train the spam detector"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            emails, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Ham', 'Spam']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Ham', 'Spam'],
                   yticklabels=['Ham', 'Spam'])
        plt.title('Spam Detection Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, emails, labels, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self
    
    def predict(self, email):
        """Predict if email is spam"""
        prediction = self.pipeline.predict([email])[0]
        probability = self.pipeline.predict_proba([email])[0]
        
        return {
            'is_spam': bool(prediction),
            'spam_probability': probability[1],
            'ham_probability': probability[0]
        }
    
    def explain_prediction(self, email, top_n=10):
        """Show top features for prediction"""
        # Get feature names
        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        
        # Transform email
        email_vec = self.pipeline.named_steps['tfidf'].transform([email])
        
        # Get feature importances
        classifier = self.pipeline.named_steps['classifier']
        
        # Get log probabilities for each feature
        feature_log_prob = classifier.feature_log_prob_
        
        # Get non-zero features in email
        nonzero_idx = email_vec.nonzero()[1]
        
        # Calculate contributions
        contributions = []
        for idx in nonzero_idx:
            feature = feature_names[idx]
            spam_score = feature_log_prob[1][idx]
            ham_score = feature_log_prob[0][idx]
            diff = spam_score - ham_score
            contributions.append((feature, diff, email_vec[0, idx]))
        
        # Sort by contribution
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\nTop {top_n} features contributing to prediction:")
        for feature, score, weight in contributions[:top_n]:
            direction = "SPAM" if score > 0 else "HAM"
            print(f"  {feature:20s}: {score:+.4f} ({direction}) [weight: {weight:.4f}]")

# Example usage
emails = [
    "Get rich quick! Buy now! Limited offer!",
    "Meeting scheduled for tomorrow at 3pm",
    "WINNER! You've won $1,000,000! Click here!",
    "Can you review the document I sent?",
    "FREE PILLS! NO PRESCRIPTION NEEDED!",
    "Let's discuss the project timeline",
]

labels = [1, 0, 1, 0, 1, 0]  # 1=spam, 0=ham

# Train detector
detector = SpamDetector()
detector.train(emails * 20, labels * 20)  # Replicate for demo

# Test
test_email = "Congratulations! You won a free iPhone! Click now!"
result = detector.predict(test_email)

print(f"\nEmail: {test_email}")
print(f"Is Spam: {result['is_spam']}")
print(f"Spam Probability: {result['spam_probability']:.4f}")

# Explain
detector.explain_prediction(test_email)
```

### 2. Sentiment Analysis API

```python
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression()
        self.is_trained = False
    
    def train(self, texts, sentiments):
        """Train sentiment analyzer"""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, sentiments)
        self.is_trained = True
    
    def predict(self, text):
        """Predict sentiment"""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        X = self.vectorizer.transform([text])
        sentiment = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        
        return {
            'text': text,
            'sentiment': 'positive' if sentiment == 1 else 'negative',
            'confidence': float(max(proba)),
            'probabilities': {
                'negative': float(proba[0]),
                'positive': float(proba[1])
            }
        }
    
    def save(self, filepath):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        analyzer = cls()
        analyzer.vectorizer = data['vectorizer']
        analyzer.model = data['model']
        analyzer.is_trained = True
        
        return analyzer

# Flask API
app = Flask(__name__)
analyzer = None

@app.before_first_request
def load_model():
    global analyzer
    # Load pre-trained model or train new one
    try:
        analyzer = SentimentAnalyzer.load('sentiment_model.pkl')
    except:
        # Train on sample data
        texts = [
            "I love this product!",
            "This is terrible",
            # ... more training data
        ]
        sentiments = [1, 0]  # 1=positive, 0=negative
        
        analyzer = SentimentAnalyzer()
        analyzer.train(texts, sentiments)
        analyzer.save('sentiment_model.pkl')

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    """API endpoint for sentiment analysis"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        result = analyzer.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_sentiment', methods=['POST'])
def batch_analyze():
    """Batch sentiment analysis"""
    data = request.get_json()
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    try:
        results = [analyzer.predict(text) for text in texts]
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run API
if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### 3. Document Clustering System

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class DocumentClusterer:
    def __init__(self, n_clusters=5, method='kmeans'):
        """
        Initialize document clusterer
        
        Args:
            n_clusters: Number of clusters
            method: 'kmeans' or 'dbscan'
        """
        self.n_clusters = n_clusters
        self.method = method
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.clusterer = None
        self.doc_vectors = None
        self.documents = []
        self.labels = None
    
    def fit(self, documents):
        """Cluster documents"""
        self.documents = documents
        
        # Vectorize
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        
        # Cluster
        if self.method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        elif self.method == 'dbscan':
            self.clusterer = DBSCAN(
                eps=0.5,
                min_samples=2,
                metric='cosine'
            )
        
        self.labels = self.clusterer.fit_predict(self.doc_vectors)
        
        return self
    
    def get_top_terms_per_cluster(self, n_terms=10):
        """Get top terms for each cluster"""
        feature_names = self.vectorizer.get_feature_names_out()
        
        if self.method == 'kmeans':
            cluster_centers = self.clusterer.cluster_centers_
            
            top_terms = {}
            for i, center in enumerate(cluster_centers):
                top_indices = center.argsort()[-n_terms:][::-1]
                top_terms[i] = [feature_names[idx] for idx in top_indices]
            
            return top_terms
        else:
            # For DBSCAN, calculate mean vector per cluster
            top_terms = {}
            for label in set(self.labels):
                if label == -1:  # Noise
                    continue
                
                cluster_docs = self.doc_vectors[self.labels == label]
                mean_vector = cluster_docs.mean(axis=0).A1
                top_indices = mean_vector.argsort()[-n_terms:][::-1]
                top_terms[label] = [feature_names[idx] for idx in top_indices]
            
            return top_terms
    
    def visualize(self):
        """Visualize clusters using PCA"""
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        coords = pca.fit_transform(self.doc_vectors.toarray())
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            coords[:, 0], 
            coords[:, 1],
            c=self.labels,
            cmap='viridis',
            alpha=0.6,
            edgecolors='black'
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'Document Clusters ({self.method.upper()})')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        
        # Add cluster centers for K-means
        if self.method == 'kmeans':
            centers_2d = pca.transform(self.clusterer.cluster_centers_)
            plt.scatter(
                centers_2d[:, 0],
                centers_2d[:, 1],
                c='red',
                marker='X',
                s=200,
                edgecolors='black',
                linewidths=2,
                label='Centroids'
            )
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def print_cluster_summary(self):
        """Print summary of clusters"""
        top_terms = self.get_top_terms_per_cluster(n_terms=5)
        
        print(f"\nCluster Summary ({self.method.upper()}):")
        print("=" * 60)
        
        for cluster_id in sorted(set(self.labels)):
            if cluster_id == -1:
                cluster_name = "Noise"
            else:
                cluster_name = f"Cluster {cluster_id}"
            
            docs_in_cluster = np.sum(self.labels == cluster_id)
            print(f"\n{cluster_name}: {docs_in_cluster} documents")
            
            if cluster_id in top_terms:
                print(f"Top terms: {', '.join(top_terms[cluster_id])}")
            
            # Show example documents
            cluster_docs = [
                self.documents[i] 
                for i, label in enumerate(self.labels) 
                if label == cluster_id
            ][:3]
            
            print("Example documents:")
            for doc in cluster_docs:
                print(f"  - {doc[:80]}...")

# Example usage
documents = [
    "Machine learning and artificial intelligence",
    "Deep learning neural networks",
    "Natural language processing techniques",
    "Computer vision image recognition",
    "Cooking recipes and food preparation",
    "Baking cakes and pastries",
    "Healthy eating and nutrition",
    "Sports and fitness training",
    "Basketball and football games",
    "Olympic sports and athletics"
]

# K-means clustering
kmeans_clusterer = DocumentClusterer(n_clusters=3, method='kmeans')
kmeans_clusterer.fit(documents)
kmeans_clusterer.print_cluster_summary()
kmeans_clusterer.visualize()

# DBSCAN clustering
dbscan_clusterer = DocumentClusterer(method='dbscan')
dbscan_clusterer.fit(documents)
dbscan_clusterer.print_cluster_summary()
dbscan_clusterer.visualize()
```

---

## Summary

### Key Takeaways

1. **Choose the right representation for your task:**
   - Information retrieval → TF-IDF
   - Semantic tasks → Word embeddings
   - Simple classification → BoW/TF-IDF
   - State-of-the-art → BERT/Transformers

2. **Trade-offs matter:**
   - Sparse vs Dense
   - Interpretability vs Performance
   - Speed vs Accuracy
   - Memory vs Capability

3. **Preprocessing is crucial:**
   - Clean text properly
   - Handle contractions
   - Remove noise
   - Consider domain-specific preprocessing

4. **Evaluation is key:**
   - Always validate on held-out data
   - Use appropriate metrics
   - Consider class imbalance
   - Test on real-world data

### Next Steps

1. **Practice implementations:**
   - Build a document search engine
   - Create a sentiment analyzer
   - Develop a text classifier

2. **Explore advanced topics:**
   - Transformers and attention
   - Transfer learning with BERT
   - Multi-lingual embeddings
   - Domain-specific embeddings

3. **Read research papers:**
   - Word2Vec original paper
   - BERT paper
   - GPT series
   - Latest NLP advances

### Resources

- **Books:**
  - "Speech and Language Processing" by Jurafsky & Martin
  - "Natural Language Processing with Python" by Bird, Klein & Loper
  
- **Courses:**
  - Stanford CS224N: NLP with Deep Learning
  - fast.ai NLP course
  - Coursera NLP Specialization

- **Libraries:**
  - spaCy, NLTK, Gensim
  - Hugging Face Transformers
  - scikit-learn

---

**End of Guide**

This comprehensive guide covers text representation from fundamentals to advanced applications. Practice with real datasets and build projects to solidify your understanding!