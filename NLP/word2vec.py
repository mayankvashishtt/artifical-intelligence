# ============================================
# INSTALLATION
# ============================================

# !pip install gensim
# Installs the Gensim library
# ! = run as shell command in Jupyter/Colab
# gensim = library for Word2Vec, document similarity, topic modeling


# ============================================
# IMPORT
# ============================================

from gensim.models import Word2Vec
# Imports the Word2Vec class from gensim
# Allows you to train word embedding models


# ============================================
# DATA PREPARATION
# ============================================

sentences = [
    "i love machine learning",
    "i love deep learning",
    "machine learning is fun",
    "deep learning is powerful",
    "i enjoy studying nlp",
    "nlp and machine learning are related"
]
# List of 6 sentences (documents)
# These are your training data
# Word2Vec will learn from the context of words in these sentences


tokenized_sentences = [sentence.split() for sentence in sentences]
# Converts sentences to lists of words
# Example: "i love machine learning" → ["i", "love", "machine", "learning"]
#
# tokenized_sentences = [
#     ["i", "love", "machine", "learning"],
#     ["i", "love", "deep", "learning"],
#     ["machine", "learning", "is", "fun"],
#     ["deep", "learning", "is", "powerful"],
#     ["i", "enjoy", "studying", "nlp"],
#     ["nlp", "and", "machine", "learning", "are", "related"]
# ]


# ============================================
# MODEL TRAINING
# ============================================

model = Word2Vec(
    sentences=tokenized_sentences,
    # Input: tokenized sentences (required)
    # This is what the model learns from
    
    vector_size=50,
    # Embedding dimension (how many numbers represent each word)
    # Lower values: faster training, less semantic information
    # Higher values: slower training, more semantic information
    # Typical range: 50-300
    # Example: "learning" = [0.25, -0.48, 0.81, ..., 0.12]  (50 numbers)
    
    window=2,
    # Context window size (how many words on each side to look at)
    # window=2 means: look 2 words LEFT and 2 words RIGHT
    #
    # Example with "learning" in "i love machine learning is fun":
    # Context window = ["machine", "is"]
    # (2 words before and after "learning")
    # 
    # Larger window (5-10): captures broader semantic relationships
    # Smaller window (2-3): captures more syntactic relationships
    
    min_count=1,
    # Minimum word frequency threshold
    # min_count=1 means include ALL words, even if they appear only once
    # min_count=5 would ignore words appearing < 5 times
    # Use higher values for large corpora (ignores rare words/typos)
    
    sg=1
    # sg=1: Skip-Gram model (context → target word)
    #       Predicts the word from its context
    #       Better for small datasets
    #
    # sg=0: CBOW (Continuous Bag of Words) model (context → target word)
    #       Predicts context from the word
    #       Faster, better for large datasets
)
# After this line, the model is TRAINED on your sentences
# It has learned 50-dimensional vectors for each word


# ============================================
# EXPLORING THE MODEL
# ============================================

model.wv.most_similar("learning")
# Find words MOST SIMILAR to "learning"
# Uses cosine similarity to compare vectors
# Returns top 10 most similar words by default
#
# Output example:
# [('deep', 0.8234),
#  ('machine', 0.7821),
#  ('studying', 0.6543),
#  ('nlp', 0.5234)]
#
# Interpretation:
# - "deep" is 82.34% similar to "learning"
# - "machine" is 78.21% similar
# - These words appear in similar contexts in your sentences


model.wv.similarity("machine", "learning")
# Calculate cosine similarity between two words
# Returns a single number between -1 and 1
# (usually 0 to 1 for word vectors)
#
# Example output: 0.8234
# Interpretation:
# - 0.8234 = 82.34% similar
# - 1.0 = identical vectors
# - 0.5 = moderately similar
# - 0.0 = not similar (perpendicular)
# - -1.0 = opposite (rarely happens with word vectors)


model.wv["learning"]
# Get the embedding vector for the word "learning"
# Returns a numpy array of 50 numbers
#
# Output example:
# array([ 0.25, -0.48,  0.81,  0.3,   -0.12,  0.56, ...], dtype=float32)
# This is the LEARNED representation of "learning"
# 
# Interpretation:
# - Each position (0-49) represents a semantic dimension
# - Semantically similar words have similar vectors
# - You can use this vector for downstream tasks (classification, clustering, etc.)

'''
Original sentences:
"i love machine learning"
"i love deep learning"
"machine learning is fun"

Word2Vec learns:
- "machine" and "learning" often appear together → similar vectors
- "deep" and "machine" both precede "learning" → somewhat similar
- "learning" appears in 3 sentences with varied contexts → captures semantic meaning

Result:
model.wv.most_similar("learning")
→ [("deep", 0.82), ("machine", 0.78), ...]
  (These words appear in similar contexts!)


'''