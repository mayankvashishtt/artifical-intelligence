# NLTK (Natural Language Toolkit): Complete Guide from Basics to Advanced

A comprehensive guide covering NLTK fundamentals, tokenization, text normalization, stopwords, stemming, lemmatization, and practical applications with examples and interview questions.

---

## Table of Contents

1. [What is NLTK?](#1-what-is-nltk)
2. [Installation and Setup](#2-installation-and-setup)
3. [Tokenization](#3-tokenization)
4. [Regular Expressions in NLP](#4-regular-expressions-in-nlp)
5. [Stopwords](#5-stopwords)
6. [Text Normalization](#6-text-normalization)
7. [Stemming](#7-stemming)
8. [Lemmatization](#8-lemmatization)
9. [POS Tagging](#9-pos-tagging)
10. [Named Entity Recognition](#10-named-entity-recognition)
11. [Text Preprocessing Pipeline](#11-text-preprocessing-pipeline)
12. [Sparse vs Dense Representations](#12-sparse-vs-dense-representations)
13. [Interview Questions](#13-interview-questions)
14. [Practical Projects](#14-practical-projects)

---

## 1. What is NLTK?

### Overview

**NLTK (Natural Language Toolkit)** is a leading platform for building Python programs to work with human language data (NLP).

### Key Features

- **Tokenization:** Break text into words, sentences
- **Text Cleaning:** Remove stopwords, punctuation
- **Text Normalization:** Stemming, lemmatization
- **POS Tagging:** Part-of-speech identification
- **NER:** Named Entity Recognition
- **Parsing:** Syntactic analysis
- **Corpora Access:** 50+ corpora and lexical resources

### Why Use NLTK?

✅ Comprehensive library for NLP tasks  
✅ Easy to learn and use  
✅ Extensive documentation  
✅ Large community support  
✅ Perfect for learning and prototyping  
✅ Over 50+ corpora datasets  

❌ Slower than spaCy for production  
❌ Not optimized for deep learning  
❌ Requires manual downloads of data  

### NLTK vs Other Libraries

| Feature | NLTK | spaCy | Gensim | TextBlob |
|---------|------|-------|--------|----------|
| **Speed** | Slow | Fast | Medium | Slow |
| **Ease of Use** | Medium | Easy | Medium | Very Easy |
| **Production Ready** | No | Yes | Yes | No |
| **Deep Learning** | No | Yes | No | No |
| **Learning Curve** | Steep | Gentle | Medium | Very Gentle |
| **Best For** | Research, Learning | Production | Topic Modeling | Quick Prototyping |

---

## 2. Installation and Setup

### Basic Installation

```bash
# Install NLTK
pip install nltk

# For Jupyter notebooks
!pip install nltk
```

### Downloading NLTK Data

NLTK requires additional data packages to be downloaded separately.

```python
import nltk

# Download all packages (not recommended - large!)
nltk.download('all')

# Download specific packages (recommended)
nltk.download('punkt')        # Tokenizer models
nltk.download('punkt_tab')    # Additional tokenizer data
nltk.download('stopwords')    # Stopwords lists
nltk.download('wordnet')      # WordNet lexical database
nltk.download('averaged_perceptron_tagger')  # POS tagger
nltk.download('maxent_ne_chunker')  # Named entity chunker
nltk.download('words')        # Word lists

# Download multiple at once
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
```

### What is 'punkt'?

**punkt** is a pre-trained tokenizer model that uses unsupervised learning to identify sentence boundaries.

**Key Features:**
- Trained on multiple languages
- Handles abbreviations (Dr., Mr., etc.)
- Recognizes decimal numbers
- Understands sentence boundaries

**Example:**

```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Dr. Smith works at U.S.A. He earns $1,000.50 daily."

# Without punkt - would split incorrectly
# With punkt - handles abbreviations correctly
sentences = sent_tokenize(text)
print(sentences)
# Output: ['Dr. Smith works at U.S.A.', 'He earns $1,000.50 daily.']
```

### Setting Up NLTK Data Path

```python
import nltk
import os

# Check current data path
print(nltk.data.path)

# Add custom data path
custom_path = '/path/to/nltk_data'
if custom_path not in nltk.data.path:
    nltk.data.path.append(custom_path)

# Set environment variable (permanent)
os.environ['NLTK_DATA'] = custom_path
```

### Complete Setup Script

```python
import nltk
import sys

def setup_nltk():
    """Complete NLTK setup with error handling"""
    
    required_packages = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'brown',
        'names'
    ]
    
    print("Setting up NLTK...")
    print("=" * 50)
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
            print(f"✓ {package} already installed")
        except LookupError:
            try:
                print(f"⬇ Downloading {package}...")
                nltk.download(package, quiet=True)
                print(f"✓ {package} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {package}: {e}")
                sys.exit(1)
    
    print("=" * 50)
    print("✓ NLTK setup complete!")
    print(f"NLTK version: {nltk.__version__}")
    print(f"Data path: {nltk.data.path[0]}")

# Run setup
setup_nltk()
```

---

## 3. Tokenization

### What is Tokenization?

**Tokenization** is the process of breaking text into smaller units called **tokens**.

**Why is it Important?**
- First step in NLP pipeline
- Converts text into processable units
- Essential for feature extraction
- Required for most NLP tasks

### Types of Tokenization

1. **Word Tokenization:** Split text into words
2. **Sentence Tokenization:** Split text into sentences
3. **Character Tokenization:** Split text into characters
4. **Subword Tokenization:** Split into subword units (BPE, WordPiece)

---

### Word Tokenization

#### Method 1: Python Split (Basic)

```python
text = "I am learning NLP!"

# Basic split
tokens = text.split()
print(tokens)
# Output: ['I', 'am', 'learning', 'NLP!']

# Problems:
# ❌ Punctuation attached to words
# ❌ Can't handle multiple separators
# ❌ No special case handling
```

**Limitations:**
- Only one separator at a time
- Punctuation not separated
- No handling of special cases

#### Method 2: NLTK Word Tokenize (Recommended)

```python
from nltk.tokenize import word_tokenize

text = "I'm learning NLP! It's amazing."

tokens = word_tokenize(text)
print(tokens)
# Output: ['I', "'m", 'learning', 'NLP', '!', 'It', "'s", 'amazing', '.']
```

**Advantages:**
✅ Separates punctuation  
✅ Handles contractions  
✅ Language-aware  
✅ Handles special characters  

**Note:** Punctuation is treated as separate tokens!

#### Method 3: Regular Expressions

```python
import re

text = "I'm learning NLP! It's amazing."

# Pattern 1: Alphanumeric and apostrophes
tokens = re.findall(r"[\w']+", text)
print("Pattern 1:", tokens)
# Output: ["I'm", 'learning', 'NLP', "It's", 'amazing']

# Pattern 2: Words only
tokens = re.findall(r'\b\w+\b', text)
print("Pattern 2:", tokens)
# Output: ['I', 'm', 'learning', 'NLP', 'It', 's', 'amazing']

# Pattern 3: Custom pattern
tokens = re.findall(r'\b[a-zA-Z]+\b', text)
print("Pattern 3:", tokens)
# Output: ['I', 'm', 'learning', 'NLP', 'It', 's', 'amazing']
```

#### Method 4: TreebankWordTokenizer

```python
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

text = "She said, \"It's a beautiful day!\""

tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['She', 'said', ',', '``', 'It', "'s", 'a', 'beautiful', 'day', '!', "''"]
```

**Advantages:**
- Handles quotes properly
- Separates punctuation
- Used in Penn Treebank

#### Method 5: WordPunctTokenizer

```python
from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()

text = "I'm learning NLP! Email: test@email.com"

tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['I', "'", 'm', 'learning', 'NLP', '!', 'Email', ':', 'test', '@', 'email', '.', 'com']
```

**Characteristics:**
- Splits all punctuation
- Very aggressive tokenization
- Useful for specific tasks

### Complete Word Tokenization Example

```python
import re
from nltk.tokenize import word_tokenize, WordPunctTokenizer, TreebankWordTokenizer

def compare_tokenizers(text):
    """Compare different tokenization methods"""
    
    print(f"Original text: {text}\n")
    print("=" * 60)
    
    # Method 1: Python split
    tokens_split = text.split()
    print(f"1. Python split(): {tokens_split}")
    print(f"   Token count: {len(tokens_split)}\n")
    
    # Method 2: NLTK word_tokenize
    tokens_nltk = word_tokenize(text)
    print(f"2. NLTK word_tokenize(): {tokens_nltk}")
    print(f"   Token count: {len(tokens_nltk)}\n")
    
    # Method 3: Regex
    tokens_regex = re.findall(r"[\w']+", text)
    print(f"3. Regex: {tokens_regex}")
    print(f"   Token count: {len(tokens_regex)}\n")
    
    # Method 4: TreebankWordTokenizer
    treebank = TreebankWordTokenizer()
    tokens_treebank = treebank.tokenize(text)
    print(f"4. TreebankWordTokenizer: {tokens_treebank}")
    print(f"   Token count: {len(tokens_treebank)}\n")
    
    # Method 5: WordPunctTokenizer
    wordpunct = WordPunctTokenizer()
    tokens_wordpunct = wordpunct.tokenize(text)
    print(f"5. WordPunctTokenizer: {tokens_wordpunct}")
    print(f"   Token count: {len(tokens_wordpunct)}\n")

# Test
text = "I'm learning NLP! Email me at test@email.com. It's amazing!"
compare_tokenizers(text)
```

---

### Sentence Tokenization

#### Method 1: Python Split (Basic)

```python
text = "Hello world. This is NLP. It's amazing."

# Split by period
sentences = text.split('. ')
print(sentences)
# Output: ['Hello world', 'This is NLP', "It's amazing."]

# Problems:
# ❌ Last sentence has period
# ❌ Fails with abbreviations (Dr., Mr.)
# ❌ Can't handle multiple punctuation
```

#### Method 2: Regular Expressions

```python
import re

text = "Hello world. This is NLP! Is it amazing? Yes."

# Split by multiple punctuation
sentences = re.split(r'[.!?] ', text)
print(sentences)
# Output: ['Hello world', 'This is NLP', 'Is it amazing', 'Yes.']

# Better regex
sentences = re.compile('[.!?] ').split(text)
print(sentences)
```

**Advantages over split():**
✅ Multiple separators  
✅ More flexible  

**Limitations:**
❌ Still fails with abbreviations  
❌ No context awareness  

#### Method 3: NLTK Sent Tokenize (Recommended)

```python
from nltk.tokenize import sent_tokenize

text = "Dr. A.P.J. Abdul Kalam was the Former President of India. He was born on Oct. 15, 1931. He passed away in 2015."

sentences = sent_tokenize(text)
print(sentences)
# Output: [
#   'Dr. A.P.J. Abdul Kalam was the Former President of India.',
#   'He was born on Oct. 15, 1931.',
#   'He passed away in 2015.'
# ]
```

**Advantages:**
✅ Handles abbreviations (Dr., Mr., Oct.)  
✅ Context-aware  
✅ Trained on multiple languages  
✅ Handles decimals (1.5, 2.3)  
✅ Multi-language support  

#### Comparison: Split vs Regex vs NLTK

```python
import re
from nltk.tokenize import sent_tokenize

def compare_sentence_tokenizers(text):
    """Compare sentence tokenization methods"""
    
    print(f"Original text:\n{text}\n")
    print("=" * 60)
    
    # Method 1: Split
    split_sentences = text.split('. ')
    print("1. Python split():")
    for i, sent in enumerate(split_sentences, 1):
        print(f"   {i}. {sent}")
    print()
    
    # Method 2: Regex
    regex_sentences = re.compile('[.!?] ').split(text)
    print("2. Regular Expression:")
    for i, sent in enumerate(regex_sentences, 1):
        print(f"   {i}. {sent}")
    print()
    
    # Method 3: NLTK
    nltk_sentences = sent_tokenize(text)
    print("3. NLTK sent_tokenize():")
    for i, sent in enumerate(nltk_sentences, 1):
        print(f"   {i}. {sent}")

# Test with tricky text
text = "Dr. Smith works at U.S.A. He earns $1,000.50 daily! Isn't that great? Yes."
compare_sentence_tokenizers(text)
```

**Output Analysis:**

```
1. Python split():
   ❌ "Dr" separated from "Smith"
   ❌ "U.S.A" broken incorrectly
   
2. Regular Expression:
   ❌ Same issues as split
   ❌ Slightly better with multiple punctuation
   
3. NLTK sent_tokenize():
   ✅ Correctly handles all cases
   ✅ Keeps "Dr. Smith" together
   ✅ Handles "U.S.A." correctly
   ✅ Preserves "$1,000.50"
```

### Advanced Tokenization

#### Custom Tokenizer

```python
from nltk.tokenize.regexp import RegexpTokenizer

# Create custom tokenizer
tokenizer = RegexpTokenizer(r'\w+')

text = "Hello! This is a test. Email: test@email.com"
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['Hello', 'This', 'is', 'a', 'test', 'Email', 'test', 'email', 'com']
```

#### Whitespace Tokenizer

```python
from nltk.tokenize import WhitespaceTokenizer

tokenizer = WhitespaceTokenizer()

text = "Hello   world\n\tThis is   NLP"
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['Hello', 'world', 'This', 'is', 'NLP']
```

#### Tweet Tokenizer (Social Media)

```python
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer(
    preserve_case=False,    # Convert to lowercase
    reduce_len=True,        # Reduce repeated chars ('hellooo' -> 'helloo')
    strip_handles=True      # Remove @mentions
)

tweet = "@user This is AMAZING!!! Check out #NLP http://example.com 😊"

tokens = tokenizer.tokenize(tweet)
print(tokens)
# Output: ['this', 'is', 'amazing', '!', 'check', 'out', '#nlp', 'http://example.com', '😊']
```

#### Multi-Word Expression (MWE) Tokenizer

```python
from nltk.tokenize import MWETokenizer

# Create tokenizer with multi-word expressions
tokenizer = MWETokenizer([
    ('New', 'York'),
    ('machine', 'learning'),
    ('natural', 'language', 'processing')
])

text = "I'm studying machine learning and natural language processing in New York"

tokens = word_tokenize(text)
tokens = tokenizer.tokenize(tokens)
print(tokens)
# Output: ['I', "'m", 'studying', 'machine_learning', 'and', 
#          'natural_language_processing', 'in', 'New_York']
```

---

## 4. Regular Expressions in NLP

### Why Regular Expressions?

Regular expressions (regex) are powerful patterns for matching text.

**Use Cases in NLP:**
- Custom tokenization
- Email/URL extraction
- Pattern matching
- Text cleaning
- Data validation

### Basic Regex Patterns

```python
import re

# Common patterns
patterns = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'hashtag': r'#\w+',
    'mention': r'@\w+',
    'number': r'\b\d+\.?\d*\b',
    'word': r'\b[a-zA-Z]+\b'
}

text = """
Contact: john.doe@email.com
Visit: https://example.com
Call: 123-456-7890
Tweet: #NLP @user
Price: $99.99
"""

# Extract patterns
for pattern_name, pattern in patterns.items():
    matches = re.findall(pattern, text)
    if matches:
        print(f"{pattern_name}: {matches}")
```

### Email Address Extraction

```python
import re

def extract_emails(text):
    """Extract email addresses from text"""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, text)
    return emails

text = """
Contact us:
- Sales: sales@company.com
- Support: support@company.com
- Info: info@company.co.uk
Invalid: test@, @test.com, test
"""

emails = extract_emails(text)
print(f"Found emails: {emails}")
# Output: ['sales@company.com', 'support@company.com', 'info@company.co.uk']
```

**Pattern Breakdown:**
```
^([a-zA-Z0-9_\-\.]+)  → Username part
@                      → @ symbol
([a-zA-Z0-9_\-\.]+)   → Domain name
\.                     → Dot
([a-zA-Z]{2,5})$      → TLD (com, org, etc.)
```

### URL Extraction and Cleaning

```python
import re

def extract_urls(text):
    """Extract URLs from text"""
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(pattern, text)
    return urls

def remove_urls(text):
    """Remove URLs from text"""
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(pattern, '', text)

text = """
Check out my blog: https://example.com
Download: http://files.example.com/file.pdf
Visit: www.example.com
"""

print("URLs found:", extract_urls(text))
print("Text without URLs:", remove_urls(text))
```

### Social Media Pattern Extraction

```python
import re

def extract_social_patterns(text):
    """Extract hashtags, mentions, and emojis"""
    
    patterns = {
        'hashtags': r'#\w+',
        'mentions': r'@\w+',
        'emojis': r'[\U0001F600-\U0001F64F]',  # Emoticons
        'urls': r'http[s]?://\S+',
        'numbers': r'\b\d+\.?\d*\b'
    }
    
    results = {}
    for name, pattern in patterns.items():
        results[name] = re.findall(pattern, text)
    
    return results

tweet = """
@user Just finished reading about #NLP and #MachineLearning 🎉
Check it out: https://example.com
Score: 95.5% 😊
"""

patterns = extract_social_patterns(tweet)
for pattern_name, matches in patterns.items():
    print(f"{pattern_name}: {matches}")
```

### Text Cleaning with Regex

```python
import re

class RegexCleaner:
    def __init__(self):
        """Initialize text cleaner with regex patterns"""
        self.patterns = {
            'url': r'http[s]?://\S+',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'mention': r'@\w+',
            'hashtag': r'#\w+',
            'number': r'\b\d+\.?\d*\b',
            'punctuation': r'[^\w\s]',
            'extra_spaces': r'\s+',
            'html_tags': r'<[^>]+>',
            'special_chars': r'[^a-zA-Z0-9\s]'
        }
    
    def remove_pattern(self, text, pattern_name):
        """Remove specific pattern from text"""
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
            return re.sub(pattern, '', text)
        return text
    
    def clean_text(self, text, remove=['url', 'email', 'html_tags']):
        """Clean text by removing specified patterns"""
        cleaned = text
        for pattern_name in remove:
            cleaned = self.remove_pattern(cleaned, pattern_name)
        
        # Remove extra spaces
        cleaned = re.sub(self.patterns['extra_spaces'], ' ', cleaned)
        return cleaned.strip()
    
    def replace_pattern(self, text, pattern_name, replacement=''):
        """Replace pattern with specified replacement"""
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
            return re.sub(pattern, replacement, text)
        return text

# Example usage
cleaner = RegexCleaner()

text = """
<h1>Title</h1>
Check out https://example.com
Contact: test@email.com
Follow @user and use #NLP
Price: $99.99
"""

# Clean text
cleaned = cleaner.clean_text(text, remove=['url', 'email', 'html_tags'])
print("Cleaned:", cleaned)

# Remove specific patterns
no_mentions = cleaner.remove_pattern(text, 'mention')
print("No mentions:", no_mentions)

# Replace numbers
no_numbers = cleaner.replace_pattern(text, 'number', '[NUMBER]')
print("Numbers replaced:", no_numbers)
```

### Advanced Regex Examples

```python
import re

# 1. Extract dates
def extract_dates(text):
    """Extract dates in various formats"""
    patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',          # MM/DD/YYYY
        r'\d{1,2}-\d{1,2}-\d{2,4}',          # MM-DD-YYYY
        r'\d{4}-\d{1,2}-\d{1,2}',            # YYYY-MM-DD
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'  # 15 Jan 2023
    ]
    
    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    return dates

# 2. Extract phone numbers
def extract_phone_numbers(text):
    """Extract phone numbers in various formats"""
    patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',    # 123-456-7890
        r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',      # (123) 456-7890
        r'\+\d{1,3}\s*\d{3}[-.]?\d{3}[-.]?\d{4}'  # +1 123-456-7890
    ]
    
    phones = []
    for pattern in patterns:
        phones.extend(re.findall(pattern, text))
    return phones

# 3. Extract prices
def extract_prices(text):
    """Extract prices from text"""
    pattern = r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?'
    prices = re.findall(pattern, text)
    return prices

# 4. Validate input
def validate_input(input_str, input_type):
    """Validate different types of input"""
    patterns = {
        'email': r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$',
        'phone': r'^\d{3}[-.]?\d{3}[-.]?\d{4}$',
        'zipcode': r'^\d{5}(-\d{4})?$',
        'username': r'^[a-zA-Z0-9_]{3,20}$',
        'password': r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$'
    }
    
    if input_type in patterns:
        return bool(re.match(patterns[input_type], input_str))
    return False

# Test examples
text = """
Dates: 12/31/2023, 2023-12-31, 15 Jan 2023
Phones: 123-456-7890, (123) 456-7890, +1 123-456-7890
Prices: $19.99, $1,234.56, $99
"""

print("Dates:", extract_dates(text))
print("Phones:", extract_phone_numbers(text))
print("Prices:", extract_prices(text))

# Validation
print("\nValidation:")
print("Email valid:", validate_input("test@email.com", "email"))
print("Phone valid:", validate_input("123-456-7890", "phone"))
print("Username valid:", validate_input("user_123", "username"))
```

---

## 5. Stopwords

### What are Stopwords?

**Stopwords** are common words that don't carry much meaning and are often filtered out in NLP tasks.

**Examples:** the, is, at, which, on, a, an, and, or, but

### Why Remove Stopwords?

✅ Reduce dimensionality  
✅ Reduce noise  
✅ Improve processing speed  
✅ Focus on meaningful words  
✅ Better model performance  

❌ Loss of context in some tasks  
❌ Can affect sentiment analysis  
❌ Important for some NLP tasks  

### When to Remove Stopwords?

**Remove:**
- Text classification
- Information retrieval
- Topic modeling
- Keyword extraction

**Keep:**
- Sentiment analysis ("not good" vs "good")
- Question answering
- Machine translation
- Named entity recognition

### NLTK Stopwords

```python
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')

# Get English stopwords
stop_words = set(stopwords.words('english'))

print(f"Total stopwords: {len(stop_words)}")
print(f"Sample stopwords: {list(stop_words)[:20]}")

# Check if word is stopword
print("'the' is stopword:", 'the' in stop_words)
print("'python' is stopword:", 'python' in stop_words)
```

### Removing Stopwords

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(text, custom_stopwords=None):
    """Remove stopwords from text"""
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Add custom stopwords
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Filter stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return filtered_tokens

# Example
text = "This is an example sentence showing the removal of stopwords"

print("Original:", text)
print("Tokens:", word_tokenize(text))
print("Filtered:", remove_stopwords(text))

# Output:
# Original: This is an example sentence showing the removal of stopwords
# Tokens: ['This', 'is', 'an', 'example', 'sentence', 'showing', 'the', 'removal', 'of', 'stopwords']
# Filtered: ['example', 'sentence', 'showing', 'removal', 'stopwords']
```

### Custom Stopwords

```python
from nltk.corpus import stopwords

# Get default stopwords
stop_words = set(stopwords.words('english'))

# Add custom stopwords
custom_stops = ['said', 'says', 'would', 'could', 'might']
stop_words.update(custom_stops)

# Remove specific stopwords
stop_words.discard('not')  # Keep 'not' for sentiment
stop_words.discard('no')

print(f"Total stopwords: {len(stop_words)}")

def remove_stopwords_custom(text, keep_words=None):
    """Remove stopwords but keep specific words"""
    stop_words = set(stopwords.words('english'))
    
    # Remove words we want to keep
    if keep_words:
        for word in keep_words:
            stop_words.discard(word)
    
    tokens = word_tokenize(text.lower())
    filtered = [word for word in tokens if word not in stop_words]
    
    return filtered

# Keep negations for sentiment analysis
text = "This is not good but not bad either"
filtered = remove_stopwords_custom(text, keep_words=['not'])
print(filtered)
# Output: ['not', 'good', 'not', 'bad', 'either']
```

### Stopwords in Multiple Languages

```python
from nltk.corpus import stopwords

# Available languages
print("Available languages:")
print(stopwords.fileids())

# Get stopwords for different languages
languages = ['english', 'spanish', 'french', 'german']

for lang in languages:
    stop_words = stopwords.words(lang)
    print(f"\n{lang.capitalize()}: {len(stop_words)} stopwords")
    print(f"Sample: {stop_words[:10]}")

# Example: Multilingual stopword removal
def remove_stopwords_multilang(text, language='english'):
    """Remove stopwords for specified language"""
    from nltk.tokenize import word_tokenize
    
    try:
        stop_words = set(stopwords.words(language))
        tokens = word_tokenize(text.lower())
        filtered = [word for word in tokens if word not in stop_words]
        return filtered
    except OSError:
        print(f"Stopwords for '{language}' not found")
        return None

# Test
spanish_text = "Este es un ejemplo de texto en español"
filtered_spanish = remove_stopwords_multilang(spanish_text, 'spanish')
print(f"\nSpanish example: {filtered_spanish}")
```

### Complete Stopword Handler

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

class StopwordHandler:
    def __init__(self, language='english', custom_stopwords=None, 
                 keep_words=None, remove_punctuation=True):
        """
        Initialize stopword handler
        
        Args:
            language: Language for stopwords
            custom_stopwords: Additional stopwords to add
            keep_words: Words to keep (remove from stopwords)
            remove_punctuation: Whether to remove punctuation
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.remove_punctuation = remove_punctuation
        
        # Add custom stopwords
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # Remove words to keep
        if keep_words:
            for word in keep_words:
                self.stop_words.discard(word)
        
        # Add punctuation to stopwords if needed
        if remove_punctuation:
            self.stop_words.update(set(string.punctuation))
    
    def remove(self, text, lowercase=True):
        """Remove stopwords from text"""
        if lowercase:
            text = text.lower()
        
        tokens = word_tokenize(text)
        filtered = [word for word in tokens if word not in self.stop_words]
        
        return filtered
    
    def get_stopwords(self):
        """Get current stopwords"""
        return self.stop_words
    
    def add_stopwords(self, words):
        """Add new stopwords"""
        if isinstance(words, str):
            words = [words]
        self.stop_words.update(words)
    
    def remove_stopwords(self, words):
        """Remove words from stopwords list"""
        if isinstance(words, str):
            words = [words]
        for word in words:
            self.stop_words.discard(word)
    
    def filter_corpus(self, documents):
        """Filter stopwords from multiple documents"""
        return [self.remove(doc) for doc in documents]

# Example usage
handler = StopwordHandler(
    language='english',
    custom_stopwords=['example', 'test'],
    keep_words=['not', 'no'],  # Keep for sentiment
    remove_punctuation=True
)

text = "This is not a test example! It's important."
filtered = handler.remove(text)
print("Filtered:", filtered)

# Add more stopwords
handler.add_stopwords(['important', 'example'])
filtered2 = handler.remove(text)
print("After adding stopwords:", filtered2)

# Process multiple documents
documents = [
    "This is the first document",
    "This is the second document",
    "And this is the third one"
]
filtered_docs = handler.filter_corpus(documents)
print("\nFiltered documents:")
for i, doc in enumerate(filtered_docs, 1):
    print(f"{i}. {doc}")
```

---

## 6. Text Normalization

### What is Text Normalization?

**Text Normalization** is the process of transforming text into a canonical (standard) form.

**Goals:**
- Reduce variations
- Standardize format
- Improve consistency
- Reduce vocabulary size

### Types of Normalization

1. **Case Normalization:** Convert to lowercase/uppercase
2. **Removing Punctuation:** Strip special characters
3. **Removing Numbers:** Filter numeric values
4. **Expanding Contractions:** "don't" → "do not"
5. **Removing Extra Whitespace:** Clean spacing
6. **Unicode Normalization:** Handle special characters
7. **Stemming:** Reduce words to root form
8. **Lemmatization:** Reduce words to dictionary form

### Case Normalization

```python
text = "Python is AWESOME! I LOVE Python."

# Lowercase (most common)
lowercase = text.lower()
print("Lowercase:", lowercase)
# Output: python is awesome! i love python.

# Uppercase
uppercase = text.upper()
print("Uppercase:", uppercase)
# Output: PYTHON IS AWESOME! I LOVE PYTHON.

# Title case
titlecase = text.title()
print("Title:", titlecase)
# Output: Python Is Awesome! I Love Python.

# Capitalize (first letter only)
capitalize = text.capitalize()
print("Capitalize:", capitalize)
# Output: Python is awesome! i love python.
```

**When to use:**
- **Lowercase:** Most NLP tasks (default)
- **Preserve case:** Named Entity Recognition, POS tagging
- **Title case:** Display purposes

### Removing Punctuation

```python
import string
from nltk.tokenize import word_tokenize

def remove_punctuation(text):
    """Remove punctuation from text"""
    # Method 1: Using translate
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_punctuation_tokens(text):
    """Remove punctuation after tokenization"""
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in string.punctuation]

# Example
text = "Hello, world! This is amazing... Right?"

print("Original:", text)
print("Method 1:", remove_punctuation(text))
print("Method 2:", remove_punctuation_tokens(text))

# Output:
# Original: Hello, world! This is amazing... Right?
# Method 1: Hello world This is amazing Right
# Method 2: ['Hello', 'world', 'This', 'is', 'amazing', 'Right']
```

### Removing Numbers

```python
import re

def remove_numbers(text):
    """Remove numbers from text"""
    return re.sub(r'\d+', '', text)

def remove_numbers_keep_text(text):
    """Remove standalone numbers but keep numbers in words"""
    return re.sub(r'\b\d+\b', '', text)

# Example
text = "I have 3 apples and 2 oranges. Python3 is great!"

print("Original:", text)
print("Remove all:", remove_numbers(text))
print("Remove standalone:", remove_numbers_keep_text(text))

# Output:
# Original: I have 3 apples and 2 oranges. Python3 is great!
# Remove all: I have  apples and  oranges. Python is great!
# Remove standalone: I have  apples and  oranges. Python3 is great!
```

### Expanding Contractions

```python
import contractions

def expand_contractions(text):
    """Expand contractions in text"""
    return contractions.fix(text)

# Manual contractions dictionary
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "I'm": "I am",
    "I've": "I have",
    "I'll": "I will",
    "you're": "you are",
    "you've": "you have",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is"
}

def expand_contractions_manual(text):
    """Expand contractions using manual dictionary"""
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text

# Example
text = "I'm learning NLP. It's amazing! I can't wait to learn more."

print("Original:", text)
print("Using contractions lib:", expand_contractions(text))
print("Using manual dict:", expand_contractions_manual(text))
```

### Removing Extra Whitespace

```python
import re

def remove_extra_whitespace(text):
    """Remove extra whitespace"""
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text

# Example
text = "  This   is   a    text  with   extra   spaces.  "

print("Original:", repr(text))
print("Cleaned:", repr(remove_extra_whitespace(text)))

# Output:
# Original: '  This   is   a    text  with   extra   spaces.  '
# Cleaned: 'This is a text with extra spaces.'
```

### Complete Text Normalizer

```python
import re
import string
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class TextNormalizer:
    def __init__(self):
        """Initialize text normalizer"""
        self.stop_words = set(stopwords.words('english'))
    
    def to_lowercase(self, text):
        """Convert to lowercase"""
        return text.lower()
    
    def remove_punctuation(self, text):
        """Remove punctuation"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_numbers(self, text, keep_in_words=True):
        """Remove numbers"""
        if keep_in_words:
            return re.sub(r'\b\d+\b', '', text)
        return re.sub(r'\d+', '', text)
    
    def expand_contractions(self, text):
        """Expand contractions"""
        return contractions.fix(text)
    
    def remove_extra_whitespace(self, text):
        """Remove extra whitespace"""
        return re.sub(r'\s+', ' ', text.strip())
    
    def remove_stopwords(self, text):
        """Remove stopwords"""
        tokens = word_tokenize(text)
        return ' '.join([word for word in tokens if word.lower() not in self.stop_words])
    
    def remove_urls(self, text):
        """Remove URLs"""
        return re.sub(r'http[s]?://\S+', '', text)
    
    def remove_emails(self, text):
        """Remove email addresses"""
        return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    def remove_special_characters(self, text, keep_spaces=True):
        """Remove special characters"""
        if keep_spaces:
            return re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return re.sub(r'[^a-zA-Z0-9]', '', text)
    
    def normalize(self, text, steps=None):
        """
        Apply normalization pipeline
        
        Args:
            text: Input text
            steps: List of normalization steps to apply
                   Default: all steps
        """
        if steps is None:
            steps = [
                'expand_contractions',
                'to_lowercase',
                'remove_urls',
                'remove_emails',
                'remove_numbers',
                'remove_punctuation',
                'remove_extra_whitespace'
            ]
        
        for step in steps:
            if hasattr(self, step):
                text = getattr(self, step)(text)
        
        return text
    
    def normalize_corpus(self, documents, steps=None):
        """Normalize multiple documents"""
        return [self.normalize(doc, steps) for doc in documents]

# Example usage
normalizer = TextNormalizer()

text = """
I'm learning NLP!   It's AMAZING.
Visit http://example.com or email me@test.com
I have 123 apples and   456    oranges.
"""

print("Original:")
print(text)
print("\nNormalized:")
print(normalizer.normalize(text))

# Custom pipeline
custom_steps = ['to_lowercase', 'remove_urls', 'remove_punctuation']
print("\nCustom normalization:")
print(normalizer.normalize(text, steps=custom_steps))

# Normalize corpus
documents = [
    "I'm learning Python!",
    "It's AMAZING!!!",
    "Visit http://example.com"
]

normalized_docs = normalizer.normalize_corpus(documents)
print("\nNormalized corpus:")
for i, doc in enumerate(normalized_docs, 1):
    print(f"{i}. {doc}")
```

---

## 7. Stemming

### What is Stemming?

**Stemming** is the process of reducing words to their root/base form (stem) by removing affixes.

**Example:**
```
running → run
runs → run
ran → ran (not perfect!)
runner → runner (keeps 'ner')
```

### Key Characteristics

✅ Fast and simple  
✅ Rule-based approach  
✅ Works without dictionary  

❌ May not produce real words  
❌ Over-stemming (university → univers)  
❌ Under-stemming (data, datum not related)  
❌ Less accurate than lemmatization  

### Types of Stemmers in NLTK

1. **Porter Stemmer** (most common)
2. **Lancaster Stemmer** (aggressive)
3. **Snowball Stemmer** (improved Porter)
4. **Regexp Stemmer** (custom rules)

---

### Porter Stemmer

**Most widely used stemmer**

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = [
    'running', 'runs', 'ran', 'runner',
    'easily', 'fairly',
    'connection', 'connected', 'connecting',
    'generously', 'generate', 'generation'
]

print("Porter Stemmer:")
for word in words:
    stem = stemmer.stem(word)
    print(f"{word:15s} → {stem}")

# Output:
# running         → run
# runs            → run
# ran             → ran
# runner          → runner
# easily          → easili
# fairly          → fairli
# connection      → connect
# connected       → connect
# connecting      → connect
# generously      → generous
# generate        → generat
# generation      → generat
```

### Lancaster Stemmer

**Most aggressive stemmer**

```python
from nltk.stem import LancasterStemmer

stemmer = LancasterStemmer()

words = ['running', 'runs', 'ran', 'runner', 'fairly', 'sportingly']

print("Lancaster Stemmer:")
for word in words:
    stem = stemmer.stem(word)
    print(f"{word:15s} → {stem}")

# Output:
# running         → run
# runs            → run
# ran             → ran
# runner          → run  (more aggressive!)
# fairly          → fair
# sportingly      → sport
```

### Snowball Stemmer

**Improved version of Porter, supports multiple languages**

```python
from nltk.stem import SnowballStemmer

# Available languages
print("Supported languages:")
print(SnowballStemmer.languages)

# English stemmer
stemmer = SnowballStemmer('english')

words = ['running', 'runs', 'ran', 'runner', 'generously']

print("\nSnowball Stemmer (English):")
for word in words:
    stem = stemmer.stem(word)
    print(f"{word:15s} → {stem}")

# French example
french_stemmer = SnowballStemmer('french')
french_words = ['courante', 'courantes', 'courir']

print("\nSnowball Stemmer (French):")
for word in french_words:
    stem = french_stemmer.stem(word)
    print(f"{word:15s} → {stem}")
```

### Regexp Stemmer

**Create custom stemming rules**

```python
from nltk.stem import RegexpStemmer

# Remove 'ing' endings
stemmer = RegexpStemmer('ing$', min=4)

words = ['running', 'sing', 'walking', 'thing']

print("Regexp Stemmer (remove 'ing'):")
for word in words:
    stem = stemmer.stem(word)
    print(f"{word:15s} → {stem}")

# Output:
# running         → runn
# sing            → sing  (too short, min=4)
# walking         → walk
# thing           → thing (min=4)

# Multiple patterns
stemmer = RegexpStemmer('ing$|s$|ed$|ly$', min=4)

words = ['running', 'runs', 'walked', 'fairly']

print("\nRegexp Stemmer (multiple patterns):")
for word in words:
    stem = stemmer.stem(word)
    print(f"{word:15s} → {stem}")
```

### Comparing Stemmers

```python
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

def compare_stemmers(words):
    """Compare different stemmers"""
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer('english')
    
    print(f"{'Word':<15} {'Porter':<15} {'Lancaster':<15} {'Snowball':<15}")
    print("=" * 60)
    
    for word in words:
        p_stem = porter.stem(word)
        l_stem = lancaster.stem(word)
        s_stem = snowball.stem(word)
        print(f"{word:<15} {p_stem:<15} {l_stem:<15} {s_stem:<15}")

# Test words
words = [
    'running', 'runner', 'ran',
    'fairly', 'fairness',
    'connection', 'connected',
    'generously', 'generate',
    'sportingly', 'organization'
]

compare_stemmers(words)

# Output shows differences:
# Porter is moderate
# Lancaster is aggressive
# Snowball is similar to Porter but improved
```

### Stemming Text

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def stem_text(text, stemmer_type='porter'):
    """Stem all words in text"""
    # Choose stemmer
    if stemmer_type == 'porter':
        stemmer = PorterStemmer()
    elif stemmer_type == 'lancaster':
        stemmer = LancasterStemmer()
    elif stemmer_type == 'snowball':
        stemmer = SnowballStemmer('english')
    else:
        raise ValueError("Unknown stemmer type")
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Stem
    stemmed = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(stemmed)

# Example
text = "The runners were running and ran through the running track"

print("Original:", text)
print("Porter:", stem_text(text, 'porter'))
print("Lancaster:", stem_text(text, 'lancaster'))
print("Snowball:", stem_text(text, 'snowball'))

# Output:
# Original: The runners were running and ran through the running track
# Porter: the runner were run and ran through the run track
# Lancaster: the run wer run and ran through the run track
# Snowball: the runner were run and ran through the run track
```

### Problems with Stemming

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Problem 1: Over-stemming
over_stem_words = ['university', 'universe', 'universal']
print("Over-stemming problem:")
for word in over_stem_words:
    print(f"{word:15s} → {stemmer.stem(word)}")
# All become 'univers' (not a real word!)

# Problem 2: Under-stemming
under_stem_words = ['data', 'datum']
print("\nUnder-stemming problem:")
for word in under_stem_words:
    print(f"{word:15s} → {stemmer.stem(word)}")
# Remain different (should be same)

# Problem 3: Incorrect stems
incorrect_words = ['news', 'history', 'meeting']
print("\nIncorrect stems:")
for word in incorrect_words:
    print(f"{word:15s} → {stemmer.stem(word)}")
# news → new (wrong!)
# history → histori (not a word)
# meeting → meet (could be correct or not depending on context)

# Problem 4: Losing meaning
meaning_words = ['excellence', 'excellent', 'excel']
print("\nLosing meaning:")
for word in meaning_words:
    print(f"{word:15s} → {stemmer.stem(word)}")
# All become 'excel' but have different meanings!
```

### When to Use Stemming

**Use Stemming When:**
- Speed is critical
- Working with large datasets
- Approximate matching is acceptable
- Resource-constrained environments
- Information retrieval systems
- Search engines

**Avoid Stemming When:**
- Exact meaning matters
- Working with medical/legal text
- Translation tasks
- Sentiment analysis
- Named entity recognition

---

## 8. Lemmatization

### What is Lemmatization?

**Lemmatization** reduces words to their dictionary form (lemma) using vocabulary and morphological analysis.

**Example:**
```
running → run
runs → run
ran → run (better than stemming!)
better → good (considers meaning!)
```

### Stemming vs Lemmatization

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| **Output** | Root stem (may not be a word) | Dictionary lemma (always a word) |
| **Method** | Rule-based suffix removal | Vocabulary & morphological analysis |
| **Speed** | Fast | Slower |
| **Accuracy** | Lower | Higher |
| **Context** | No context | Uses POS tags |
| **Example** | caring → car | caring → care |

### Visual Comparison

```python
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = [
    'running', 'runs', 'ran',
    'better', 'good', 'best',
    'caring', 'cares', 'cared',
    'meeting', 'meetings',
    'leaves', 'leaving'
]

print(f"{'Word':<15} {'Stemming':<15} {'Lemmatization':<15}")
print("=" * 45)

for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word, pos='v')  # verb
    print(f"{word:<15} {stem:<15} {lemma:<15}")

# Output shows lemmatization produces real words
```

---

### WordNet Lemmatizer

**WordNet** is a large lexical database of English.

```python
from nltk.stem import WordNetLemmatizer
import nltk

# Download required data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Create lemmatizer
lemmatizer = WordNetLemmatizer()

words = ['running', 'runs', 'ran', 'better', 'cacti', 'geese']

print("Basic lemmatization:")
for word in words:
    lemma = lemmatizer.lemmatize(word)
    print(f"{word:15s} → {lemma}")

# Output:
# running         → running (wrong without POS!)
# runs            → run
# ran             → ran
# better          → better (wrong without POS!)
# cacti           → cactus
# geese           → goose
```

### Importance of POS Tags

**POS (Part of Speech)** tags are crucial for accurate lemmatization!

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Same word, different POS tags
word = "running"

print("Lemmatization with different POS tags:")
print(f"No POS:    {lemmatizer.lemmatize(word)}")
print(f"Noun (n):  {lemmatizer.lemmatize(word, pos='n')}")
print(f"Verb (v):  {lemmatizer.lemmatize(word, pos='v')}")
print(f"Adj (a):   {lemmatizer.lemmatize(word, pos='a')}")
print(f"Adv (r):   {lemmatizer.lemmatize(word, pos='r')}")

# Output:
# No POS:    running  (wrong!)
# Noun (n):  running
# Verb (v):  run      (correct!)
# Adj (a):   running
# Adv (r):   running

# Another example
word = "better"
print(f"\nWord: {word}")
print(f"No POS:    {lemmatizer.lemmatize(word)}")
print(f"Adj (a):   {lemmatizer.lemmatize(word, pos='a')}")  # good!
```

**POS Tags:**
- `n` = noun
- `v` = verb
- `a` = adjective
- `r` = adverb
- `s` = satellite adjective

### Automatic POS Tagging

```python
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tag to wordnet POS tag"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def lemmatize_text(text):
    """Lemmatize text with automatic POS tagging"""
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # POS tag
    pos_tags = pos_tag(tokens)
    
    # Lemmatize with POS
    lemmatized = []
    for word, tag in pos_tags:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word.lower(), pos=wn_tag)
        lemmatized.append(lemma)
    
    return lemmatized

# Example
text = "The leaves are leaving the trees. Better times are coming."

print("Original:", text)
print("Tokens:", word_tokenize(text))
print("POS tags:", pos_tag(word_tokenize(text)))
print("Lemmatized:", lemmatize_text(text))

# Output shows correct lemmatization:
# leaves (noun) → leaf
# leaving (verb) → leave
# Better (adj) → good
# coming (verb) → come
```

### Complete Lemmatization Class

```python
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import string

class Lemmatizer:
    def __init__(self, lowercase=True, remove_punctuation=True):
        """
        Initialize lemmatizer
        
        Args:
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
        """
        self.lemmatizer = WordNetLemmatizer()
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
    
    def get_wordnet_pos(self, treebank_tag):
        """Convert treebank POS to wordnet POS"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def lemmatize(self, text, return_tokens=False):
        """
        Lemmatize text
        
        Args:
            text: Input text
            return_tokens: Return list of tokens or joined string
        
        Returns:
            Lemmatized text or tokens
        """
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Tokenize
        tokens = word_token// filepath: /Users/mayankvashisht/Desktop/AI-ML/AI-ML/NLP/nltk_canvas.md

# NLTK (Natural Language Toolkit): Complete Guide from Basics to Advanced

A comprehensive guide covering NLTK fundamentals, tokenization, text normalization, stopwords, stemming, lemmatization, and practical applications with examples and interview questions.

---

## Table of Contents

1. [What is NLTK?](#1-what-is-nltk)
2. [Installation and Setup](#2-installation-and-setup)
3. [Tokenization](#3-tokenization)
4. [Regular Expressions in NLP](#4-regular-expressions-in-nlp)
5. [Stopwords](#5-stopwords)
6. [Text Normalization](#6-text-normalization)
7. [Stemming](#7-stemming)
8. [Lemmatization](#8-lemmatization)
9. [POS Tagging](#9-pos-tagging)
10. [Named Entity Recognition](#10-named-entity-recognition)
11. [Text Preprocessing Pipeline](#11-text-preprocessing-pipeline)
12. [Sparse vs Dense Representations](#12-sparse-vs-dense-representations)
13. [Interview Questions](#13-interview-questions)
14. [Practical Projects](#14-practical-projects)

---

## 1. What is NLTK?

### Overview

**NLTK (Natural Language Toolkit)** is a leading platform for building Python programs to work with human language data (NLP).

### Key Features

- **Tokenization:** Break text into words, sentences
- **Text Cleaning:** Remove stopwords, punctuation
- **Text Normalization:** Stemming, lemmatization
- **POS Tagging:** Part-of-speech identification
- **NER:** Named Entity Recognition
- **Parsing:** Syntactic analysis
- **Corpora Access:** 50+ corpora and lexical resources

### Why Use NLTK?

✅ Comprehensive library for NLP tasks  
✅ Easy to learn and use  
✅ Extensive documentation  
✅ Large community support  
✅ Perfect for learning and prototyping  
✅ Over 50+ corpora datasets  

❌ Slower than spaCy for production  
❌ Not optimized for deep learning  
❌ Requires manual downloads of data  

### NLTK vs Other Libraries

| Feature | NLTK | spaCy | Gensim | TextBlob |
|---------|------|-------|--------|----------|
| **Speed** | Slow | Fast | Medium | Slow |
| **Ease of Use** | Medium | Easy | Medium | Very Easy |
| **Production Ready** | No | Yes | Yes | No |
| **Deep Learning** | No | Yes | No | No |
| **Learning Curve** | Steep | Gentle | Medium | Very Gentle |
| **Best For** | Research, Learning | Production | Topic Modeling | Quick Prototyping |

---

## 2. Installation and Setup

### Basic Installation

```bash
# Install NLTK
pip install nltk

# For Jupyter notebooks
!pip install nltk
```

### Downloading NLTK Data

NLTK requires additional data packages to be downloaded separately.

```python
import nltk

# Download all packages (not recommended - large!)
nltk.download('all')

# Download specific packages (recommended)
nltk.download('punkt')        # Tokenizer models
nltk.download('punkt_tab')    # Additional tokenizer data
nltk.download('stopwords')    # Stopwords lists
nltk.download('wordnet')      # WordNet lexical database
nltk.download('averaged_perceptron_tagger')  # POS tagger
nltk.download('maxent_ne_chunker')  # Named entity chunker
nltk.download('words')        # Word lists

# Download multiple at once
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
```

### What is 'punkt'?

**punkt** is a pre-trained tokenizer model that uses unsupervised learning to identify sentence boundaries.

**Key Features:**
- Trained on multiple languages
- Handles abbreviations (Dr., Mr., etc.)
- Recognizes decimal numbers
- Understands sentence boundaries

**Example:**

```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Dr. Smith works at U.S.A. He earns $1,000.50 daily."

# Without punkt - would split incorrectly
# With punkt - handles abbreviations correctly
sentences = sent_tokenize(text)
print(sentences)
# Output: ['Dr. Smith works at U.S.A.', 'He earns $1,000.50 daily.']
```

### Setting Up NLTK Data Path

```python
import nltk
import os

# Check current data path
print(nltk.data.path)

# Add custom data path
custom_path = '/path/to/nltk_data'
if custom_path not in nltk.data.path:
    nltk.data.path.append(custom_path)

# Set environment variable (permanent)
os.environ['NLTK_DATA'] = custom_path
```

### Complete Setup Script

```python
import nltk
import sys

def setup_nltk():
    """Complete NLTK setup with error handling"""
    
    required_packages = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'brown',
        'names'
    ]
    
    print("Setting up NLTK...")
    print("=" * 50)
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
            print(f"✓ {package} already installed")
        except LookupError:
            try:
                print(f"⬇ Downloading {package}...")
                nltk.download(package, quiet=True)
                print(f"✓ {package} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {package}: {e}")
                sys.exit(1)
    
    print("=" * 50)
    print("✓ NLTK setup complete!")
    print(f"NLTK version: {nltk.__version__}")
    print(f"Data path: {nltk.data.path[0]}")

# Run setup
setup_nltk()
```

---

## 3. Tokenization

### What is Tokenization?

**Tokenization** is the process of breaking text into smaller units called **tokens**.

**Why is it Important?**
- First step in NLP pipeline
- Converts text into processable units
- Essential for feature extraction
- Required for most NLP tasks

### Types of Tokenization

1. **Word Tokenization:** Split text into words
2. **Sentence Tokenization:** Split text into sentences
3. **Character Tokenization:** Split text into characters
4. **Subword Tokenization:** Split into subword units (BPE, WordPiece)

---

### Word Tokenization

#### Method 1: Python Split (Basic)

```python
text = "I am learning NLP!"

# Basic split
tokens = text.split()
print(tokens)
# Output: ['I', 'am', 'learning', 'NLP!']

# Problems:
# ❌ Punctuation attached to words
# ❌ Can't handle multiple separators
# ❌ No special case handling
```

**Limitations:**
- Only one separator at a time
- Punctuation not separated
- No handling of special cases

#### Method 2: NLTK Word Tokenize (Recommended)

```python
from nltk.tokenize import word_tokenize

text = "I'm learning NLP! It's amazing."

tokens = word_tokenize(text)
print(tokens)
# Output: ['I', "'m", 'learning', 'NLP', '!', 'It', "'s", 'amazing', '.']
```

**Advantages:**
✅ Separates punctuation  
✅ Handles contractions  
✅ Language-aware  
✅ Handles special characters  

**Note:** Punctuation is treated as separate tokens!

#### Method 3: Regular Expressions

```python
import re

text = "I'm learning NLP! It's amazing."

# Pattern 1: Alphanumeric and apostrophes
tokens = re.findall(r"[\w']+", text)
print("Pattern 1:", tokens)
# Output: ["I'm", 'learning', 'NLP', "It's", 'amazing']

# Pattern 2: Words only
tokens = re.findall(r'\b\w+\b', text)
print("Pattern 2:", tokens)
# Output: ['I', 'm', 'learning', 'NLP', 'It', 's', 'amazing']

# Pattern 3: Custom pattern
tokens = re.findall(r'\b[a-zA-Z]+\b', text)
print("Pattern 3:", tokens)
# Output: ['I', 'm', 'learning', 'NLP', 'It', 's', 'amazing']
```

#### Method 4: TreebankWordTokenizer

```python
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

text = "She said, \"It's a beautiful day!\""

tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['She', 'said', ',', '``', 'It', "'s", 'a', 'beautiful', 'day', '!', "''"]
```

**Advantages:**
- Handles quotes properly
- Separates punctuation
- Used in Penn Treebank

#### Method 5: WordPunctTokenizer

```python
from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()

text = "I'm learning NLP! Email: test@email.com"

tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['I', "'", 'm', 'learning', 'NLP', '!', 'Email', ':', 'test', '@', 'email', '.', 'com']
```

**Characteristics:**
- Splits all punctuation
- Very aggressive tokenization
- Useful for specific tasks

### Complete Word Tokenization Example

```python
import re
from nltk.tokenize import word_tokenize, WordPunctTokenizer, TreebankWordTokenizer

def compare_tokenizers(text):
    """Compare different tokenization methods"""
    
    print(f"Original text: {text}\n")
    print("=" * 60)
    
    # Method 1: Python split
    tokens_split = text.split()
    print(f"1. Python split(): {tokens_split}")
    print(f"   Token count: {len(tokens_split)}\n")
    
    # Method 2: NLTK word_tokenize
    tokens_nltk = word_tokenize(text)
    print(f"2. NLTK word_tokenize(): {tokens_nltk}")
    print(f"   Token count: {len(tokens_nltk)}\n")
    
    # Method 3: Regex
    tokens_regex = re.findall(r"[\w']+", text)
    print(f"3. Regex: {tokens_regex}")
    print(f"   Token count: {len(tokens_regex)}\n")
    
    # Method 4: TreebankWordTokenizer
    treebank = TreebankWordTokenizer()
    tokens_treebank = treebank.tokenize(text)
    print(f"4. TreebankWordTokenizer: {tokens_treebank}")
    print(f"   Token count: {len(tokens_treebank)}\n")
    
    # Method 5: WordPunctTokenizer
    wordpunct = WordPunctTokenizer()
    tokens_wordpunct = wordpunct.tokenize(text)
    print(f"5. WordPunctTokenizer: {tokens_wordpunct}")
    print(f"   Token count: {len(tokens_wordpunct)}\n")

# Test
text = "I'm learning NLP! Email me at test@email.com. It's amazing!"
compare_tokenizers(text)
```

---

### Sentence Tokenization

#### Method 1: Python Split (Basic)

```python
text = "Hello world. This is NLP. It's amazing."

# Split by period
sentences = text.split('. ')
print(sentences)
# Output: ['Hello world', 'This is NLP', "It's amazing."]

# Problems:
# ❌ Last sentence has period
# ❌ Fails with abbreviations (Dr., Mr.)
# ❌ Can't handle multiple punctuation
```

#### Method 2: Regular Expressions

```python
import re

text = "Hello world. This is NLP! Is it amazing? Yes."

# Split by multiple punctuation
sentences = re.split(r'[.!?] ', text)
print(sentences)
# Output: ['Hello world', 'This is NLP', 'Is it amazing', 'Yes.']

# Better regex
sentences = re.compile('[.!?] ').split(text)
print(sentences)
```

**Advantages over split():**
✅ Multiple separators  
✅ More flexible  

**Limitations:**
❌ Still fails with abbreviations  
❌ No context awareness  

#### Method 3: NLTK Sent Tokenize (Recommended)

```python
from nltk.tokenize import sent_tokenize

text = "Dr. A.P.J. Abdul Kalam was the Former President of India. He was born on Oct. 15, 1931. He passed away in 2015."

sentences = sent_tokenize(text)
print(sentences)
# Output: [
#   'Dr. A.P.J. Abdul Kalam was the Former President of India.',
#   'He was born on Oct. 15, 1931.',
#   'He passed away in 2015.'
# ]
```

**Advantages:**
✅ Handles abbreviations (Dr., Mr., Oct.)  
✅ Context-aware  
✅ Trained on multiple languages  
✅ Handles decimals (1.5, 2.3)  
✅ Multi-language support  

#### Comparison: Split vs Regex vs NLTK

```python
import re
from nltk.tokenize import sent_tokenize

def compare_sentence_tokenizers(text):
    """Compare sentence tokenization methods"""
    
    print(f"Original text:\n{text}\n")
    print("=" * 60)
    
    # Method 1: Split
    split_sentences = text.split('. ')
    print("1. Python split():")
    for i, sent in enumerate(split_sentences, 1):
        print(f"   {i}. {sent}")
    print()
    
    # Method 2: Regex
    regex_sentences = re.compile('[.!?] ').split(text)
    print("2. Regular Expression:")
    for i, sent in enumerate(regex_sentences, 1):
        print(f"   {i}. {sent}")
    print()
    
    # Method 3: NLTK
    nltk_sentences = sent_tokenize(text)
    print("3. NLTK sent_tokenize():")
    for i, sent in enumerate(nltk_sentences, 1):
        print(f"   {i}. {sent}")

# Test with tricky text
text = "Dr. Smith works at U.S.A. He earns $1,000.50 daily! Isn't that great? Yes."
compare_sentence_tokenizers(text)
```

**Output Analysis:**

```
1. Python split():
   ❌ "Dr" separated from "Smith"
   ❌ "U.S.A" broken incorrectly
   
2. Regular Expression:
   ❌ Same issues as split
   ❌ Slightly better with multiple punctuation
   
3. NLTK sent_tokenize():
   ✅ Correctly handles all cases
   ✅ Keeps "Dr. Smith" together
   ✅ Handles "U.S.A." correctly
   ✅ Preserves "$1,000.50"
```

### Advanced Tokenization

#### Custom Tokenizer

```python
from nltk.tokenize.regexp import RegexpTokenizer

# Create custom tokenizer
tokenizer = RegexpTokenizer(r'\w+')

text = "Hello! This is a test. Email: test@email.com"
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['Hello', 'This', 'is', 'a', 'test', 'Email', 'test', 'email', 'com']
```

#### Whitespace Tokenizer

```python
from nltk.tokenize import WhitespaceTokenizer

tokenizer = WhitespaceTokenizer()

text = "Hello   world\n\tThis is   NLP"
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['Hello', 'world', 'This', 'is', 'NLP']
```

#### Tweet Tokenizer (Social Media)

```python
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer(
    preserve_case=False,    # Convert to lowercase
    reduce_len=True,        # Reduce repeated chars ('hellooo' -> 'helloo')
    strip_handles=True      # Remove @mentions
)

tweet = "@user This is AMAZING!!! Check out #NLP http://example.com 😊"

tokens = tokenizer.tokenize(tweet)
print(tokens)
# Output: ['this', 'is', 'amazing', '!', 'check', 'out', '#nlp', 'http://example.com', '😊']
```

#### Multi-Word Expression (MWE) Tokenizer

```python
from nltk.tokenize import MWETokenizer

# Create tokenizer with multi-word expressions
tokenizer = MWETokenizer([
    ('New', 'York'),
    ('machine', 'learning'),
    ('natural', 'language', 'processing')
])

text = "I'm studying machine learning and natural language processing in New York"

tokens = word_tokenize(text)
tokens = tokenizer.tokenize(tokens)
print(tokens)
# Output: ['I', "'m", 'studying', 'machine_learning', 'and', 
#          'natural_language_processing', 'in', 'New_York']
```

---

## 4. Regular Expressions in NLP

### Why Regular Expressions?

Regular expressions (regex) are powerful patterns for matching text.

**Use Cases in NLP:**
- Custom tokenization
- Email/URL extraction
- Pattern matching
- Text cleaning
- Data validation

### Basic Regex Patterns

```python
import re

# Common patterns
patterns = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'hashtag': r'#\w+',
    'mention': r'@\w+',
    'number': r'\b\d+\.?\d*\b',
    'word': r'\b[a-zA-Z]+\b'
}

text = """
Contact: john.doe@email.com
Visit: https://example.com
Call: 123-456-7890
Tweet: #NLP @user
Price: $99.99
"""

# Extract patterns
for pattern_name, pattern in patterns.items():
    matches = re.findall(pattern, text)
    if matches:
        print(f"{pattern_name}: {matches}")
```

### Email Address Extraction

```python
import re

def extract_emails(text):
    """Extract email addresses from text"""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, text)
    return emails

text = """
Contact us:
- Sales: sales@company.com
- Support: support@company.com
- Info: info@company.co.uk
Invalid: test@, @test.com, test
"""

emails = extract_emails(text)
print(f"Found emails: {emails}")
# Output: ['sales@company.com', 'support@company.com', 'info@company.co.uk']
```

**Pattern Breakdown:**
```
^([a-zA-Z0-9_\-\.]+)  → Username part
@                      → @ symbol
([a-zA-Z0-9_\-\.]+)   → Domain name
\.                     → Dot
([a-zA-Z]{2,5})$      → TLD (com, org, etc.)
```

### URL Extraction and Cleaning

```python
import re

def extract_urls(text):
    """Extract URLs from text"""
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(pattern, text)
    return urls

def remove_urls(text):
    """Remove URLs from text"""
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(pattern, '', text)

text = """
Check out my blog: https://example.com
Download: http://files.example.com/file.pdf
Visit: www.example.com
"""

print("URLs found:", extract_urls(text))
print("Text without URLs:", remove_urls(text))
```

### Social Media Pattern Extraction

```python
import re

def extract_social_patterns(text):
    """Extract hashtags, mentions, and emojis"""
    
    patterns = {
        'hashtags': r'#\w+',
        'mentions': r'@\w+',
        'emojis': r'[\U0001F600-\U0001F64F]',  # Emoticons
        'urls': r'http[s]?://\S+',
        'numbers': r'\b\d+\.?\d*\b'
    }
    
    results = {}
    for name, pattern in patterns.items():
        results[name] = re.findall(pattern, text)
    
    return results

tweet = """
@user Just finished reading about #NLP and #MachineLearning 🎉
Check it out: https://example.com
Score: 95.5% 😊
"""

patterns = extract_social_patterns(tweet)
for pattern_name, matches in patterns.items():
    print(f"{pattern_name}: {matches}")
```

### Text Cleaning with Regex

```python
import re

class RegexCleaner:
    def __init__(self):
        """Initialize text cleaner with regex patterns"""
        self.patterns = {
            'url': r'http[s]?://\S+',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'mention': r'@\w+',
            'hashtag': r'#\w+',
            'number': r'\b\d+\.?\d*\b',
            'punctuation': r'[^\w\s]',
            'extra_spaces': r'\s+',
            'html_tags': r'<[^>]+>',
            'special_chars': r'[^a-zA-Z0-9\s]'
        }
    
    def remove_pattern(self, text, pattern_name):
        """Remove specific pattern from text"""
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
            return re.sub(pattern, '', text)
        return text
    
    def clean_text(self, text, remove=['url', 'email', 'html_tags']):
        """Clean text by removing specified patterns"""
        cleaned = text
        for pattern_name in remove:
            cleaned = self.remove_pattern(cleaned, pattern_name)
        
        # Remove extra spaces
        cleaned = re.sub(self.patterns['extra_spaces'], ' ', cleaned)
        return cleaned.strip()
    
    def replace_pattern(self, text, pattern_name, replacement=''):
        """Replace pattern with specified replacement"""
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
            return re.sub(pattern, replacement, text)
        return text

# Example usage
cleaner = RegexCleaner()

text = """
<h1>Title</h1>
Check out https://example.com
Contact: test@email.com
Follow @user and use #NLP
Price: $99.99
"""

# Clean text
cleaned = cleaner.clean_text(text, remove=['url', 'email', 'html_tags'])
print("Cleaned:", cleaned)

# Remove specific patterns
no_mentions = cleaner.remove_pattern(text, 'mention')
print("No mentions:", no_mentions)

# Replace numbers
no_numbers = cleaner.replace_pattern(text, 'number', '[NUMBER]')
print("Numbers replaced:", no_numbers)
```

### Advanced Regex Examples

```python
import re

# 1. Extract dates
def extract_dates(text):
    """Extract dates in various formats"""
    patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',          # MM/DD/YYYY
        r'\d{1,2}-\d{1,2}-\d{2,4}',          # MM-DD-YYYY
        r'\d{4}-\d{1,2}-\d{1,2}',            # YYYY-MM-DD
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'  # 15 Jan 2023
    ]
    
    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    return dates

# 2. Extract phone numbers
def extract_phone_numbers(text):
    """Extract phone numbers in various formats"""
    patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',    # 123-456-7890
        r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',      # (123) 456-7890
        r'\+\d{1,3}\s*\d{3}[-.]?\d{3}[-.]?\d{4}'  # +1 123-456-7890
    ]
    
    phones = []
    for pattern in patterns:
        phones.extend(re.findall(pattern, text))
    return phones

# 3. Extract prices
def extract_prices(text):
    """Extract prices from text"""
    pattern = r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?'
    prices = re.findall(pattern, text)
    return prices

# 4. Validate input
def validate_input(input_str, input_type):
    """Validate different types of input"""
    patterns = {
        'email': r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$',
        'phone': r'^\d{3}[-.]?\d{3}[-.]?\d{4}$',
        'zipcode': r'^\d{5}(-\d{4})?$',
        'username': r'^[a-zA-Z0-9_]{3,20}$',
        'password': r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$'
    }
    
    if input_type in patterns:
        return bool(re.match(patterns[input_type], input_str))
    return False

# Test examples
text = """
Dates: 12/31/2023, 2023-12-31, 15 Jan 2023
Phones: 123-456-7890, (123) 456-7890, +1 123-456-7890
Prices: $19.99, $1,234.56, $99
"""

print("Dates:", extract_dates(text))
print("Phones:", extract_phone_numbers(text))
print("Prices:", extract_prices(text))

# Validation
print("\nValidation:")
print("Email valid:", validate_input("test@email.com", "email"))
print("Phone valid:", validate_input("123-456-7890", "phone"))
print("Username valid:", validate_input("user_123", "username"))
```

---

## 5. Stopwords

### What are Stopwords?

**Stopwords** are common words that don't carry much meaning and are often filtered out in NLP tasks.

**Examples:** the, is, at, which, on, a, an, and, or, but

### Why Remove Stopwords?

✅ Reduce dimensionality  
✅ Reduce noise  
✅ Improve processing speed  
✅ Focus on meaningful words  
✅ Better model performance  

❌ Loss of context in some tasks  
❌ Can affect sentiment analysis  
❌ Important for some NLP tasks  

### When to Remove Stopwords?

**Remove:**
- Text classification
- Information retrieval
- Topic modeling
- Keyword extraction

**Keep:**
- Sentiment analysis ("not good" vs "good")
- Question answering
- Machine translation
- Named entity recognition

### NLTK Stopwords

```python
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')

# Get English stopwords
stop_words = set(stopwords.words('english'))

print(f"Total stopwords: {len(stop_words)}")
print(f"Sample stopwords: {list(stop_words)[:20]}")

# Check if word is stopword
print("'the' is stopword:", 'the' in stop_words)
print("'python' is stopword:", 'python' in stop_words)
```

### Removing Stopwords

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(text, custom_stopwords=None):
    """Remove stopwords from text"""
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Add custom stopwords
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Filter stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return filtered_tokens

# Example
text = "This is an example sentence showing the removal of stopwords"

print("Original:", text)
print("Tokens:", word_tokenize(text))
print("Filtered:", remove_stopwords(text))

# Output:
# Original: This is an example sentence showing the removal of stopwords
# Tokens: ['This', 'is', 'an', 'example', 'sentence', 'showing', 'the', 'removal', 'of', 'stopwords']
# Filtered: ['example', 'sentence', 'showing', 'removal', 'stopwords']
```

### Custom Stopwords

```python
from nltk.corpus import stopwords

# Get default stopwords
stop_words = set(stopwords.words('english'))

# Add custom stopwords
custom_stops = ['said', 'says', 'would', 'could', 'might']
stop_words.update(custom_stops)

# Remove specific stopwords
stop_words.discard('not')  # Keep 'not' for sentiment
stop_words.discard('no')

print(f"Total stopwords: {len(stop_words)}")

def remove_stopwords_custom(text, keep_words=None):
    """Remove stopwords but keep specific words"""
    stop_words = set(stopwords.words('english'))
    
    # Remove words we want to keep
    if keep_words:
        for word in keep_words:
            stop_words.discard(word)
    
    tokens = word_tokenize(text.lower())
    filtered = [word for word in tokens if word not in stop_words]
    
    return filtered

# Keep negations for sentiment analysis
text = "This is not good but not bad either"
filtered = remove_stopwords_custom(text, keep_words=['not'])
print(filtered)
# Output: ['not', 'good', 'not', 'bad', 'either']
```

### Stopwords in Multiple Languages

```python
from nltk.corpus import stopwords

# Available languages
print("Available languages:")
print(stopwords.fileids())

# Get stopwords for different languages
languages = ['english', 'spanish', 'french', 'german']

for lang in languages:
    stop_words = stopwords.words(lang)
    print(f"\n{lang.capitalize()}: {len(stop_words)} stopwords")
    print(f"Sample: {stop_words[:10]}")

# Example: Multilingual stopword removal
def remove_stopwords_multilang(text, language='english'):
    """Remove stopwords for specified language"""
    from nltk.tokenize import word_tokenize
    
    try:
        stop_words = set(stopwords.words(language))
        tokens = word_tokenize(text.lower())
        filtered = [word for word in tokens if word not in stop_words]
        return filtered
    except OSError:
        print(f"Stopwords for '{language}' not found")
        return None

# Test
spanish_text = "Este es un ejemplo de texto en español"
filtered_spanish = remove_stopwords_multilang(spanish_text, 'spanish')
print(f"\nSpanish example: {filtered_spanish}")
```

### Complete Stopword Handler

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

class StopwordHandler:
    def __init__(self, language='english', custom_stopwords=None, 
                 keep_words=None, remove_punctuation=True):
        """
        Initialize stopword handler
        
        Args:
            language: Language for stopwords
            custom_stopwords: Additional stopwords to add
            keep_words: Words to keep (remove from stopwords)
            remove_punctuation: Whether to remove punctuation
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.remove_punctuation = remove_punctuation
        
        # Add custom stopwords
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # Remove words to keep
        if keep_words:
            for word in keep_words:
                self.stop_words.discard(word)
        
        # Add punctuation to stopwords if needed
        if remove_punctuation:
            self.stop_words.update(set(string.punctuation))
    
    def remove(self, text, lowercase=True):
        """Remove stopwords from text"""
        if lowercase:
            text = text.lower()
        
        tokens = word_tokenize(text)
        filtered = [word for word in tokens if word not in self.stop_words]
        
        return filtered
    
    def get_stopwords(self):
        """Get current stopwords"""
        return self.stop_words
    
    def add_stopwords(self, words):
        """Add new stopwords"""
        if isinstance(words, str):
            words = [words]
        self.stop_words.update(words)
    
    def remove_stopwords(self, words):
        """Remove words from stopwords list"""
        if isinstance(words, str):
            words = [words]
        for word in words:
            self.stop_words.discard(word)
    
    def filter_corpus(self, documents):
        """Filter stopwords from multiple documents"""
        return [self.remove(doc) for doc in documents]

# Example usage
handler = StopwordHandler(
    language='english',
    custom_stopwords=['example', 'test'],
    keep_words=['not', 'no'],  # Keep for sentiment
    remove_punctuation=True
)

text = "This is not a test example! It's important."
filtered = handler.remove(text)
print("Filtered:", filtered)

# Add more stopwords
handler.add_stopwords(['important', 'example'])
filtered2 = handler.remove(text)
print("After adding stopwords:", filtered2)

# Process multiple documents
documents = [
    "This is the first document",
    "This is the second document",
    "And this is the third one"
]
filtered_docs = handler.filter_corpus(documents)
print("\nFiltered documents:")
for i, doc in enumerate(filtered_docs, 1):
    print(f"{i}. {doc}")
```

---

## 6. Text Normalization

### What is Text Normalization?

**Text Normalization** is the process of transforming text into a canonical (standard) form.

**Goals:**
- Reduce variations
- Standardize format
- Improve consistency
- Reduce vocabulary size

### Types of Normalization

1. **Case Normalization:** Convert to lowercase/uppercase
2. **Removing Punctuation:** Strip special characters
3. **Removing Numbers:** Filter numeric values
4. **Expanding Contractions:** "don't" → "do not"
5. **Removing Extra Whitespace:** Clean spacing
6. **Unicode Normalization:** Handle special characters
7. **Stemming:** Reduce words to root form
8. **Lemmatization:** Reduce words to dictionary form

### Case Normalization

```python
text = "Python is AWESOME! I LOVE Python."

# Lowercase (most common)
lowercase = text.lower()
print("Lowercase:", lowercase)
# Output: python is awesome! i love python.

# Uppercase
uppercase = text.upper()
print("Uppercase:", uppercase)
# Output: PYTHON IS AWESOME! I LOVE PYTHON.

# Title case
titlecase = text.title()
print("Title:", titlecase)
# Output: Python Is Awesome! I Love Python.

# Capitalize (first letter only)
capitalize = text.capitalize()
print("Capitalize:", capitalize)
# Output: Python is awesome! i love python.
```

**When to use:**
- **Lowercase:** Most NLP tasks (default)
- **Preserve case:** Named Entity Recognition, POS tagging
- **Title case:** Display purposes

### Removing Punctuation

```python
import string
from nltk.tokenize import word_tokenize

def remove_punctuation(text):
    """Remove punctuation from text"""
    # Method 1: Using translate
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_punctuation_tokens(text):
    """Remove punctuation after tokenization"""
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in string.punctuation]

# Example
text = "Hello, world! This is amazing... Right?"

print("Original:", text)
print("Method 1:", remove_punctuation(text))
print("Method 2:", remove_punctuation_tokens(text))

# Output:
# Original: Hello, world! This is amazing... Right?
# Method 1: Hello world This is amazing Right
# Method 2: ['Hello', 'world', 'This', 'is', 'amazing', 'Right']
```

### Removing Numbers

```python
import re

def remove_numbers(text):
    """Remove numbers from text"""
    return re.sub(r'\d+', '', text)

def remove_numbers_keep_text(text):
    """Remove standalone numbers but keep numbers in words"""
    return re.sub(r'\b\d+\b', '', text)

# Example
text = "I have 3 apples and 2 oranges. Python3 is great!"

print("Original:", text)
print("Remove all:", remove_numbers(text))
print("Remove standalone:", remove_numbers_keep_text(text))

# Output:
# Original: I have 3 apples and 2 oranges. Python3 is great!
# Remove all: I have  apples and  oranges. Python is great!
# Remove standalone: I have  apples and  oranges. Python3 is great!
```

### Expanding Contractions

```python
import contractions

def expand_contractions(text):
    """Expand contractions in text"""
    return contractions.fix(text)

# Manual contractions dictionary
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "I'm": "I am",
    "I've": "I have",
    "I'll": "I will",
    "you're": "you are",
    "you've": "you have",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is"
}

def expand_contractions_manual(text):
    """Expand contractions using manual dictionary"""
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text

# Example
text = "I'm learning NLP. It's amazing! I can't wait to learn more."

print("Original:", text)
print("Using contractions lib:", expand_contractions(text))
print("Using manual dict:", expand_contractions_manual(text))
```

### Removing Extra Whitespace

```python
import re

def remove_extra_whitespace(text):
    """Remove extra whitespace"""
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text

# Example
text = "  This   is   a    text  with   extra   spaces.  "

print("Original:", repr(text))
print("Cleaned:", repr(remove_extra_whitespace(text)))

# Output:
# Original: '  This   is   a    text  with   extra   spaces.  '
# Cleaned: 'This is a text with extra spaces.'
```

### Complete Text Normalizer

```python
import re
import string
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class TextNormalizer:
    def __init__(self):
        """Initialize text normalizer"""
        self.stop_words = set(stopwords.words('english'))
    
    def to_lowercase(self, text):
        """Convert to lowercase"""
        return text.lower()
    
    def remove_punctuation(self, text):
        """Remove punctuation"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_numbers(self, text, keep_in_words=True):
        """Remove numbers"""
        if keep_in_words:
            return re.sub(r'\b\d+\b', '', text)
        return re.sub(r'\d+', '', text)
    
    def expand_contractions(self, text):
        """Expand contractions"""
        return contractions.fix(text)
    
    def remove_extra_whitespace(self, text):
        """Remove extra whitespace"""
        return re.sub(r'\s+', ' ', text.strip())
    
    def remove_stopwords(self, text):
        """Remove stopwords"""
        tokens = word_tokenize(text)
        return ' '.join([word for word in tokens if word.lower() not in self.stop_words])
    
    def remove_urls(self, text):
        """Remove URLs"""
        return re.sub(r'http[s]?://\S+', '', text)
    
    def remove_emails(self, text):
        """Remove email addresses"""
        return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    def remove_special_characters(self, text, keep_spaces=True):
        """Remove special characters"""
        if keep_spaces:
            return re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return re.sub(r'[^a-zA-Z0-9]', '', text)
    
    def normalize(self, text, steps=None):
        """
        Apply normalization pipeline
        
        Args:
            text: Input text
            steps: List of normalization steps to apply
                   Default: all steps
        """
        if steps is None:
            steps = [
                'expand_contractions',
                'to_lowercase',
                'remove_urls',
                'remove_emails',
                'remove_numbers',
                'remove_punctuation',
                'remove_extra_whitespace'
            ]
        
        for step in steps:
            if hasattr(self, step):
                text = getattr(self, step)(text)
        
        return text
    
    def normalize_corpus(self, documents, steps=None):
        """Normalize multiple documents"""
        return [self.normalize(doc, steps) for doc in documents]

# Example usage
normalizer = TextNormalizer()

text = """
I'm learning NLP!   It's AMAZING.
Visit http://example.com or email me@test.com
I have 123 apples and   456    oranges.
"""

print("Original:")
print(text)
print("\nNormalized:")
print(normalizer.normalize(text))

# Custom pipeline
custom_steps = ['to_lowercase', 'remove_urls', 'remove_punctuation']
print("\nCustom normalization:")
print(normalizer.normalize(text, steps=custom_steps))

# Normalize corpus
documents = [
    "I'm learning Python!",
    "It's AMAZING!!!",
    "Visit http://example.com"
]

normalized_docs = normalizer.normalize_corpus(documents)
print("\nNormalized corpus:")
for i, doc in enumerate(normalized_docs, 1):
    print(f"{i}. {doc}")
```

---

## 7. Stemming

### What is Stemming?

**Stemming** is the process of reducing words to their root/base form (stem) by removing affixes.

**Example:**
```
running → run
runs → run
ran → ran (not perfect!)
runner → runner (keeps 'ner')
```

### Key Characteristics

✅ Fast and simple  
✅ Rule-based approach  
✅ Works without dictionary  

❌ May not produce real words  
❌ Over-stemming (university → univers)  
❌ Under-stemming (data, datum not related)  
❌ Less accurate than lemmatization  

### Types of Stemmers in NLTK

1. **Porter Stemmer** (most common)
2. **Lancaster Stemmer** (aggressive)
3. **Snowball Stemmer** (improved Porter)
4. **Regexp Stemmer** (custom rules)

---

### Porter Stemmer

**Most widely used stemmer**

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = [
    'running', 'runs', 'ran', 'runner',
    'easily', 'fairly',
    'connection', 'connected', 'connecting',
    'generously', 'generate', 'generation'
]

print("Porter Stemmer:")
for word in words:
    stem = stemmer.stem(word)
    print(f"{word:15s} → {stem}")

# Output:
# running         → run
# runs            → run
# ran             → ran
# runner          → runner
# easily          → easili
# fairly          → fairli
# connection      → connect
# connected       → connect
# connecting      → connect
# generously      → generous
# generate        → generat
# generation      → generat
```

### Lancaster Stemmer

**Most aggressive stemmer**

```python
from nltk.stem import LancasterStemmer

stemmer = LancasterStemmer()

words = ['running', 'runs', 'ran', 'runner', 'fairly', 'sportingly']

print("Lancaster Stemmer:")
for word in words:
    stem = stemmer.stem(word)
    print(f"{word:15s} → {stem}")

# Output:
# running         → run
# runs            → run
# ran             → ran
# runner          → run  (more aggressive!)
# fairly          → fair
# sportingly      → sport
```

### Snowball Stemmer

**Improved version of Porter, supports multiple languages**

```python
from nltk.stem import SnowballStemmer

# Available languages
print("Supported languages:")
print(SnowballStemmer.languages)

# English stemmer
stemmer = SnowballStemmer('english')

words = ['running', 'runs', 'ran', 'runner', 'generously']

print("\nSnowball Stemmer (English):")
for word in words:
    stem = stemmer.stem(word)
    print(f"{word:15s} → {stem}")

# French example
french_stemmer = SnowballStemmer('french')
french_words = ['courante', 'courantes', 'courir']

print("\nSnowball Stemmer (French):")
for word in french_words:
    stem = french_stemmer.stem(word)
    print(f"{word:15s} → {stem}")
```

### Regexp Stemmer

**Create custom stemming rules**

```python
from nltk.stem import RegexpStemmer

# Remove 'ing' endings
stemmer = RegexpStemmer('ing$', min=4)

words = ['running', 'sing', 'walking', 'thing']

print("Regexp Stemmer (remove 'ing'):")
for word in words:
    stem = stemmer.stem(word)
    print(f"{word:15s} → {stem}")

# Output:
# running         → runn
# sing            → sing  (too short, min=4)
# walking         → walk
# thing           → thing (min=4)

# Multiple patterns
stemmer = RegexpStemmer('ing$|s$|ed$|ly$', min=4)

words = ['running', 'runs', 'walked', 'fairly']

print("\nRegexp Stemmer (multiple patterns):")
for word in words:
    stem = stemmer.stem(word)
    print(f"{word:15s} → {stem}")
```

### Comparing Stemmers

```python
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

def compare_stemmers(words):
    """Compare different stemmers"""
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    snowball = SnowballStemmer('english')
    
    print(f"{'Word':<15} {'Porter':<15} {'Lancaster':<15} {'Snowball':<15}")
    print("=" * 60)
    
    for word in words:
        p_stem = porter.stem(word)
        l_stem = lancaster.stem(word)
        s_stem = snowball.stem(word)
        print(f"{word:<15} {p_stem:<15} {l_stem:<15} {s_stem:<15}")

# Test words
words = [
    'running', 'runner', 'ran',
    'fairly', 'fairness',
    'connection', 'connected',
    'generously', 'generate',
    'sportingly', 'organization'
]

compare_stemmers(words)

# Output shows differences:
# Porter is moderate
# Lancaster is aggressive
# Snowball is similar to Porter but improved
```

### Stemming Text

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def stem_text(text, stemmer_type='porter'):
    """Stem all words in text"""
    # Choose stemmer
    if stemmer_type == 'porter':
        stemmer = PorterStemmer()
    elif stemmer_type == 'lancaster':
        stemmer = LancasterStemmer()
    elif stemmer_type == 'snowball':
        stemmer = SnowballStemmer('english')
    else:
        raise ValueError("Unknown stemmer type")
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Stem
    stemmed = [stemmer.stem(token) for token in tokens]
    
    return ' '.join(stemmed)

# Example
text = "The runners were running and ran through the running track"

print("Original:", text)
print("Porter:", stem_text(text, 'porter'))
print("Lancaster:", stem_text(text, 'lancaster'))
print("Snowball:", stem_text(text, 'snowball'))

# Output:
# Original: The runners were running and ran through the running track
# Porter: the runner were run and ran through the run track
# Lancaster: the run wer run and ran through the run track
# Snowball: the runner were run and ran through the run track
```

### Problems with Stemming

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Problem 1: Over-stemming
over_stem_words = ['university', 'universe', 'universal']
print("Over-stemming problem:")
for word in over_stem_words:
    print(f"{word:15s} → {stemmer.stem(word)}")
# All become 'univers' (not a real word!)

# Problem 2: Under-stemming
under_stem_words = ['data', 'datum']
print("\nUnder-stemming problem:")
for word in under_stem_words:
    print(f"{word:15s} → {stemmer.stem(word)}")
# Remain different (should be same)

# Problem 3: Incorrect stems
incorrect_words = ['news', 'history', 'meeting']
print("\nIncorrect stems:")
for word in incorrect_words:
    print(f"{word:15s} → {stemmer.stem(word)}")
# news → new (wrong!)
# history → histori (not a word)
# meeting → meet (could be correct or not depending on context)

# Problem 4: Losing meaning
meaning_words = ['excellence', 'excellent', 'excel']
print("\nLosing meaning:")
for word in meaning_words:
    print(f"{word:15s} → {stemmer.stem(word)}")
# All become 'excel' but have different meanings!
```

### When to Use Stemming

**Use Stemming When:**
- Speed is critical
- Working with large datasets
- Approximate matching is acceptable
- Resource-constrained environments
- Information retrieval systems
- Search engines

**Avoid Stemming When:**
- Exact meaning matters
- Working with medical/legal text
- Translation tasks
- Sentiment analysis
- Named entity recognition

---

## 8. Lemmatization

### What is Lemmatization?

**Lemmatization** reduces words to their dictionary form (lemma) using vocabulary and morphological analysis.

**Example:**
```
running → run
runs → run
ran → run (better than stemming!)
better → good (considers meaning!)
```

### Stemming vs Lemmatization

| Aspect | Stemming | Lemmatization |
|--------|----------|---------------|
| **Output** | Root stem (may not be a word) | Dictionary lemma (always a word) |
| **Method** | Rule-based suffix removal | Vocabulary & morphological analysis |
| **Speed** | Fast | Slower |
| **Accuracy** | Lower | Higher |
| **Context** | No context | Uses POS tags |
| **Example** | caring → car | caring → care |

### Visual Comparison

```python
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = [
    'running', 'runs', 'ran',
    'better', 'good', 'best',
    'caring', 'cares', 'cared',
    'meeting', 'meetings',
    'leaves', 'leaving'
]

print(f"{'Word':<15} {'Stemming':<15} {'Lemmatization':<15}")
print("=" * 45)

for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word, pos='v')  # verb
    print(f"{word:<15} {stem:<15} {lemma:<15}")

# Output shows lemmatization produces real words
```

---

### WordNet Lemmatizer

**WordNet** is a large lexical database of English.

```python
from nltk.stem import WordNetLemmatizer
import nltk

# Download required data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Create lemmatizer
lemmatizer = WordNetLemmatizer()

words = ['running', 'runs', 'ran', 'better', 'cacti', 'geese']

print("Basic lemmatization:")
for word in words:
    lemma = lemmatizer.lemmatize(word)
    print(f"{word:15s} → {lemma}")

# Output:
# running         → running (wrong without POS!)
# runs            → run
# ran             → ran
# better          → better (wrong without POS!)
# cacti           → cactus
# geese           → goose
```

### Importance of POS Tags

**POS (Part of Speech)** tags are crucial for accurate lemmatization!

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Same word, different POS tags
word = "running"

print("Lemmatization with different POS tags:")
print(f"No POS:    {lemmatizer.lemmatize(word)}")
print(f"Noun (n):  {lemmatizer.lemmatize(word, pos='n')}")
print(f"Verb (v):  {lemmatizer.lemmatize(word, pos='v')}")
print(f"Adj (a):   {lemmatizer.lemmatize(word, pos='a')}")
print(f"Adv (r):   {lemmatizer.lemmatize(word, pos='r')}")

# Output:
# No POS:    running  (wrong!)
# Noun (n):  running
# Verb (v):  run      (correct!)
# Adj (a):   running
# Adv (r):   running

# Another example
word = "better"
print(f"\nWord: {word}")
print(f"No POS:    {lemmatizer.lemmatize(word)}")
print(f"Adj (a):   {lemmatizer.lemmatize(word, pos='a')}")  # good!
```

**POS Tags:**
- `n` = noun
- `v` = verb
- `a` = adjective
- `r` = adverb
- `s` = satellite adjective

### Automatic POS Tagging

```python
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tag to wordnet POS tag"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def lemmatize_text(text):
    """Lemmatize text with automatic POS tagging"""
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # POS tag
    pos_tags = pos_tag(tokens)
    
    # Lemmatize with POS
    lemmatized = []
    for word, tag in pos_tags:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word.lower(), pos=wn_tag)
        lemmatized.append(lemma)
    
    return lemmatized

# Example
text = "The leaves are leaving the trees. Better times are coming."

print("Original:", text)
print("Tokens:", word_tokenize(text))
print("POS tags:", pos_tag(word_tokenize(text)))
print("Lemmatized:", lemmatize_text(text))

# Output shows correct lemmatization:
# leaves (noun) → leaf
# leaving (verb) → leave
# Better (adj) → good
# coming (verb) → come
```

### Complete Lemmatization Class

```python
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import string

class Lemmatizer:
    def __init__(self, lowercase=True, remove_punctuation=True):
        """
        Initialize lemmatizer
        
        Args:
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation
        """
        self.lemmatizer = WordNetLemmatizer()
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
    
    def get_wordnet_pos(self, treebank_tag):
        """Convert treebank POS to wordnet POS"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def lemmatize(self, text, return_tokens=False):
        """
        Lemmatize text
        
        Args:
            text: Input text
            return_tokens: Return list of tokens or joined string
        
        Returns:
            Lemmatized text or tokens
        """
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # POS tag
        pos_tags = pos_tag(tokens)
        
        # Remove punctuation if needed
        if self.remove_punctuation:
            pos_tags = [(word, tag) for word, tag in pos_tags if word not in string.punctuation]
        
        # Lemmatize with POS
        lemmatized = []
        for word, tag in pos_tags:
            wn_tag = self.get_wordnet_pos(tag)
            lemma = self.lemmatizer.lemmatize(word, pos=wn_tag)
            lemmatized.append(lemma)
        
        # Return tokens or joined string
        if return_tokens:
            return lemmatized
        return ' '.join(lemmatized)

# Example usage
lemmatizer = Lemmatizer(lowercase=True, remove_punctuation=True)

text = "The runners were running quickly. They had run marathons before."

print("Original:", text)
print("Lemmatized:", lemmatizer.lemmatize(text))
print("Tokens:", lemmatizer.lemmatize(text, return_tokens=True))

# Output:
# Original: The runners were running quickly. They had run marathons before.
# Lemmatized: the runner be run quickly they have run marathon before
# Tokens: ['the', 'runner', 'be', 'run', 'quickly', 'they', 'have', 'run', 'marathon', 'before']
```

---

## 9. POS Tagging

### What is POS Tagging?

**Part-of-Speech (POS) Tagging** assigns grammatical labels to each word.

**Example:**
```
The/DET cat/NOUN sat/VERB on/ADP the/DET mat/NOUN
```

### Common POS Tags

| Tag | Meaning | Examples |
|-----|---------|----------|
| **NN** | Noun | cat, dog, book |
| **VB** | Verb | run, sit, jump |
| **JJ** | Adjective | big, red, happy |
| **RB** | Adverb | quickly, slowly, well |
| **DET** | Determiner | the, a, an |
| **PRP** | Pronoun | I, you, he, she |
| **IN** | Preposition | in, on, at |
| **CC** | Conjunction | and, but, or |

### NLTK POS Tagger

```python
from nltk import pos_tag
from nltk.tokenize import word_tokenize

text = "The quick brown fox jumps over the lazy dog"

tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)

print("POS Tags:")
for word, tag in pos_tags:
    print(f"{word:10s} → {tag}")
```

---

## 10. Named Entity Recognition

### What is NER?

**Named Entity Recognition** identifies and classifies named entities in text.

**Entity Types:**
- **PERSON:** Names of people
- **LOCATION:** Geographic locations
- **ORGANIZATION:** Companies, institutions
- **DATE:** Dates and times
- **MONEY:** Monetary values
- **PERCENT:** Percentages

### NLTK NER

```python
from nltk import ne_chunk, pos_tag
from nltk.tokenize import word_tokenize

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
named_entities = ne_chunk(pos_tags)

print(named_entities)
```

---

## 11. Text Preprocessing Pipeline

### Complete NLP Pipeline

```python
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
import string

class NLPPipeline:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Tokenize sentences
        sentences = sent_tokenize(text)
        
        # Process each sentence
        processed_sentences = []
        for sentence in sentences:
            # Tokenize words
            tokens = word_tokenize(sentence)
            
            # Remove punctuation and stopwords
            tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
            
            # Lemmatize
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
            processed_sentences.append(' '.join(tokens))
        
        return processed_sentences

pipeline = NLPPipeline()
text = "Machine learning is amazing! Visit https://example.com for more info."
result = pipeline.preprocess(text)
print(result)
```

---

## 12. Sparse vs Dense Representations

### Sparse Representations

Sparse representations use discrete features (Bag of Words, TF-IDF).

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = [
    "Python is great",
    "I love Python",
    "Python machine learning"
]

# Bag of Words
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(documents)
print("Bag of Words:")
print(bow.toarray())

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)
print("\nTF-IDF:")
print(tfidf_matrix.toarray())
```

### Dense Representations

Dense representations use continuous embeddings (Word2Vec, GloVe).

```python
from gensim.models import Word2Vec

sentences = [
    ['python', 'is', 'great'],
    ['i', 'love', 'python'],
    ['machine', 'learning', 'rocks']
]

model = Word2Vec(sentences, min_count=1)
vector = model.wv['python']
print("Word2Vec vector for 'python':", vector)
```

---

## 13. Interview Questions

### Q1: What is the difference between stemming and lemmatization?

**Answer:**
- **Stemming:** Rule-based, faster, may produce non-words
- **Lemmatization:** Dictionary-based, slower, produces real words
- Stemming: running → runn, Lemmatization: running → run

### Q2: When should you remove stopwords?

**Answer:**
Remove for: text classification, topic modeling, information retrieval
Keep for: sentiment analysis, machine translation, question answering

### Q3: What is tokenization?

**Answer:**
Process of breaking text into smaller units (words, sentences, characters).

### Q4: Explain POS tagging importance.

**Answer:**
- Helps in lemmatization (know word's part of speech)
- Named entity recognition
- Dependency parsing
- Grammar checking

### Q5: What are the challenges in NLP?

**Answer:**
- Ambiguity (word sense, structural)
- Idioms and phrases
- Context understanding
- Language variation
- Multilingual processing

---

## 14. Practical Projects

### Project 1: Sentiment Analysis

```python
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def analyze(self, text):
        scores = self.sia.polarity_scores(text)
        return scores

analyzer = SentimentAnalyzer()
text = "This movie is absolutely wonderful!"
print(analyzer.analyze(text))
```

### Project 2: Text Classification

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample data
texts = [
    "Python programming is fun",
    "Java is a great language",
    "Machine learning with Python",
    "Deep learning neural networks"
]
labels = [0, 0, 1, 1]  # 0=programming, 1=ML

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

pipeline.fit(texts, labels)
prediction = pipeline.predict(["Python machine learning"])
print(f"Predicted label: {prediction}")
```

### Project 3: Text Summarization

```python
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter

class TextSummarizer:
    def __init__(self, ratio=0.3):
        self.ratio = ratio
        self.stop_words = set(stopwords.words('english'))
    
    def summarize(self, text):
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Calculate word frequencies
        word_freq = Counter([w for w in words if w.isalnum() and w not in self.stop_words])
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in word_tokenize(sentence.lower()):
                if word in word_freq:
                    sentence_scores[i] = sentence_scores.get(i, 0) + word_freq[word]
        
        # Get top sentences
        top_n = max(1, int(len(sentences) * self.ratio))
        top_sentences = sorted(sentence_scores.keys(), key=lambda x: sentence_scores[x], reverse=True)[:top_n]
        
        # Return summary in original order
        return ' '.join([sentences[i] for i in sorted(top_sentences)])

summarizer = TextSummarizer(ratio=0.4)
text = "Python is great. Machine learning is powerful. Deep learning uses neural networks. These are important technologies."
print(summarizer.summarize(text))
```

---

## Summary

This guide covers:
✅ NLTK fundamentals and setup  
✅ Tokenization methods and comparisons  
✅ Regular expressions for NLP  
✅ Stopword handling  
✅ Text normalization techniques  
✅ Stemming and lemmatization  
✅ POS tagging and NER  
✅ Complete preprocessing pipeline  
✅ Sparse vs dense representations  
✅ Common interview questions  
✅ Practical projects  

**Best Practices:**
- Always preprocess text before NLP tasks
- Use lemmatization for better accuracy
- Keep POS context for accurate processing
- Test different pipelines for your use case
- Combine multiple techniques for best results
        