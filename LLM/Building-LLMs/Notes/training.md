# Training LLMs: A Teen-Friendly Canvas (Basics → Advanced)

## 1) The Big Picture

- Goal: turn lots of text into a model that can predict the next token and, surprisingly, do much more (Q&A, translation, multiple choice).
- Two main stages: **pretraining** (learn everything you can) → **fine-tuning** (specialize for a job).

## 2) Pretraining (Learn Everything)

- **What it is:** Train a huge model on a massive, mostly unlabeled text mix to learn general language patterns, facts, and reasoning by next-token prediction.
- **Typical data sources:** Common Crawl (web), WebText2 (Reddit-linked web pages), Books1/Books2 (published books), Wikipedia (curated facts).
- **Why unlabeled?** Raw text already contains structure; the model learns by predicting the next word/token without human-provided tags.
- **Outcome:** A generalist model that “gets” grammar, style, and broad world knowledge.

## 3) Fine-Tuning (Specialize)

- **What it is:** Start from a pretrained base and keep training on a smaller, task/domain-specific dataset.
- **Why:** Improve performance where you need it: sentiment analysis, support chat, medical/legal text, etc., especially when labeled data is limited.
- **Data types:** Usually **labeled** (input + desired output). Examples: sentiment labels, intent labels, or instruction → answer pairs.

## 4) Two Common Fine-Tune Flavors

- **Instruction fine-tuning:** Dataset of prompts/instructions with target answers. Makes the model follow human-style instructions better.
- **Task/classification fine-tuning:** Labeled examples mapping text → class (e.g., spam vs not-spam). Boosts accuracy on that task.

## 5) Labeled vs Unlabeled Data

- **Unlabeled (raw text):** Sentences with no extra tags. Used in pretraining; learning signal comes from predicting next tokens.
- **Labeled:** Text plus tags/answers (e.g., “email → spam”). Used in fine-tuning to steer the model toward specific behaviors.

## 6) Is ChatGPT Fine-Tuned?

- Yes. Models like GPT start with huge **pretraining**, then go through **instruction tuning** and alignment steps (often including human feedback) so they follow instructions and stay on-policy.

## 7) Why Pretraining Is Expensive

- Needs billions of tokens, large GPUs/TPUs, and long training runs → costs can reach millions of dollars. Most teams reuse open-weight pretrained bases instead of training from scratch.

## 8) Emergent Skills (Why Next-Token Prediction Is Enough)

- By predicting the next token across diverse text, the model indirectly learns to answer questions, translate, and reason. Scale (data + parameters) unlocks these abilities.

## 9) Quick Study Map

- Start: raw text → tokenize → pretrain on unlabeled data (Common Crawl, books, Wikipedia).
- Then: fine-tune with labeled data for your task (instruction pairs or class labels).
- Result: a model that is general-purpose but can be specialized for domains where data is limited.

## 10) Practice Prompts (Teen-Friendly)

- "Explain pretraining like teaching someone to guess the next lyric in a song."
- "Why do we need labels for fine-tuning but not for pretraining?"
- "How would you fine-tune an LLM to spot spam emails?"
- "Describe why pretraining costs so much and why most people start from open-weight models."

## 11) One-Liner Recap

Pretraining teaches broad language skills from raw text; fine-tuning adds the specialist badge with labeled data so the model excels at your specific task.
