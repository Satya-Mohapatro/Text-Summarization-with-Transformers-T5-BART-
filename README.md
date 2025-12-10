#  Text Summarization with Transformers (T5/BART)
Fine-tuning state-of-the-art Transformer models (T5 / BART) on the **CNN/DailyMail Abstractive Summarization Dataset** to generate high-quality summaries of long news articles. Implemented fully in **Google Colab** using Hugging Face **Transformers**, **Datasets**, and **evaluate** libraries.

---

##  Project Overview
This project focuses on **abstractive text summarization**, where the model learns to generate new sentences rather than extract existing ones. Using powerful Seq2Seq Transformer architectures (T5 or BART), the model is trained to condense long news articles into short, meaningful summaries.

### ✔ Why Summarization?
- Helps process large volumes of text quickly  
- Useful in journalism, legal tech, medical documentation  
- Improves search engines, chatbots, and recommendation systems  

### ✔ Input / Output Format
| Input | Output |
|-------|--------|
| Long news article text | Short abstractive summary |

---

##  Dataset — CNN/DailyMail
We use the **CNN/DailyMail v3.0.0** dataset via Hugging Face Datasets.

### Dataset Features
- `article`: Full news article  
- `highlights`: Human-written summary  
- Train/Validation/Test splits provided  

Preprocessing includes:
- Whitespace normalization  
- Tokenization using model tokenizer  
- Truncation to 512 tokens  
- Label tokenization for summaries  

---

##  Preprocessing Pipeline
The notebook performs:
- Text cleaning  
- Tokenization  
- Padding + masking  
- Creation of `input_ids`, `attention_mask`, `labels`  
- Dynamic padding with `DataCollatorForSeq2Seq`  

---

##  Model Architecture
Two models can be used:

### **1️⃣ BART — `facebook/bart-large-cnn`**
Fine-tuned specifically for summarization tasks.

### **2️⃣ T5 — `t5-base` / `t5-small`**
Text-to-text architecture where tasks are written as prompts.

Both use:
- Transformer **Encoder–Decoder**
- Auto-regressive decoding  
- Pretrained weights for knowledge transfer  

---

##  Training Setup
Training is done using **Seq2SeqTrainer** from Hugging Face.

Includes:
- ROUGE metric evaluation  
- Beam search generation  
- Checkpoint saving  
- Mixed precision (FP16) if GPU available  

### Losses & Metrics:
- **CrossEntropy Loss**
- **ROUGE-1, ROUGE-2, ROUGE-L**

---

##  Evaluation & Results
The notebook includes:
- ROUGE score evaluation  
- Loss curves  
- Sample predictions vs. ground-truth summaries  
- Observations on strengths & weaknesses  
- Discussion about dataset biases & hallucinations  

---

