# 📄 Named Entity Recognition for Legal Documents

## 🚀 Overview
This project focuses on building a **domain-specific Named Entity Recognition (NER) system** for legal documents using transformer-based models. Unlike generic NER systems, this model identifies **legal-specific entities** such as Petitioner, Respondent, Judge, Statute, and Precedent.

The objective is to enable **automated legal text analysis**, improving tasks like document search, summarization, and information extraction.

---

## 📊 Dataset
- **Dataset:** InLegalNER (`opennyaiorg/InLegalNER`)
- **Source:** Indian legal court judgments  
- **Size:** ~11,000 annotated samples  
- **Labels:** 14 legal entity types (e.g., COURT, PETITIONER, JUDGE, STATUTE, PRECEDENT)  
- **Format:** Token-level annotations with NER tags  

---

## 🧠 Model
- **Architecture:** RoBERTa (Transformer-based model)  
- **Framework:** Hugging Face Transformers  
- **Approach:** Fine-tuning on domain-specific dataset  
- **Advantage:** Improved performance on legal text compared to generic NER models  

---

## ⚙️ Methodology
1. Loaded and explored the dataset  
2. Tokenized text using RoBERTa tokenizer  
3. Aligned labels with tokenized inputs  
4. Fine-tuned the model using Hugging Face Trainer API  
5. Evaluated using Precision, Recall, and F1-score  

---

## 📈 Results
- Achieved strong performance across multiple entity classes  
- Successfully extracted entities from unseen legal documents  
- Demonstrated effectiveness of domain-specific training  

---

## 🔍 Example Output

**Input:**
