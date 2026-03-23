# Named-Entity-Recognition-for-Legal-Documents
Overview
This project focuses on building a domain-specific Named Entity Recognition (NER) system for legal documents using transformer-based models. Unlike generic NER systems, this model is designed to identify legal-specific entities such as Petitioner, Respondent, Judge, Statute, and Precedent.

The goal is to enable automated legal text analysis, making tasks like document search, summarization, and information extraction more efficient.

Dataset
Dataset: InLegalNER (opennyaiorg/InLegalNER)
Source: Indian legal court judgments
Size: ~11,000 annotated samples
Labels: 14 legal entity types (e.g., COURT, PETITIONER, JUDGE, STATUTE, PRECEDENT)
Format: Token-level annotations with corresponding NER tags
Model
Architecture: RoBERTa (Transformer-based model)
Framework: Hugging Face Transformers
Approach: Fine-tuning on domain-specific dataset
Key Advantage: Better performance on legal text compared to generic NER models
Methodology
Loaded and explored the InLegalNER dataset
Tokenized text using RoBERTa tokenizer
Aligned labels with tokenized inputs
Fine-tuned the model using Hugging Face Trainer API
Evaluated performance using standard metrics (Precision, Recall, F1-score)
Results
Achieved strong performance across multiple entity classes
Successfully identified legal entities in unseen court documents
Demonstrated effectiveness of domain-specific fine-tuning
