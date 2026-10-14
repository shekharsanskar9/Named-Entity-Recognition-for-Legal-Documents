# ⚖️ Legal NER: Named Entity Recognition for Indian Court Judgments

[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Model-legal__ner__model-yellow)](https://huggingface.co/ssanskar9/legal_ner_model)
[![Live Demo](https://img.shields.io/badge/🤗%20Space-Live%20Demo-blue)](https://huggingface.co/spaces/ssanskar9/legal-ner-demo)
![Python](https://img.shields.io/badge/Python-3.x-3776AB)
![Transformers](https://img.shields.io/badge/HF-Transformers-orange)

A **RoBERTa model fine-tuned to extract legal entities** from Indian court judgments: petitioners, respondents, judges, statutes, provisions, precedents and more.

General-purpose NER models only see `PERSON`, `ORG` and `LOCATION`. They can't tell a **petitioner** from a **judge**, or a **statute** from an ordinary organisation name. This project closes that gap by training on expert-annotated Indian legal text, so the output can drive downstream tasks such as judgment search, case summarisation, citation graphs and clause analysis.

**👉 [Try the live demo](https://huggingface.co/spaces/ssanskar9/legal-ner-demo)**: paste a paragraph from a judgment and see the entities highlighted.

---

## ✨ Highlights

- **14 legal-specific entity types**, not generic PER/ORG/LOC
- Fine-tuned **RoBERTa** with proper sub-word label alignment
- Evaluated with **entity-level** precision, recall and F1 (`seqeval`), not token accuracy
- Published on the **Hugging Face Hub** with an interactive **Gradio demo**

---

## 🏷️ Entity Types

| Label | What it captures |
|---|---|
| `COURT` | Name of the court (e.g. Supreme Court of India, High Court of Delhi) |
| `PETITIONER` | Party who filed the case |
| `RESPONDENT` | Party the case is filed against |
| `JUDGE` | Judges named in the judgment |
| `LAWYER` | Counsel appearing for either side |
| `DATE` | Dates relevant to the case |
| `ORG` | Organisations other than courts |
| `GPE` | Geopolitical entities: countries, states, cities |
| `STATUTE` | Acts and laws (e.g. Indian Penal Code) |
| `PROVISION` | Sections, articles, clauses (e.g. Section 302) |
| `PRECEDENT` | Earlier cases cited as authority |
| `CASE_NUMBER` | Case / appeal / petition numbers |
| `WITNESS` | Witnesses named in the case |
| `OTHER_PERSON` | People not covered by the roles above |

Tags follow the **BIO scheme** (`B-STATUTE`, `I-STATUTE`, `O`).

---

## 📊 Dataset

| | |
|---|---|
| **Name** | InLegalNER ([`opennyaiorg/InLegalNER`](https://huggingface.co/datasets/opennyaiorg/InLegalNER)) |
| **Source** | Indian Supreme Court and High Court judgments, annotated by legal experts (OpenNyAI) |
| **Size** | ~11,000 annotated training samples |
| **Content** | Sentences from judgment bodies plus judgment preambles |
| **Format** | Token-level BIO tags, 14 entity types |

<!-- TODO: add your exact train / validation / test split sizes -->

---

## 🧠 Model & Training

| Setting | Value |
|---|---|
| Base model | `roberta-base` <!-- TODO: confirm checkpoint --> |
| Task head | Token classification |
| Framework | Hugging Face Transformers (`Trainer` API) |
| Max sequence length | `TODO` |
| Learning rate | `TODO` |
| Batch size | `TODO` |
| Epochs | `TODO` |
| Hardware | `TODO` (e.g. Colab T4 GPU) |

---

## ⚙️ Pipeline

```
Raw judgment text
      │
      ▼
RoBERTa tokenizer  ──►  sub-word tokens
      │
      ▼
Label alignment    ──►  first sub-word keeps the word's tag, the rest get -100 (ignored in loss)
      │
      ▼
Fine-tuning        ──►  Trainer API, cross-entropy over BIO tags
      │
      ▼
Evaluation         ──►  entity-level P / R / F1 with seqeval
```

1. **Load & explore** the dataset: label distribution, sentence lengths, class imbalance.
2. **Tokenize** with the RoBERTa tokenizer (`is_split_into_words=True`, `add_prefix_space=True`).
3. **Align labels**: RoBERTa splits words into sub-words, so each word's tag is assigned to its first sub-word and the remaining sub-words are masked with `-100`.
4. **Fine-tune** with the Hugging Face `Trainer`, evaluating on the validation set each epoch and keeping the best checkpoint.
5. **Evaluate** at the entity level with `seqeval`: an entity only counts as correct if both its span and its type match exactly.

---

## 📈 Results

**Overall (test set, entity-level)**

| Metric | Score |
|---|---|
| Precision | `TODO` |
| Recall | `TODO` |
| **F1** | `TODO` |

**Per-entity F1**

| Entity | F1 | | Entity | F1 |
|---|---|---|---|---|
| COURT | `TODO` | | STATUTE | `TODO` |
| PETITIONER | `TODO` | | PROVISION | `TODO` |
| RESPONDENT | `TODO` | | PRECEDENT | `TODO` |
| JUDGE | `TODO` | | CASE_NUMBER | `TODO` |
| LAWYER | `TODO` | | WITNESS | `TODO` |
| DATE | `TODO` | | OTHER_PERSON | `TODO` |
| ORG | `TODO` | | GPE | `TODO` |

**Observations**
<!-- TODO: replace with what you actually saw. Typical things worth noting: -->
- Entities with consistent surface forms (e.g. `COURT`, `DATE`, `PROVISION`) are the easiest.
- Role-based person entities (`PETITIONER` vs `RESPONDENT` vs `WITNESS`) are harder, because the same name pattern can take different roles depending on context.
- Long `PRECEDENT` spans are the most error-prone because of boundary mistakes.

---

## 🚀 Quick Start

```bash
pip install transformers torch
```

```python
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="ssanskar9/legal_ner_model",
    aggregation_strategy="simple",   # merge sub-words into whole entities
)

text = (
    "The appellant was convicted under Section 302 of the Indian Penal Code "
    "by the High Court of Delhi."
)

for ent in ner(text):
    print(f"{ent['entity_group']:<12} {ent['word']:<28} {ent['score']:.2f}")
```

Example output (illustrative):

```
PROVISION    Section 302                  0.98
STATUTE      Indian Penal Code            0.99
COURT        High Court of Delhi          0.97
```

---

## 🔭 Limitations & Future Work

**Limitations**
- Trained only on Indian judgments, so performance on other jurisdictions or on contracts is untested.
- Long judgments must be chunked to fit the model's maximum sequence length.
- Role labels depend on context that may fall outside a single chunk.

**Next steps**
- **Legal document analyzer**: upload a document and get a plain-language summary, key obligations, red-flag clauses and deadlines, combining this NER model with an LLM.
- Compare against legal-domain encoders such as InLegalBERT.
- Add a CRF layer on top of the encoder for more consistent span boundaries.

---

## 🙏 Acknowledgements

- **Dataset:** [OpenNyAI](https://opennyai.org/) for InLegalNER. See Kalamkar et al., *Named Entity Recognition in Indian Court Judgments* (NLLP Workshop, 2022).
- **Libraries:** Hugging Face Transformers, Datasets, `seqeval`, Gradio.

---

## 👤 Author

**Sanskar Shekhar**, B.Tech CSE, BIT Mesra
[GitHub](https://github.com/shekharsanskar9) · [Hugging Face](https://huggingface.co/ssanskar9)
