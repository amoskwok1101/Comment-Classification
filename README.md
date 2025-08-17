# Toxic Comment Classification

This repository contains code and experiments for **toxic comment classification**.  
The task is a **multi-label classification problem**, where each comment may belong to one or more categories of toxicity.  

We experiment with both:
- **BiLSTM** (Bidirectional LSTM)
- **BERT** (Transformer-based model)

---

## Dataset

Each comment is annotated with **6 binary labels**, indicating the presence (`1`) or absence (`0`) of a toxicity type.

**Label list:**
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

### Example format

| comment_text                  | toxic | severe_toxic | obscene | threat | insult | identity_hate |
|-------------------------------|-------|--------------|---------|--------|--------|----------------|
| "You are so stupid"           | 1     | 0            | 0       | 0      | 1      | 0              |
| "I will find you and harm"    | 0     | 1            | 0       | 1      | 0      | 0              |
