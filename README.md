# EmoSense
EmoSense is a transformer-based NLP model to classify emotions in English text.

## GoEmotions → 8-Class Emotion Classifier (DistilBERT)

This project fine-tunes **DistilBERT** on the **GoEmotions (simplified)** dataset to build an **8-class emotion classifier**. It maps the original GoEmotions labels into 8 broader categories and trains a weighted cross-entropy model to handle class imbalance.

## What this code does

1. **Installs dependencies** (`torch`, `transformers`, `datasets`, `evaluate`, etc.)
2. **Loads GoEmotions (simplified)** from Hugging Face
3. **Maps original labels → 8 emotions** (anger, disgust, fear, joy, sadness, surprise, neutral, love)
4. **Checks the class distribution** after mapping
5. **Computes class weights** with `sklearn.utils.class_weight` to reduce imbalance effects
6. **Tokenizes text** using `distilbert-base-uncased`
7. **Fine-tunes DistilBERT** for sequence classification using Hugging Face `Trainer`
8. **Evaluates** with Accuracy + Macro F1
9. **Saves** the best model and tokenizer to disk

---

## Label mapping (8 classes)

The original GoEmotions dataset contains many fine-grained labels. This project collapses them into 8 coarse categories:

| ID | Label     |
|----|-----------|
| 0  | anger     |
| 1  | disgust   |
| 2  | fear      |
| 3  | joy       |
| 4  | sadness   |
| 5  | surprise  |
| 6  | neutral   |
| 7  | love      |

> Mapping is defined in `simplified_map` inside the code (e.g., *annoyance → anger*, *curiosity → surprise*, *gratitude → joy*, etc.).

---

## Metrics

During evaluation, the code reports:

- **Accuracy**
- **Macro F1** (recommended for imbalanced multi-class classification)

Macro F1 computes F1 for each class and averages them equally, so small classes matter.

---

## How to run (Google Colab)

### 1) Open in Colab
Upload the notebook or open it from Google Drive.

### 2) (Recommended) Mount Google Drive
The code saves checkpoints/logs to Drive paths like:
- `/content/drive/MyDrive/emotion_model`
- `/content/drive/MyDrive/emotion_model/logs`

So mount Drive first:

```python
from google.colab import drive
drive.mount('/content/drive')
