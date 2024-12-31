# **IMDB Sentiment Analysis**
Fine-tuning a BERT-based language model to classify movie reviews as positive or negative using the IMDB dataset.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Installation](#installation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Demo](#demo)
9. [Contributing](#contributing)
10. [License](#license)

---

## **Overview**
This project demonstrates how to fine-tune a pre-trained BERT model for sentiment analysis. Using the IMDB dataset, we classify movie reviews as either **positive** or **negative**.

Key features:
- Preprocessing and tokenization using Hugging Face's `transformers`.
- Fine-tuning with Hugging Face's `Trainer` API.
- Real-world predictions on unseen text data.
- Interactive demo via Gradio (run locally).

---

## **Dataset**
The dataset used is the [IMDB dataset](https://huggingface.co/datasets/imdb), which contains:
- **50,000 movie reviews**: 25,000 for training and 25,000 for testing.
- Labels: `positive` and `negative`.

---

## **Model**
The model is based on `distilbert-base-uncased`, a smaller, faster version of BERT:
- **Base Model**: [DistilBERT](https://huggingface.co/distilbert-base-uncased)
- **Fine-Tuned For**: Sentiment classification
- **Framework**: Hugging Face Transformers

---

## **Installation**
To run the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/katherine-welbourne/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
   ```

2. Install dependencies:
   ```bash
   pip install transformers datasets torch gradio
   ```

---

## **Training**
1. Preprocess the dataset:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("imdb")
   ```

2. Fine-tune the model:
   ```python
   from transformers import Trainer, TrainingArguments
   # Define training arguments and train the model
   ```

3. Save the fine-tuned model:
   ```python
   model.save_pretrained("./sentiment_model")
   tokenizer.save_pretrained("./sentiment_model")
   ```

---

## **Evaluation**
Evaluate the model's performance on the test set:
```python
trainer.evaluate()
```

Metrics:
- **Accuracy**: 94%
- **F1-Score**: 93%

---

## **Results**
Example predictions:
| Input Text                                | Prediction       | Confidence |
|-------------------------------------------|------------------|------------|
| "This movie was amazing! I loved it."     | :blush: Positive      | 97.34%     |
| "The plot was boring and uninspired."     | :rage: Negative      | 92.88%     |

---

## **Demo**

You can test the sentiment analysis model locally using two approaches:

### **Option 1: Gradio App**
1. Make sure you have `gradio` installed:
   ```bash
      pip install gradio
   ```
2. Run the Gradio app:
   ```bash
   python3 run_gradio_movie_sentiment.py
   ```
### **Option 2: Terminal-Based App**
1. Run the terminal-based script:
   ```bash
      python3 run_terminal_movie_sentiment.py
   ```
2. Enter movie reviews directly into the terminal when prompted. Example:
   ```bash
   Enter a movie review (or type 'exit' to quit): This movie was fantastic!

   Input: This movie was fantastic!
   Prediction: 😊 Positive (Confidence: 97.34%)

   ```
   
## **Demo**
You can run the app locally to test the model:
1. Run the following command:
   ```bash
   python3 run_movie_sentiment.py

   ```
2. Open the Gradio app in your browser at the provided local URL.

---

## **Contributing**
Feel free to contribute by submitting issues or pull requests. For major changes, please open an issue first to discuss what you’d like to change. :tada:

---

## **License**
This project is licensed under the [MIT License](LICENSE).
