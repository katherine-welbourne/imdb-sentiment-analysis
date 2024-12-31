# **IMDB Sentiment Analysis** 🎬🤖
Fine-tuning a BERT-based language model to classify movie reviews as positive or negative using the IMDB dataset.

---

## **Table of Contents** 📚
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Installation](#installation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Demo](#demo)
9. [Pre-Trained Model](#pre-trained-model)
10. [Contributing](#contributing)
11. [License](#license)

---

## **Overview** 📝
This project demonstrates how to fine-tune a pre-trained BERT model for sentiment analysis. Using the IMDB dataset, we classify movie reviews as either **positive** or **negative**.

Key features:
- Preprocessing and tokenization using Hugging Face's `transformers`. ✂️🛠️
- Fine-tuning with Hugging Face's `Trainer` API. 🏋️
- Real-world predictions on unseen text data. 🌍
- Interactive demo via Gradio (run locally). 🎨

---

## **Dataset** 📊
The dataset used is the [IMDB dataset](https://huggingface.co/datasets/imdb), which contains:
- **50,000 movie reviews**: 25,000 for training and 25,000 for testing. 📥📤
- Labels: `positive` 😊 and `negative` 😡.

---

## **Model** 🤖
The model is based on `distilbert-base-uncased`, a smaller, faster version of BERT:
- **Base Model**: [DistilBERT](https://huggingface.co/distilbert-base-uncased) 🚀
- **Fine-Tuned For**: Sentiment classification
- **Framework**: Hugging Face Transformers 🛠️

---

## **Installation** ⚙️
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

## **Training** 🏋️
   ```bash
   python3 train_movie_sentiment.py
   ```

---

## **Evaluation** 📏
Evaluate the model's performance on the test set:
```python
trainer.evaluate()
```

Metrics:
- **Accuracy**: 94% ✅
- **F1-Score**: 93% 🎯

---

## **Results** 🏆
Example predictions:
| Input Text                                | Prediction       | Confidence |
|-------------------------------------------|------------------|------------|
| "This movie was amazing! I loved it."     | :blush: Positive | 97.34%     |
| "The plot was boring and uninspired."     | :rage: Negative  | 92.88%     |

---

## **Demo** 🖥️
You can test the sentiment analysis model locally using two approaches:

### **Option 1: Gradio App** 🎨
1. Make sure you have `gradio` installed:
   ```bash
   pip install gradio
   ```

2. Run the Gradio app:
   ```bash
   python3 run_gradio_movie_sentiment.py
   ```

3. Open the Gradio app in your browser using the local URL provided in the terminal.

### **Option 2: Terminal-Based App** 💻
1. Run the terminal-based script:
   ```bash
   python3 run_terminal_movie_sentiment.py
   ```

2. Enter movie reviews directly into the terminal when prompted. Example:
   ```plaintext
   Enter a movie review (or type 'exit' to quit): This movie was fantastic!

   Input: This movie was fantastic!
   Prediction: 😊 Positive (Confidence: 97.34%)
   ```

---

## **Pre-Trained Model** 🚀
If you do not have access to high RAM or a GPU like the A100, you can download the pre-trained model directly:

- [Download Pre-Trained Model from Google Drive](https://drive.google.com/file/d/1Mb2Jjw1p5e02a5luQCl3e29v3LC9f7dA/view?usp=sharing) 📥

After downloading:
1. Extract the model into the project folder.
2. Use it with the provided scripts for inference.

---

## **Contributing** 🤝
Feel free to contribute by submitting issues or pull requests. 💡✨

---

## **License** 📜
This project is licensed under the [MIT License](LICENSE). 📖

---
