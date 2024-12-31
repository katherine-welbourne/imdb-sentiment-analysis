from transformers import pipeline
import gradio as gr

# Load the model and tokenizer from the current directory
model_path = "./movie_sentiment_model"
# Load the fine-tuned model and tokenizer from the local directory
sentiment_analyzer = pipeline("text-classification", model=model_path, tokenizer=model_path)

# Define label mapping with emojis
label_mapping = {
    "LABEL_0": "😡 Negative",
    "LABEL_1": "😊 Positive"
}

# Define prediction function
def predict_sentiment(text):
    results = sentiment_analyzer(text)
    label = label_mapping[results[0]['label']]  # Convert LABEL_0 or LABEL_1 to human-readable text with emojis
    confidence = results[0]['score'] * 100  # Convert confidence to percentage
    return f"{label}\nConfidence: {confidence:.2f}%"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter a movie review here..."),
    outputs="text",
    title="🎬 Sentiment Analysis",
    description="Analyze the positive-negative sentiment of any movie review with emojis! 😊😡",
    theme="default"
)

# Launch the app
if __name__ == "__main__":
    interface.launch(share=True)
