from transformers import pipeline
import gradio as gr

# Define label mapping with emojis
label_map = {
    "LABEL_0": "😡 Negative",
    "LABEL_1": "😊 Positive"
}

# Load the model and tokenizer from the current directory
model_path = "./movie_sentiment_model"
sentiment_analyzer = pipeline("text-classification", model=model_path, tokenizer=model_path)

# Define prediction function
def predict_sentiment(review):
    result = sentiment_analyzer(review)
    label = label_map[result[0]['label']]  # Map label to emoji
    confidence = result[0]['score'] * 100  # Convert confidence to percentage
    return f"{label} (Confidence: {confidence:.2f}%)"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter a movie review..."),
    outputs="text",
    title="🎬 Movie Sentiment Analysis",
    description="Analyze the sentiment of movie reviews with emojis and confidence scores!"
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
