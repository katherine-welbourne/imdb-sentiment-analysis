from transformers import pipeline

# Define label mapping with emojis
label_map = {
    "LABEL_0": "😡 Negative",
    "LABEL_1": "😊 Positive"
}

# Load the model and tokenizer from the current directory
model_path = "./movie_sentiment_model"
sentiment_analyzer = pipeline("text-classification", model=model_path, tokenizer=model_path)

# Run a loop to allow multiple inputs
while True:
    # Prompt the user for input
    sample_text = input("\nEnter a movie review (or type 'exit' to quit): ")

    # Exit condition
    if sample_text.lower() == "exit":
        print("Exiting sentiment analysis. Goodbye!")
        break

    # Get sentiment prediction
    result = sentiment_analyzer(sample_text)
    label = label_map[result[0]['label']]  # Map label to emoji
    confidence = result[0]['score'] * 100  # Convert confidence to percentage

    # Display the result with emoji and confidence
    print(f"\nInput: {sample_text}")
    print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
