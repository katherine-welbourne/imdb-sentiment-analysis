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
if __name__ == "__main__":
    print("Welcome to the Terminal Movie Sentiment Analysis App!")
    print("Type your movie review below to get the sentiment analysis.")
    print("Type 'exit' to quit the app.\n")
    
    while True:
        # Prompt the user for input
        review = input("Enter a movie review: ")
        
        # Exit condition
        if review.lower() == "exit":
            print("Exiting the app. Goodbye!")
            break
        
        # Get sentiment prediction
        result = sentiment_analyzer(review)
        label = label_map[result[0]['label']]  # Map label to emoji
        confidence = result[0]['score'] * 100  # Convert confidence to percentage
        
        # Display the result
        print(f"\nInput: {review}")
        print(f"Prediction: {label} (Confidence: {confidence:.2f}%)\n")
