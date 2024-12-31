from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load a dataset (e.g., IMDB for sentiment analysis)
dataset = load_dataset("imdb")

# Preprocess dataset for fine-tuning
def preprocess_data(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

# Convert to Hugging Face Dataset
train_dataset = dataset['train'].map(preprocess_data, batched=True)
test_dataset = dataset['test'].map(preprocess_data, batched=True)

# Load a pre-trained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Import additional libraries
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

# Define a custom compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    push_to_hub=False,
)

# Create Trainer with compute_metrics added
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Include metrics
)


# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save Locally
model.save_pretrained("./movie_sentiment_model")
tokenizer.save_pretrained("./movie_sentiment_model")

# Save to drive
#from google.colab import drive
#drive.mount('/content/drive')
#model.save_pretrained("/content/drive/My Drive/movie_sentiment_model")
#tokenizer.save_pretrained("/content/drive/My Drive/movie_sentiment_model")
