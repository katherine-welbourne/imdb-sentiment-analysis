from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained tokenizer first
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load a pre-trained model and tokenizer (e.g., DistilBERT)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

from transformers import TrainingArguments
from transformers import Trainer


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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
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
