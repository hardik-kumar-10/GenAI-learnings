import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
###make a small sample dataset###
data = {
    'text': [
        "This movie was absolutely amazing! Great acting and plot.",
        "Terrible film, waste of time. Poor story and acting.",
        "I loved every minute of it. Brilliant cinematography.",
        "Not worth watching. Very boring and predictable.",
        "Outstanding performance by the lead actor. Highly recommended!",
        "Worst movie I've ever seen. Complete disaster.",
        "Beautiful story with excellent character development.",
        "Disappointing. Expected much better from this director.",
        "Fantastic movie! Will definitely watch again.",
        "Poorly written script and bad direction.",
        "One of the best films of the year. Must watch!",
        "Boring and uninteresting. Couldn't finish it.",
        "Great special effects and engaging storyline.",
        "Overrated. Don't believe the hype.",
        "Perfect blend of action and emotion. Loved it!",
        "Confusing plot and weak characters.",
        "Masterpiece! Every scene was perfectly crafted.",
        "Cheap production values and amateur acting.",
        "Incredible movie with deep meaningful themes.",
        "Painfully slow and tedious to watch."
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
}

df = pd.DataFrame(data)
print(f"Dataset size: {len(df)} samples")
print(df.head())

###prepare the dataset for training###
from transformers import AutoTokenizer

# Initialize tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.3, random_state=42
)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)

# Create datasets
train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
test_dataset = Dataset.from_dict({'text': test_texts, 'labels': test_labels})

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

###model setup###
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import warnings

# Suppress the expected warning about uninitialized weights
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized.*")

# Load pre-trained model
print("Loading pre-trained DistilBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2  # binary classification
)

print("✅ Model loaded successfully!")
print("Note: The warning about uninitialized classifier weights is EXPECTED.")
print("These layers will be trained during fine-tuning.\n")

# Define metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

###configure training parameters###
# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              # Small number for quick training
    per_device_train_batch_size=4,   # Small batch size for limited data
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

###train the model###
# Start training
print("Starting training...")
trainer.train()

# Evaluate the model
print("\nEvaluating model...")
results = trainer.evaluate()
print(f"Test Accuracy: {results['eval_accuracy']:.3f}")

###test the model###
from transformers import pipeline

# Create prediction pipeline
classifier = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer,
    return_all_scores=True
)

# Test with new examples
test_reviews = [
    "This movie exceeded all my expectations!",
    "Absolutely horrible film, don't waste your time.",
    "Pretty good movie with some great moments.",
]

print("\nTesting fine-tuned model:")
for review in test_reviews:
    result = classifier(review)
    print(f"Review: {review}")
    
    # Handle the result format properly
    if isinstance(result[0], list):
        # If return_all_scores=True, result is [[{label, score}, {label, score}]]
        scores = result[0]
        best_prediction = max(scores, key=lambda x: x['score'])
        print(f"Prediction: {best_prediction['label']} (Confidence: {best_prediction['score']:.3f})")
    else:
        # If return_all_scores=False, result is [{label, score}]
        print(f"Prediction: {result[0]['label']} (Confidence: {result[0]['score']:.3f})")
    print()

##saving the model###
# Save the fine-tuned model
model.save_pretrained('./fine-tuned-sentiment-model')
tokenizer.save_pretrained('./fine-tuned-sentiment-model')
print("Model saved to './fine-tuned-sentiment-model'")

# Load it later with:
# model = AutoModelForSequenceClassification.from_pretrained('./fine-tuned-sentiment-model')
# tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-sentiment-model')