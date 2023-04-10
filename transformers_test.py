import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed

# Set seed
set_seed(42)

# Load the dataset
data = pd.read_csv('./res/train.csv')

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define the model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize the data
def tokenize_data(data):
    return tokenizer(data["text"], padding="max_length", truncation=True, max_length=256)

train_dataset = train_data.apply(tokenize_data, axis=1)
test_dataset = test_data.apply(tokenize_data, axis=1)

# Define the label-to-id mapping
label_map = {"EAP": 0, "HPL": 1, "MWS": 2}

# Prepare the dataset for the Trainer
class SpookyAuthorDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {key: torch.tensor(self.data.iloc[idx][key]) for key in self.data.iloc[idx].keys()}
        item["labels"] = torch.tensor(label_map[self.labels.iloc[idx]])
        return item

train_dataset = SpookyAuthorDataset(train_dataset, train_data["author"])
test_dataset = SpookyAuthorDataset(test_dataset, test_data["author"])

# Define the compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# Set up the Trainer and training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics, 
)
# Fine-tune the model
trainer.train()

# Evaluate the model
evaluation_results = trainer.evaluate()

# Save the fine-tuned model
model.save_pretrained("./spooky_author_model")
tokenizer.save_pretrained("./spooky_author_model")

# Make predictions on new text
def predict_author(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    author_idx = np.argmax(outputs.logits.detach().cpu().numpy())
    authors = ["EAP", "HPL", "MWS"]
    return authors[author_idx]

new_text = "The world is indeed comic, but the joke is on mankind."
predicted_author = predict_author(new_text)
print("Predicted author:", predicted_author)

