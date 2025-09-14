import json
from datasets import Dataset, DatasetDict
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Load JSONL dataset
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

# Load dataset
full_dataset = load_jsonl("gtm.jsonl")

# Split into train/validation (e.g. 90% train, 10% eval)
dataset = full_dataset.train_test_split(test_size=0.1, seed=42)

max_input_length = 512
max_output_length = 64

def preprocess(batch):
    inputs = batch.get("input", batch.get("source"))
    targets = batch.get("output", batch.get("target"))

    if not isinstance(inputs, list):
        inputs = [str(inputs)]
    else:
        inputs = [str(i) for i in inputs]

    if not isinstance(targets, list):
        targets = [str(targets)]
    else:
        targets = [str(t) for t in targets]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True
    )
    labels = tokenizer(
        targets,
        max_length=max_output_length,
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=full_dataset.column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gtm-light",
    save_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50,
    fp16=False
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()


# Train the model
trainer.train()
