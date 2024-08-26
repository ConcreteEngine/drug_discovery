import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer, pipeline
from datasets import load_dataset
import re

# Use a smaller model for faster testing
device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

# Load 1% of the dataset
dataset = load_dataset("allenai/c4", "realnewslike", split='train[:1%]')

# Further downsample to 0.1%
fraction = 0.1
dataset = dataset.shuffle(seed=42).select([i for i in range(int(len(dataset) * fraction))])

# Load the tokenizer and add a padding token
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as the padding token

# Preprocess and tokenize the dataset with explicit truncation
def preprocess_and_tokenize(examples):
    examples['text'] = [text.lower() for text in examples['text']]
    tokenized = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=38)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess_and_tokenize, batched=True)

# Split the dataset into training and validation subsets
dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Use fewer epochs and smaller batches
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,  # Limited epochs due to CPU constraints
    weight_decay=0.01,
    no_cuda=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Refined prompt for generating text
prompt = "Doxycycline works to treat infections by preventing the growth and spread of bacteria. It works by"

# Create a text generation pipeline with adjusted parameters to reduce repetition
text_generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    device=-1, 
    repetition_penalty=1.5,  # Increase repetition penalty
    no_repeat_ngram_size=3,  # Prevent repeating 3-grams
    top_p=0.9,  # Use nucleus sampling for more diverse output
    max_length=150,  # Adjust max length as needed
    num_beams=5,  # Beam search with 5 beams
    early_stopping=True
)

# Generate potential new drug uses with adjusted parameters
generated_text = text_generator(prompt)

# Post-processing to clean the generated text and remove incomplete words
def clean_generated_text(text):
    # Remove any nonsensical words or phrases, using regex or a predefined list
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    
    # Remove any incomplete words at the end of the sentence
    if text[-1] != "." and " " in text:
        last_space = text.rfind(" ")
        text = text[:last_space] + "."
    return text

cleaned_text = clean_generated_text(generated_text[0]['generated_text'])
print("Generated Hypothesis:", cleaned_text)


