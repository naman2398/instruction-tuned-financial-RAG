"""
Financial RAG Pipeline - Model Finetuning Module

* Team Members:
  - Dhairya Umrania
  - Naman Deep
  - Devaansh Kataria

* Description:
  Contains code for finetuning a language model on financial data.
  Loads the FinLang/investopedia-instruction-tuning-dataset,
  preprocesses the dataset for finetuning a seq2seq model,
  implements a training loop with early stopping, and
  tracks and plots training and validation losses.

* NLP Class Concepts Applied:
  I. Syntax | Classification: 
     - Tokenization and processing of financial text
  II. Semantics | Probabilistic Models: 
     - Cross-entropy loss function for model training
  III. Language Modeling | Transformers: 
     - Finetuning transformer models (T5)
     - Transfer learning techniques
     - Model adaptation with learning rate scheduling
     - Instruction tuning approaches
  IV. Applications | Custom Statistical or Symbolic: 
     - Domain-specific model adaptation to financial text
     - Instruction tuning for financial QA
     - Evaluation metrics for model performance

* System Information:
  - Windows OS Terminal
  - CUDA-enabled
  - GPU: NVIDIA RTX 4060
  - GPU Memory: 8GB
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import get_scheduler
from datasets import load_dataset
from tqdm import tqdm
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
model_name = "google/flan-t5-small"
output_dir = "./finetuned_flant5_full"
learning_rate = 2e-5
batch_size = 4
epochs = 5  # Maximum number of epochs
max_length = 512

# Early stopping parameters
patience = 2  
early_stopping_threshold = 0.01 

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("FinLang/investopedia-instruction-tuning-dataset", split='train')
print(f"Dataset loaded: {len(dataset)} examples")

# Take 30% of the data
sample_size = 0.3
sample_indices = random.sample(range(len(dataset)), int(len(dataset) * sample_size))
dataset = dataset.select(sample_indices)
print(f"Using {len(dataset)} examples for training ({sample_size*100}% of the dataset)")

# Split into train and validation sets
train_val_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split['train']
val_dataset = train_val_split['test']
print(f"Training set: {len(train_dataset)} examples")
print(f"Validation set: {len(val_dataset)} examples")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)

def preprocess_function(examples):
    """Process the dataset for Flan-T5."""
    inputs = []
    targets = []
    
    for i in range(len(examples['Context'])):
        context = examples['Context'][i]
        question = examples['Question'][i]
        answer = examples['Answer'][i]
        
        # Skip incomplete examples
        if not context or not question or not answer:
            continue

        # Flan-T5 works well with natural language instructions
        input_text = f"Answer the following question based on the given context. Context: {context} Question: {question}"
        target_text = answer
        
        inputs.append(input_text)
        targets.append(target_text)
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        padding="max_length",
        truncation=True
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_length // 2,  # Shorter max length for answers
            padding="max_length",
            truncation=True
        )
    
    # Set up labels
    model_inputs["labels"] = labels["input_ids"]
    
    # Replace padding token IDs with -100 in labels
    for i in range(len(model_inputs["labels"])):
        model_inputs["labels"][i] = [
            -100 if token == tokenizer.pad_token_id else token 
            for token in model_inputs["labels"][i]
        ]
    
    return model_inputs

# Process the datasets
print("Processing datasets...")
train_encodings = train_dataset.map(preprocess_function, batched=True)
val_encodings = val_dataset.map(preprocess_function, batched=True)

# Remove columns we don't need
columns_to_remove = train_dataset.column_names
train_encodings = train_encodings.remove_columns(columns_to_remove)
val_encodings = val_encodings.remove_columns(columns_to_remove)

# Set the format for PyTorch
train_encodings.set_format("torch")
val_encodings.set_format("torch")

# Create data loaders
train_loader = DataLoader(train_encodings, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_encodings, batch_size=batch_size)

# Prepare optimizer and scheduler
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01,
    eps=1e-8
)

total_steps = len(train_loader) * epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Initialize lists to store loss values for plotting
train_losses = []
val_losses = []
step_losses = []  
lr_values = []    

# Function to plot losses
def plot_losses(train_losses, val_losses, step_losses=None, lr_values=None):
    """Plot training and validation losses, and optionally step losses and learning rate."""
    # Create a figure with specified size
    plt.figure(figsize=(12, 8))
    
    # Plot epoch losses
    plt.subplot(2, 1, 1)
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot step losses if provided
    if step_losses:
        plt.subplot(2, 1, 2)
        steps_range = range(1, len(step_losses) + 1)
        plt.plot(steps_range, step_losses, 'go-', alpha=0.5)
        plt.title('Loss per Step')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        
        # Add learning rate on secondary y-axis if provided
        if lr_values:
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(steps_range, lr_values, 'm-', alpha=0.5)
            ax2.set_ylabel('Learning Rate', color='m')
            ax2.tick_params(axis='y', labelcolor='m')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()

# Training loop with loss tracking
def train_epoch(epoch):
    model.train()
    total_loss = 0
    valid_batches = 0
    epoch_step_losses = []
    epoch_lr_values = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Clear gradients
        optimizer.zero_grad()
        
        try:
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf loss detected at batch {batch_idx}, skipping")
                continue
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            
            # Track current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            epoch_lr_values.append(current_lr)
            
            # Update scheduler
            scheduler.step()
            
            # Track loss
            current_loss = loss.item()
            total_loss += current_loss
            valid_batches += 1
            epoch_step_losses.append(current_loss)
            
            # Update progress bar
            progress_bar.set_postfix({"loss": current_loss})
            
            # Save a checkpoint occasionally
            if (batch_idx + 1) % 500 == 0:
                checkpoint_dir = f"{output_dir}/checkpoint-{epoch+1}-{batch_idx+1}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint to {checkpoint_dir}")
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Compute average loss
    avg_loss = total_loss / max(1, valid_batches)
    
    # Update global loss tracking
    step_losses.extend(epoch_step_losses)
    lr_values.extend(epoch_lr_values)
    
    return avg_loss

# Evaluation loop
def evaluate():
    model.eval()
    total_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluation")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            try:
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs.loss
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Track loss
                current_loss = loss.item()
                total_loss += current_loss
                valid_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": current_loss})
            
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    # Compute average loss
    avg_loss = total_loss / max(1, valid_batches)
    return avg_loss

# Function to test the model
def test_model(context, question):
    model.eval()
    
    # Format input according to our preprocessing
    input_text = f"Answer the following question based on the given context. Context: {context} Question: {question}"
    
    # Tokenize
    inputs = tokenizer(
        input_text, 
        return_tensors="pt",
        max_length=max_length,
        truncation=True
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=2,
            early_stopping=True
        )
    
    # Decode
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.start_time = None
        
    def __call__(self, val_loss, model, epoch, output_dir):
        if self.start_time is None:
            self.start_time = time.time()
            
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            
            # Save the best model
            best_model_dir = f"{output_dir}/best_model"
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            print(f"New best model saved to {best_model_dir} with validation loss: {val_loss:.4f}")
            
            return False
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                training_time = time.time() - self.start_time
                print(f"Early stopping triggered after {epoch+1} epochs, {training_time/60:.2f} minutes")
                self.early_stop = True
                return True
            return False

# Main training loop
print("Starting training...")
start_time = time.time()

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize early stopping
early_stopping = EarlyStopping(patience=patience, min_delta=early_stopping_threshold)

for epoch in range(epochs):
    epoch_start = time.time()
    
    # Training
    train_loss = train_epoch(epoch)
    train_losses.append(train_loss)
    print(f"Training loss: {train_loss:.4f}")
    
    # Validation
    val_loss = evaluate()
    val_losses.append(val_loss)
    print(f"Validation loss: {val_loss:.4f}")
    
    # Plot and save losses after each epoch
    plot_losses(train_losses, val_losses, step_losses, lr_values)
    print(f"Loss curves updated and saved to {output_dir}/loss_curves.png")
    
    # Save checkpoint
    model.save_pretrained(f"{output_dir}/checkpoint-{epoch+1}")
    tokenizer.save_pretrained(f"{output_dir}/checkpoint-{epoch+1}")
    print(f"Saved checkpoint to {output_dir}/checkpoint-{epoch+1}")
    
    # Test the model with an example
    test_context = "Stock market indices measure the value of a section of the stock market."
    test_question = "What is a stock market index?"
    answer = test_model(test_context, test_question)
    print(f"\nTest question: {test_question}")
    print(f"Model answer: {answer}\n")
    
    # Report epoch time
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} completed in {epoch_time/60:.2f} minutes")
    
    # Check for early stopping
    if early_stopping(val_loss, model, epoch, output_dir):
        print("Early stopping triggered. Training stopped.")
        break

# If we didn't stop early and the final model is better than the best model, save it
if not early_stopping.early_stop and val_losses[-1] <= early_stopping.best_loss:
    # Save final model as best
    best_model_dir = f"{output_dir}/best_model"
    os.makedirs(best_model_dir, exist_ok=True)
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"Final model saved as best model with validation loss: {val_losses[-1]:.4f}")

# Final save
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Create final loss plot
plot_losses(train_losses, val_losses, step_losses, lr_values)
print(f"Final loss curves saved to {output_dir}/loss_curves.png")

total_time = time.time() - start_time
print(f"Training complete in {total_time/60:.2f} minutes")
print(f"Best validation loss: {early_stopping.best_loss:.4f}")
print(f"Final model saved to {output_dir}")

# Test the final model
test_cases = [
    {
        "context": "Stock market indices measure the value of a section of the stock market. They are computed from the prices of selected stocks.",
        "question": "What is a stock market index and why is it important for investors?"
    },
    {
        "context": "Exchange-traded funds (ETFs) are a type of investment fund and exchange-traded product traded on stock exchanges. ETFs are similar in many ways to mutual funds, except that ETFs are bought and sold throughout the day on stock exchanges while mutual funds are bought and sold based on their price at day's end.",
        "question": "How do ETFs work and what advantages do they offer compared to mutual funds?"
    }
]

print("\nTesting the final model:")
for i, test in enumerate(test_cases):
    print(f"\nTest Case {i+1}:")
    print(f"Question: {test['question']}")
    answer = test_model(test["context"], test["question"])
    print(f"Generated Answer: {answer}")