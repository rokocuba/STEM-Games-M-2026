import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load a small, free model locally
model_id = "gpt2" # You can also use small modern models like "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def calculate_perplexity(text):
    # 2. Convert the text into numbers (tokens)
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        # 3. Ask the model to calculate the loss (how surprised it is)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
    # 4. Calculate Perplexity (e^loss)
    perplexity = math.exp(loss.item())
    return perplexity

# 5. Test it!
print(calculate_perplexity("The quick brown fox jumps over the lazy dog.")) # Low score (Human-like / predictable)
print(calculate_perplexity("The quick brown potato flies over the lazy galaxy.")) # High score (Unexpected)

