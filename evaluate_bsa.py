from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from time import perf_counter
import json
from tqdm import tqdm

model_name = "gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name) 
tokenizer = AutoTokenizer.from_pretrained(model_name)
ds = load_dataset("")

prompts = []
for split in ds:
    row = ds[split].select(range(0))
    prompt = row["context"] + "\n" + row["prompt"]
    prompts.append(prompt)

num_prompts = len(prompts)
mean_squared_error = [{}]

baseline_attention_results = []
baseline_attention_time = 0.0

sparsities = [0.75, 0.5, 0.25, 0.2, 0.15, 0.1, 0.05]

attention_approaches = [

] # Here, block size is 64

mse = {
    sparsity: {
        approach: [] for approach in list(attention_approaches.keys())
    } for sparsity in sparsities
}

for start in range(0, num_prompts, 4):
    end = min(start + 10, num_prompts)
    batch = prompts[start:end]
    
    # Perform a single forward pass through the model
    batch_tokenized = tokenizer(batch, return_tensors="pt", padding=True)
    print("Tokenized matrix: ", batch_tokenized)
    start_time = perf_counter()
    state = model.model.embed_tokens(batch_tokenized)
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.model.layers):
            state = layer(state)
            print(f"State at layer {layer_idx} ", state)
            baseline_attention_results.append(state)
    baseline_attention_time = perf_counter() - start_time
    
    # then, evaluate the attention mechanisms using various monkeypatches
    for approach in tqdm(attention_approaches):
        # TODO: apply the monkeypatch to the model
        start_time = perf_counter()
        state = model.model.embed_tokens(batch_tokenized)
        with torch.no_grad():
            for layer_idx, layer in enumerate(model.model.layers):
                state = layer(state)
                print(f"State at layer {layer_idx} ", state)
                # Calculate the mean squared error between the baseline and the monkeypatched attention mechanisms
                baseline_attention_results.append(state)
        end_time = perf_counter() - start_time

with open("bsa_results.json", "w") as f:
    json.dump(mse, f)