# %%
%load_ext autoreload
%autoreload 2
import torch
from transformer_lens import HookedTransformer, ActivationCache
from tasks.capitals import CAPITAL_MAP, NAMES, capitals_generator
from functools import partial
from tqdm import tqdm

torch.set_grad_enabled(False)

device = "mps"

import os

# %%
model = HookedTransformer.from_pretrained_no_processing(
    "pythia-1B",
    # "meta-llama/Llama-3.2-3B",
    # "gemma-2-27b",
    device=device,
    dtype=torch.bfloat16
)

# %%
hooks_of_interest = []
for hook in model.hook_dict.keys():
    if "hook_resid_pre" in hook:
        hooks_of_interest.append(hook)



# %%
test_context = """\
Repeat the following lists of numbers.
 2 6 4 8 1 => 2 6 4 8 1
 9 3 0 5 7 => 9 3 0 5 7
 6 3 9 1 4"""

# %%

tokenized = model.tokenizer.encode(test_context, return_tensors='pt').to(device)

for i, token in enumerate(tokenized.squeeze()):
    print(i, token, repr(model.tokenizer.decode(token)))

CONTEXT_LENGTH = 39

# %%
def move_elements(tensor, i, j, dim=1):
    if i == j:
        return tensor
    
    indices = list(range(tensor.size(dim)))
    indices.pop(i)
    indices.insert(j, i)
    
    return tensor.index_select(dim, torch.tensor(indices, device=tensor.device))

def swap_elements(tensor, i, j, dim=1):
    if i == j:
        return tensor
    
    # Create list of indices
    indices = list(range(tensor.size(dim)))
    # Swap the positions
    indices[i], indices[j] = indices[j], indices[i]
    
    return tensor.index_select(dim, torch.tensor(indices, device=tensor.device))

# %%
def permute_acts(target_ids: torch.Tensor, target_cache: ActivationCache, i: int, j: int):
    # move dim i to j
    def position_patch_hook(activation: torch.Tensor, hook, hook_name: str, i: int, j: int):
        target_acts = target_cache[hook_name]
        # if hook_name.split(".")[1] == "0":
            # activation[:, :CONTEXT_LENGTH] = move_elements(target_acts, i, j, dim=1)
        # activation[:, :CONTEXT_LENGTH] = move_elements(target_acts, i, j, dim=1)
        activation[:, :CONTEXT_LENGTH] = swap_elements(target_acts, i, j, dim=1)
        return activation
    
    corrupt_logits = model.run_with_hooks(
        target_ids,
        fwd_hooks=[
            (hook_name, partial(position_patch_hook, hook_name=hook_name, i=i, j=j)) for hook_name in hooks_of_interest
        ]
    )

    return corrupt_logits


# %%
_, target_cache = model.run_with_cache(
    tokenized
)

# %%
query = " =>"
query_ids = model.tokenizer.encode(query, return_tensors='pt', add_special_tokens=False).to(device)
full_ids = torch.cat([tokenized, query_ids], dim=1)

i = 38
j = 34
print(model.tokenizer.decode(move_elements(tokenized, i, j).squeeze()[-5:]))

for _ in range(5):
    corrupt_logits = permute_acts(full_ids, target_cache, i, j)[0, -1, :]
    generated_token = corrupt_logits.argmax(keepdim=True)
    print(model.tokenizer.decode(generated_token.squeeze()), end=', ')
    print(corrupt_logits[generated_token.item()].item())
    full_ids = torch.cat([full_ids, generated_token[None, :]], dim=1)

print()

test_tokens = torch.cat([tokenized, query_ids], dim=1)

for _ in range(5):
    # logits = model(move_elements(test_tokens, i, j))[0, -1, :]
    logits = model(test_tokens)[0, -1, :]
    generated_token = logits.argmax(keepdim=True)
    print(model.tokenizer.decode(generated_token.squeeze()), end=', ')
    print(logits[generated_token.item()].item())
    test_tokens = torch.cat([test_tokens, generated_token[None, :]], dim=1)





# %%
full_ids = torch.cat([tokenized, query_ids], dim=1)
moved_tokenized = move_elements(full_ids, 36, 34)

for _ in range(5):
    logits = model(moved_tokenized)[0, -1, :]
    generated_token = logits.argmax(keepdim=True)
    print(model.tokenizer.decode(generated_token.squeeze()))
    moved_tokenized = torch.cat([moved_tokenized, generated_token[None, :]], dim=1)




# %%
# pythia and gemma 2
E_0_POS = 18
E_1_POS = 27
A_0_POS = 25
A_1_POS = 34

CONTEXT_LENGTH = 36

# llama 3.2
# E_0_POS = 17
# E_1_POS = 26
# A_0_POS = 24
# A_1_POS = 33

# CONTEXT_LENGTH = 35

# %%
my_capitals_generator = capitals_generator()
capitals_examples = [
    next(my_capitals_generator) for _ in range(500)
]

# %%
my_example = capitals_examples[0]

print(my_example.context)

from collections import defaultdict
import gc

out = defaultdict(list)

def cleanup_tensors():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

for example in capitals_examples[:10]:
    context_ids = model.tokenizer.encode(example.context, return_tensors='pt').to(device)
    _, target_cache = model.run_with_cache(context_ids)

    queries = [example.query_E_0, example.query_E_1]
    answers = [example.answer_0, example.answer_1]

    query_token_ids = torch.stack([
        model.tokenizer.encode(query, return_tensors="pt", add_special_tokens=False).squeeze() 
        for query in queries
    ]).to(device)

    answer_token_ids = torch.stack([
        model.tokenizer.encode(f" {answer}", return_tensors="pt", add_special_tokens=False).squeeze() 
        for answer in answers
    ]).to(device)
    
    full_query_ids = torch.cat([context_ids.expand(2, -1), query_token_ids], dim=1)

    for i in range(36):
        print(i)
        corrupt_logits = permute_acts(full_query_ids, target_cache, E_0_POS, i)[:, -1, :]
        answer_logits = corrupt_logits[:, answer_token_ids].detach().cpu()
        out[i].append(answer_logits)

        del corrupt_logits
        cleanup_tensors()


# %%
import plotly.express as px
import pandas as pd

positions = []
values = []
line_types = []

# Process the dictionary of tensor lists
for i, tensor_list in out.items():
    # Stack all tensors in the list and compute mean
    avg_tensor = torch.stack(tensor_list).mean(dim=0).float()
    
    # Convert to numpy for easier processing
    avg_tensor_np = avg_tensor.cpu().numpy()
    
    # Add data points for each element in the 2x2 tensor
    positions.extend([i] * 4)  # Each position appears 4 times
    values.extend([
        avg_tensor_np[0, 0],  # First row, first column
        avg_tensor_np[0, 1],  # First row, second column
        avg_tensor_np[1, 0],  # Second row, first column
        avg_tensor_np[1, 1]   # Second row, second column
    ])
    line_types.extend([
        "Query 0, Answer 0",
        "Query 0, Answer 1",
        "Query 1, Answer 0",
        "Query 1, Answer 1"
    ])

# Create DataFrame for plotting
df = pd.DataFrame({
    'Position': positions,
    'Value': values,
    'Line': line_types
})

# Create the plot
fig = px.line(
    df,
    x='Position',
    y='Value',
    color='Line',
    title='Swapping position of E0',
    labels={
        'Position': 'Position Index (i)',
        'Value': 'Average Value',
        'Line': 'Tensor Position'
    }
)

# Add vertical lines
fig.add_vline(x=E_0_POS, line_dash="dash", line_color="gray", annotation_text="control condition", annotation_position="top")
fig.add_vline(x=E_1_POS, line_dash="dash", line_color="green", annotation_text="swap condition", annotation_position="top")

# Update layout for better readability
fig.update_layout(
    xaxis_title="Position Index (i)",
    yaxis_title="Average Value",
    legend_title="Tensor Position",
    hovermode='x unified'
)

# Show the plot
fig.show()
# %%
