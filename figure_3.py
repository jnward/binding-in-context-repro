# %%
%load_ext autoreload
%autoreload 2
import torch
from transformer_lens import HookedTransformer, ActivationCache
from tasks.capitals import CAPITAL_MAP, NAMES, capitals_generator
from functools import partial
from tqdm import tqdm

device = "cuda"

import os
# %%
model = HookedTransformer.from_pretrained_no_processing(
    # "pythia-12B",
    "meta-llama/Llama-3.2-3B",
    device=device,
    dtype=torch.bfloat16
)

# %%
hooks_of_interest = []
for hook in model.hook_dict.keys():
    if "block" in hook and "resid" in hook:
        hooks_of_interest.append(hook)
    # else:
    #     if source_cache[hook].shape[:2] == torch.Size([1, 16]):
    #         hooks_of_interest[hook] = source_cache[hook]

# pythia
# E_0_POS = 18
# E_1_POS = 27
# A_0_POS = 25
# A_1_POS = 34

# CONTEXT_LENGTH = 36

# llama 3.2
E_0_POS = 17
E_1_POS = 26
A_0_POS = 24
A_1_POS = 33

CONTEXT_LENGTH = 35

# %%
# we run a forward pass on the query sentence, patching in _all_ of the activations
# from the target context. We also patch in activations from the source context only
# at pos.
def patch_all_acts_at_positions(target_ids: torch.Tensor, source_cache: ActivationCache, target_cache: ActivationCache, positions: list[int]):
    def position_patch_hook(activation: torch.Tensor, hook, hook_name: str, positions: list[int]):

        source_acts = source_cache[hook_name]
        target_acts = target_cache[hook_name]
        activation[:, :CONTEXT_LENGTH] = target_acts
        for pos in positions:
            activation[:, pos] = source_acts[:, pos]
        return activation
    
    corrupt_logits = model.run_with_hooks(
        target_ids,
        fwd_hooks=[
            (hook_name, partial(position_patch_hook, hook_name=hook_name, positions=positions)) for hook_name in hooks_of_interest
        ]
    )

    return corrupt_logits

# %%
my_capitals_generator = capitals_generator()
capitals_examples = [
    next(my_capitals_generator) for _ in range(64)
]

# %%
my_example = capitals_examples[0]

print(my_example.context)
print(my_example.query_E_0)
print(my_example.query_E_1)
print(my_example.context_p)
print(my_example.query_E_0p)
print(my_example.query_E_1p)

tokenized = model.tokenizer.encode(my_example.context, return_tensors='pt')

for i, token in enumerate(tokenized.squeeze()):
    print(i, token, model.tokenizer.decode(token))

# %%

def get_logit_matrices(test_example, model, change_right=False):
    # Move cleanup into the function
    def cleanup_tensors():
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # Process inputs
    target_context_ids = model.tokenizer.encode(test_example.context, return_tensors="pt").to(device)
    source_context_ids = model.tokenizer.encode(test_example.context_p, return_tensors="pt").to(device)
    
    # Run caches and immediately process
    _, target_cache = model.run_with_cache(target_context_ids)
    _, source_cache = model.run_with_cache(source_context_ids)
    
    # Clear encoded ids we don't need anymore
    del target_context_ids
    del source_context_ids
    cleanup_tensors()

    # Process query and answer tokens all at once to avoid multiple encode operations
    queries = [test_example.query_E_0, test_example.query_E_1, test_example.query_E_0p, test_example.query_E_1p]
    answers = [test_example.answer_0, test_example.answer_1, test_example.answer_0p, test_example.answer_1p]

    query_token_ids = torch.stack([
        model.tokenizer.encode(query, return_tensors="pt", add_special_tokens=False).squeeze() 
        for query in queries
    ]).to(device)
    
    answer_token_ids = torch.stack([
        model.tokenizer.encode(f" {answer}", return_tensors="pt", add_special_tokens=False).squeeze() 
        for answer in answers
    ]).to(device)

    target_mask_ids = torch.ones((1, CONTEXT_LENGTH), device=device, dtype=torch.long) * 5
    full_query_ids = torch.cat([target_mask_ids.expand(4, -1), query_token_ids], dim=1)

    # Run all interventions
    patch_configs = [
        [], [A_0_POS], [E_0_POS], [A_0_POS, E_0_POS],
        [A_1_POS], [E_1_POS], [A_1_POS, E_1_POS]
    ]
    
    all_logits = []
    for positions in patch_configs:
        logits = patch_all_acts_at_positions(full_query_ids, source_cache, target_cache, positions)
        probs = logits[:, -1, answer_token_ids].detach()  # Detach here
        all_logits.append(probs)
        del logits
        cleanup_tensors()
    
    # Clean up remaining caches and tensors
    del source_cache
    del target_cache
    del full_query_ids
    del query_token_ids
    
    # Stack results and move to CPU to free GPU memory
    result = torch.stack(all_logits).cpu()
    del all_logits
    cleanup_tensors()
    
    return result

# %%
import gc
def cleanup():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    
# %%

all_logits = []
for example in tqdm(capitals_examples):
    try:
        logit_matrix = get_logit_matrices(example, model)
        all_logits.append(logit_matrix)
    except Exception as e:
        print(f"Error processing example: {e}")
    finally:
        cleanup()

# %%
all_avg = torch.stack(all_logits).mean(0)
all_avg = all_avg.float() - all_avg.max() + 1

left_avg = all_avg[[0, 1, 2, 3]]
right_avg = all_avg[[0, 4, 5, 6]]

# %%
import plotly.subplots as sp
import plotly.graph_objects as go

# Create labels
query_labels = ['E₀', 'E₁', 'E′₀', 'E′₁'][::-1]
attr_labels = ['A₀', 'A₁', 'A′₀', 'A′₁']
titles = ['None', 'Attribute', 'Entity', 'Both']

# Create 2x2 subplot
fig = sp.make_subplots(rows=2, cols=2, 
                      subplot_titles=titles,
                      x_title="Attributes",
                      y_title="Query name")

# Add each heatmap
for idx, matrix in enumerate(left_avg):
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    fig.add_trace(
        go.Heatmap(
            z=matrix.cpu().numpy()[::-1],
            x=attr_labels,
            y=query_labels,
            text=[[f'{x:.2f}' for x in r] for r in matrix.cpu().numpy()][::-1],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorscale='RdBu_r',
            zmid=0
        ),
        row=row, col=col
    )
    
    # Add red boxes for cells of interest
    # For (E₀, A₀)
    fig.add_shape(
        type="rect",
        x0=-0.5 + (col-1) * 2, x1=0.5 + (col-1) * 2,  # A₀
        y0=2.5 - (row-1) * 2, y1=3.5 - (row-1) * 2,  # E₀
        line=dict(color="red", width=2),
        row=row, col=col,
        fillcolor="rgba(0,0,0,0)"  # Transparent fill
    )
    
    # For (E1, A1)
    fig.add_shape(
        type="rect",
        x0=0.5, x1=1.5,  # A1
        y0=1.5, y1=2.5,  # E1
        line=dict(color="red", width=2),
        row=row, col=col,
        fillcolor="rgba(0,0,0,0)"  # Transparent fill
    )

# Update layout
fig.update_layout(
    height=600,
    width=514,
    showlegend=False,
    title_text="Swapping entity/attribute for (E₀, A₀)"
)

fig.update_xaxes(scaleanchor="y", scaleratio=1)
fig.update_yaxes(scaleanchor="x", scaleratio=1)


# Make sure labels are shown on all subplots
fig.update_xaxes(tickangle=0)
fig.update_yaxes(tickangle=0)

fig.show()

# %%
import plotly.subplots as sp
import plotly.graph_objects as go

# Create labels
query_labels = ['E₀', 'E₁', 'E′₀', 'E′₁'][::-1]
attr_labels = ['A₀', 'A₁', 'A′₀', 'A′₁']
titles = ['None', 'Attribute', 'Entity', 'Both']

# Create 2x2 subplot
fig = sp.make_subplots(rows=2, cols=2, 
                      subplot_titles=titles,
                      x_title="Attributes",
                      y_title="Query name")

# Add each heatmap
for idx, matrix in enumerate(right_avg):
    row = idx // 2 + 1
    col = idx % 2 + 1
    
    fig.add_trace(
        go.Heatmap(
            z=matrix.cpu().numpy()[::-1],
            x=attr_labels,
            y=query_labels,
            text=[[f'{x:.2f}' for x in r] for r in matrix.cpu().numpy()][::-1],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorscale='RdBu_r',
            zmid=0
        ),
        row=row, col=col
    )
    
    # Add red boxes for cells of interest
    # For (E₀, A₀)
    fig.add_shape(
        type="rect",
        x0=0.5 + (col-1) * 2, x1=1.5 + (col-1) * 2,  # A₀
        y0=1.5 - (row-1) * 2, y1=2.5 - (row-1) * 2,  # E₀
        line=dict(color="red", width=2),
        row=row, col=col,
        fillcolor="rgba(0,0,0,0)"  # Transparent fill
    )
    
    # For (E0, A0)
    fig.add_shape(
        type="rect",
        x0=-0.5, x1=0.5,  # A0
        y0=2.5, y1=3.5,  # E0
        line=dict(color="red", width=2),
        row=row, col=col,
        fillcolor="rgba(0,0,0,0)"  # Transparent fill
    )

# Update layout
fig.update_layout(
    height=600,
    width=514,
    showlegend=False,
    title_text="Swapping entity/attribute for (E1, A1)"
)

fig.update_xaxes(scaleanchor="y", scaleratio=1)
fig.update_yaxes(scaleanchor="x", scaleratio=1)


# Make sure labels are shown on all subplots
fig.update_xaxes(tickangle=0)
fig.update_yaxes(tickangle=0)

fig.show()

# %%
