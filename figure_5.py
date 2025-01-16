# %%
# %load_ext autoreload
# %autoreload 2
import torch
from transformer_lens import HookedTransformer, ActivationCache
from tasks.capitals import CAPITAL_MAP, NAMES, capitals_generator
from functools import partial
from tqdm import tqdm

device = "cuda"

import os
os.environ["HF_TOKEN"] = "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"

# %%
model = HookedTransformer.from_pretrained_no_processing(
    "pythia-12B",
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

N = 13

# pythia and gemma 2
E_0_POS = 18
E_1_POS = 27
A_0_POS = 25
A_1_POS = 34

N_LENGTH = 9

CONTEXT_LENGTH = 18 + N_LENGTH * N

# llama 3.2
# E_0_POS = 17
# E_1_POS = 26
# A_0_POS = 24
# A_1_POS = 33

# CONTEXT_LENGTH = 35

# %%


my_capitals_generator = capitals_generator(n=N)
capitals_examples = [
    next(my_capitals_generator) for _ in range(500)
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
my_tokens = model.tokenizer.encode(my_example.context)
print(repr(model.tokenizer.decode(my_tokens[E_0_POS: E_0_POS + 2])))
print(repr(model.tokenizer.decode(my_tokens[E_1_POS: E_1_POS + 2])))

# %%
def get_activation_difference(context: str) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    tokenized = model.tokenizer.encode(context, return_tensors='pt')
    tokenized = tokenized.to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokenized)
    E_acts = [[] for _ in range(N)]
    E_next_acts = [[] for _ in range(N)]
    A_acts = [[] for _ in range(N)]

    for hook in hooks_of_interest:
        acts = cache[hook].squeeze()
        for i in range(N):
            E_acts[i].append(acts[E_0_POS + i * N_LENGTH])
            E_next_acts[i].append(acts[E_0_POS + 1 + i * N_LENGTH])
            A_acts[i].append(acts[A_0_POS + i * N_LENGTH])

    E_acts = [torch.stack(acts) for acts in E_acts]
    E_next_acts = [torch.stack(acts) for acts in E_next_acts]
    A_acts = [torch.stack(acts) for acts in A_acts]

    E_diffs = [E_acts[i] - E_acts[0] for i in range(1, N)]
    E_next_diffs = [E_next_acts[i] - E_next_acts[0] for i in range(1, N)]
    A_diffs = [A_acts[i] - A_acts[0] for i in range(1, N)]

    for i in range(E_0_POS, CONTEXT_LENGTH, N_LENGTH):
        e_token = tokenized[0, i]
        e = model.tokenizer.decode(e_token)
        print(e)

    # C_A_0 + A_diffs[0] => A_0 bound to E_1
    # C_A_1 - A_diffs[0] => A_1 bound to E_0
    return E_diffs, E_next_diffs, A_diffs

# %%
E_diffs = []
E_next_diffs = []
A_diffs = []

for example in tqdm(capitals_examples):
    E_diff, E_next_diff, A_diff = get_activation_difference(example.context)
    E_diffs.append(E_diff)
    E_next_diffs.append(E_next_diff)
    A_diffs.append(A_diff)

# %%

E_deltas = torch.stack([torch.stack(E_diff) for E_diff in E_diffs]).mean(0)
E_next_deltas = torch.stack([torch.stack(E_next_diff) for E_next_diff in E_next_diffs]).mean(0)
A_deltas = torch.stack([torch.stack(A_diff) for A_diff in A_diffs]).mean(0)


# %%
delta_stack = torch.cat([E_deltas, E_next_deltas, A_deltas])
delta_stack.shape

# %%
my_data = A_deltas
my_data = my_data.reshape(my_data.shape[0], -1)
my_data = my_data / my_data.norm(2, 1, keepdim=True)


# %%
U, S, V = my_data.reshape(my_data.shape[0], -1).float().svd()

# %%
U.shape

# %%
px.line(V.float().cpu()[:, :2])

# %%
fig = px.line(
    V.float().cpu()[:, 0].reshape(A_deltas[0].shape).abs().max(1)[0],
    title="Max magnitude of PC0 weights for each layer",
    labels={'value': 'Max Magnitude', 'index': 'Layer'}
)
# fig.s
fig.show()



# %%
def compute_variance_explained(S):
    # Square the singular values to get the eigenvalues
    variances = S ** 2
    
    # Total variance is sum of squared singular values
    total_variance = variances.sum()
    
    # Compute percentage of variance explained by each component
    variance_explained = variances / total_variance * 100
    
    # Compute cumulative variance explained
    cumulative_variance = torch.cumsum(variance_explained, dim=0)
    
    return variance_explained, cumulative_variance

variance_explained, cumulative_variance = compute_variance_explained(S)

print("Variance explained by each component:")
for i, (var, cum_var) in enumerate(zip(variance_explained, cumulative_variance)):
    print(f"PC {i+1}: {var:.2f}% (Cumulative: {cum_var:.2f}%)")

# %%
import torch
import plotly.express as px
import pandas as pd

def project_and_plot(data_tensor, V):
    # Reshape data tensor from (12, 36, 5120) to (12, 36*5120)
    B = data_tensor.shape[0]
    flattened = data_tensor.reshape(B, -1)
    
    # Center the data
    mean = flattened.mean(dim=0, keepdim=True)
    centered = flattened - mean
    
    # Project onto first 2 PCs
    # V should be your right singular vectors from SVD
    projections = centered @ V[:, :2]  # Shape: (12, 2)
    
    # Convert to numpy for plotting
    proj_np = projections.cpu().numpy()
    
    # Create category and label arrays
    if V.shape[1] == 36:
        categories = ['E deltas'] * 12 + ['E+1 deltas'] * 12 + ['A deltas'] * 12
        labels = [str(i % 12 + 1) for i in range(36)]
    else:
        categories = ['Deltas'] * 12
        labels = [str(i) for i in range(12)]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'PC1': proj_np[:, 0],
        'PC2': proj_np[:, 1],
        'Category': categories,
        'Label': labels
    })
    
    # Create scatter plot with color groups
    fig = px.scatter(df, x='PC1', y='PC2', 
                    color='Category',
                    text='Label',
                    title='Data Projected onto First Two Principal Components',
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])  # Nice color scheme
    
    # Update trace settings
    fig.update_traces(textposition='top center', marker=dict(size=10))
    
    # Update layout
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    fig.show()

# Usage:
project_and_plot(my_data.float(), V.float())

# %%
import torch
import plotly.express as px
import pandas as pd

def project_and_plot_3d(data_tensor, V):
    # Reshape data tensor to (36, 36*5120)
    B = data_tensor.shape[0]
    flattened = data_tensor.reshape(B, -1)
    
    # Center the data
    mean = flattened.mean(dim=0, keepdim=True)
    centered = flattened - mean
    
    # Project onto first 3 PCs
    projections = centered @ V[:, :3]  # Shape: (36, 3)
    
    # Convert to numpy for plotting
    proj_np = projections.cpu().numpy()
    
    # Create category and label arrays
    if V.shape[1] == 36:
        categories = ['E deltas'] * 12 + ['E+1 deltas'] * 12 + ['A deltas'] * 12
        labels = [str(i % 12 + 1) for i in range(36)]
    else:
        categories = ['Deltas'] * 12
        labels = [str(i) for i in range(12)]
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'PC1': proj_np[:, 0],
        'PC2': proj_np[:, 1],
        'PC3': proj_np[:, 2],
        'Category': categories,
        'Label': labels
    })
    
    # Create 3D scatter plot with color groups
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3',
                       color='Category',
                       text='Label',
                       title='Data Projected onto First Three Principal Components',
                       color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Update trace settings
    fig.update_traces(
        marker=dict(size=8),
        textposition='top center'
    )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode='cube',  # Make the plot cubic
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3"
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    fig.show()

# Usage:
project_and_plot_3d(my_data.float(), V.float())
#%%

mean_E_diff = E_deltas[0]
mean_E_next_diff = E_next_deltas[0]
mean_A_diff = A_deltas[0]

# %%
E_diff_norms = torch.norm(mean_E_diff, dim=-1)
E_next_diff_norms = torch.norm(mean_E_next_diff, dim=-1)
A_diff_norms = torch.norm(mean_A_diff, dim=-1)

# %%
# r_E = torch.randn(len(E_diff_norms), mean_E_diff.shape[-1], device=device, dtype=E_diff_norms.dtype)
# r_E_next = torch.randn(len(E_next_diff_norms), mean_E_diff.shape[-1], device=device, dtype=E_diff_norms.dtype)
# r_A = torch.randn(len(A_diff_norms), mean_E_diff.shape[-1], device=device, dtype=E_diff_norms.dtype)

# r_E /= torch.norm(r_E, dim=1, keepdim=True)
# r_E_next /= torch.norm(r_E_next, dim=1, keepdim=True)
# r_A /= torch.norm(r_A, dim=1, keepdim=True)

# r_E *= E_diff_norms.unsqueeze(1)
# r_E_next *= E_next_diff_norms.unsqueeze(1)
# r_A *= A_diff_norms.unsqueeze(1)

# mean_E_diff = r_E
# mean_E_next_diff = r_E_next
# mean_A_diff = r_A

# %%
def get_hook_idx(hook_name):
    idx = 0
    for idx, hn in enumerate(hooks_of_interest):
        if hn == hook_name:
            return idx
    raise ValueError(f"hook name {hook_name} not in HoI {hooks_of_interest}")



# %%
def patch_all_acts_at_positions(target_ids: torch.Tensor, target_cache: ActivationCache, vectors: dict[int, torch.Tensor]):
    def position_patch_hook(activation: torch.Tensor, hook, hook_name: str, vectors: dict[int, torch.Tensor]):
        target_acts = target_cache[hook_name]
        activation[:, :CONTEXT_LENGTH] = target_acts

        vec_idx = get_hook_idx(hook_name)
        for pos, vec in vectors.items():
            activation[:, pos] += vec[vec_idx]

        return activation
    
    corrupt_logits = model.run_with_hooks(
        target_ids,
        fwd_hooks=[
            (hook_name, partial(position_patch_hook, hook_name=hook_name, vectors=vectors)) for hook_name in hooks_of_interest
        ]
    )

    return corrupt_logits

# %%
len(A_deltas)


# %%

control_0 = 0
control_1 = 0
attribute_0 = 0
attribute_1 = 0
entity_0 = 0
entity_1 = 0
both_0 = 0
both_1 = 0

n_examples = 100
# check accuracy for task with control
for example in tqdm(capitals_examples[:n_examples]):
    context_plus_E0_query = example.context + example.query_E_0
    context_plus_E1_query = example.context + example.query_E_1

    answer_0 = example.answer_0
    answer_1 = example.answer_1

    answer_token_ids = torch.stack([
        model.tokenizer.encode(f" {answer}", return_tensors="pt", add_special_tokens=False).squeeze() 
        for answer in [answer_0, answer_1]
    ]).to(device)

    tokenized_0 = model.tokenizer.encode(context_plus_E0_query, return_tensors='pt').to(device)
    tokenized_1 = model.tokenizer.encode(context_plus_E1_query, return_tensors='pt').to(device)

    with torch.no_grad():
        _, target_cache = model.run_with_cache(tokenized_0[:, :CONTEXT_LENGTH])
        logits_0 = model(tokenized_0)[0, -1, answer_token_ids]
        logits_1 = model(tokenized_1)[0, -1, answer_token_ids]

        corrupt_A_logits_0 = patch_all_acts_at_positions(tokenized_0, target_cache, {A_0_POS: mean_A_diff, A_1_POS: -mean_A_diff})[0, -1, answer_token_ids]
        tokenized_0[:, :CONTEXT_LENGTH-1] = 3
        z2 = patch_all_acts_at_positions(tokenized_0, target_cache, {A_0_POS: mean_A_diff, A_1_POS: -mean_A_diff})[0, -1, answer_token_ids]
        assert(torch.allclose(corrupt_A_logits_0, z2))
        # print(z2)
        corrupt_A_logits_1 = patch_all_acts_at_positions(tokenized_1, target_cache, {A_0_POS: mean_A_diff, A_1_POS: -mean_A_diff})[0, -1, answer_token_ids]
        corrupt_E_logits_0 = patch_all_acts_at_positions(tokenized_0, target_cache, {E_0_POS: mean_E_diff, E_0_POS + 1: mean_E_next_diff, E_1_POS: -mean_E_diff, E_1_POS + 1: -mean_E_next_diff})[0, -1, answer_token_ids]
        corrupt_E_logits_1 = patch_all_acts_at_positions(tokenized_1, target_cache, {E_0_POS: mean_E_diff, E_0_POS + 1: mean_E_next_diff, E_1_POS: -mean_E_diff, E_1_POS + 1: -mean_E_next_diff})[0, -1, answer_token_ids]
        corrupt_both_logits_0 = patch_all_acts_at_positions(tokenized_0, target_cache, {A_0_POS: mean_A_diff, A_1_POS: -mean_A_diff, E_0_POS: mean_E_diff, E_0_POS + 1: mean_E_next_diff, E_1_POS: -mean_E_diff, E_1_POS + 1: -mean_E_next_diff})[0, -1, answer_token_ids]
        corrupt_both_logits_1 = patch_all_acts_at_positions(tokenized_1, target_cache, {A_0_POS: mean_A_diff, A_1_POS: -mean_A_diff, E_0_POS: mean_E_diff, E_0_POS + 1: mean_E_next_diff, E_1_POS: -mean_E_diff, E_1_POS + 1: -mean_E_next_diff})[0, -1, answer_token_ids]

    control_0 += 1 * logits_0.argmax().item() == 0
    control_1 += 1 * logits_1.argmax().item() == 1
    entity_0 += 1 * corrupt_E_logits_0.argmax().item() == 0
    entity_1 += 1 * corrupt_E_logits_1.argmax().item() == 1
    attribute_0 += 1 * corrupt_A_logits_0.argmax().item() == 0
    attribute_1 += 1 * corrupt_A_logits_1.argmax().item() == 1
    both_0 += 1 * corrupt_both_logits_0.argmax().item() == 0
    both_1 += 1 * corrupt_both_logits_1.argmax().item() == 1

# %%

print(f"Control:\t{control_0 / n_examples:.2f}, {control_1 / n_examples:.2f}")
print(f"Entity:\t{entity_0 / n_examples:.2f}, {entity_1 / n_examples:.2f}")
print(f"Attribute:\t{attribute_0 / n_examples:.2f}, {attribute_1 / n_examples:.2f}")
print(f"Both:\t{both_0 / n_examples:.2f}, {both_1 / n_examples:.2f}")

# %%
