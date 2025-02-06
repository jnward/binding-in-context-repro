# %%
# %load_ext autoreload
# %autoreload 2
import torch
from transformer_lens import HookedTransformer, ActivationCache
from tasks.capitals import CAPITAL_MAP, NAMES, capitals_generator
from functools import partial
from tqdm import tqdm

device = "cuda"
torch.set_grad_enabled(False)

import os
os.environ["HF_TOKEN"] = "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"

model_name = "gemma-2-2b"

# %%
# Load the model
model = HookedTransformer.from_pretrained(
    # "pythia-1B",
    # "meta-llama/Llama-3.2-1B",
    model_name,
    device=device,
    dtype=torch.bfloat16,
    center_unembed=False,
    center_writing_weights=False,
)

# %%
# We want to get all the pre-transformer activations in the residual stream
hooks_of_interest = []
for hook in model.hook_dict.keys():
    if "hook_resid_pre" in hook:
        hooks_of_interest.append(hook)

# %%

LAYERS = [20]
hooks_of_interest = hooks_of_interest[LAYERS[0]:LAYERS[-1] + 1]
hooks_of_interest

# %%
def get_activation_difference(
    tokenized: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    _, cache = model.run_with_cache(tokenized)
    all_acts = []

    for hook in hooks_of_interest:
        acts = cache[hook].squeeze()

        # normalize activations
        acts /= acts.norm(2, -1, keepdim=True)

        all_acts.append(acts)

    all_acts = torch.stack(all_acts)  # (n_layers, ctx_len, d_model)
    diffs = torch.stack([all_acts[:, i] - all_acts[:, 0] for i in range(1, tokenized.shape[1])])
    return diffs  # (ctx_len - 1, n_layers, d_model)


# %%
import datasets
my_dataset = datasets.load_dataset(
    "monology/pile-uncopyrighted",
    split="train",
    # shuffle=True,
    streaming=True,
)
my_data_iter = iter(my_dataset)

# %%

diffs = []
n_examples = 6000
for _ in tqdm(range(n_examples)):
    # tokenized = torch.randint(1, 50265, (1, 200))
    # ctx = model.tokenizer.decode(tokenized[0])
    # tokenized = model.tokenizer.encode(ctx, return_tensors="pt").to(device)[:, :128]
    example = next(my_data_iter)
    tokenized = model.tokenizer.encode(example["text"], return_tensors="pt").to(device)
    tokenized = tokenized[:, :128]
    if tokenized.shape[1] < 128:
        continue
    diff = get_activation_difference(tokenized)
    diffs.append(diff)

# %%
diffs = torch.stack(diffs)
deltas = diffs.mean(0)
deltas.shape
# %%
import plotly.express as px

# deltas_norm = deltas.norm(dim=2)
# fig = px.line(deltas_norm.float().cpu())
# fig.show()

# deltas_normed = deltas / deltas_norm[None, :, None]
# deltas_normed_norm = deltas_normed.mean(0).norm(dim=1)
# fig = px.line(deltas_normed_norm.float().cpu())
# fig.show()

# %%
my_data = deltas
my_data.shape
# %%
U, S, V = my_data.reshape(my_data.shape[0], -1).float().svd()

# %%
import plotly.express as px

# plot "weights" of each element of the first two singular vectors
# px.line(V.float().cpu()[:, :2])


# %%
# claude wrote all the rest of this
def compute_variance_explained(S):
    # Square the singular values to get the eigenvalues
    variances = S**2

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
    print(f"PC {i+1}: {var:.5f}% (Cumulative: {cum_var:.5f}%)")

# %%
import torch
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_all_pcs(data_tensor, V, num_pcs=None):
    """
    Plot individual 1D line plots for each principal component.
    
    Args:
        data_tensor: The input tensor of shape [B, *]
        V: The right singular vectors from SVD
        num_pcs: Number of PCs to plot. If None, plots all PCs.
    """
    B = data_tensor.shape[0]
    flattened = data_tensor.reshape(B, -1)
    print(data_tensor.shape, flattened.shape)
    
    # Center the data
    mean = flattened.mean(dim=0, keepdim=True)
    centered = flattened - mean
    
    # Project onto all PCs
    projections = centered @ V
    proj_np = projections.cpu().numpy()
    
    if num_pcs is None:
        num_pcs = proj_np.shape[1]
    
    # Create subplot grid
    rows = (num_pcs + 2) // 3  # 3 plots per row, rounded up
    cols = min(3, num_pcs)
    
    fig = sp.make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=[f'PC {i+1} - {cumulative_variance[i].cpu().item():.2f}% total variance explained' for i in range(num_pcs)]
    )
    
    # Define categories and labels like in your original code
    n_deltas = data_tensor.shape[0]
    categories = ["Deltas"] * n_deltas
    labels = [str(i + 1) for i in range(n_deltas)]
    
    # Add traces for each PC
    for i in range(num_pcs):
        row = i // 3 + 1
        col = i % 3 + 1
        
        # Create DataFrame for this PC
        df = pd.DataFrame({
            'Index': range(len(proj_np[:, i])),
            'Projection': proj_np[:, i],
            'Category': categories,
            'Label': labels
        })
        
        # Add scatter plot for this PC
        for category in set(categories):
            category_data = df[df['Category'] == category]
            fig.add_trace(
                go.Scatter(
                    x=category_data['Index'],
                    y=category_data['Projection'],
                    mode='lines+markers+text',
                    name=category,
                    text=category_data['Label'],
                    textposition="top center",
                    showlegend=(i == 0),  # Only show legend for first subplot
                ),
                row=row,
                col=col
            )
    
    # Update layout
    fig.update_layout(
        height=300 * rows,  # Adjust height based on number of rows
        width=1200,         # Fixed width
        title_text="Projections onto Principal Components",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Update all subplot axes
    fig.update_xaxes(title_text="Index")
    fig.update_yaxes(title_text="Projection")
    
    return fig

# Example usage:
# fig = plot_all_pcs(my_data.float(), V.float(), num_pcs=10)
# fig.show()



# %%
import torch
import plotly.express as px
import pandas as pd


def project_and_plot(data_tensor, V):
    B = data_tensor.shape[0]
    flattened = data_tensor.reshape(B, -1)

    mean = flattened.mean(dim=0, keepdim=True)
    centered = flattened - mean

    projections = centered @ V[:, :2]
    proj_np = projections.cpu().numpy()

    df = pd.DataFrame(
        {
            "PC1": proj_np[:, 0],
            "PC2": proj_np[:, 1],
            "Point": range(B),
            "Index": [f"Point {i}" for i in range(B)]  # Add index labels for hover
        }
    )

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="Point",
        custom_data=["Index"],  # Include Index in hover data
        title="Data Projected onto First Two Principal Components",
        # set rainbow color scale
        color_continuous_scale=px.colors.sequential.Rainbow,

    )
    
    # Update traces including hover template
    fig.update_traces(
        marker=dict(size=10),
        showlegend=False,
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                      "PC1: %{x:.3f}<br>" +
                      "PC2: %{y:.3f}<br>" +
                      "<extra></extra>"  # This removes the secondary box
    )
    
    return fig



fig = project_and_plot(my_data.float(), V.float())
fig.show()

# %%
# ablate the first 6 principal components from deltas:
def ablate_pcs(data_tensor, V, num_pcs_to_ablate):
    V_ablated = V.clone()
    B = data_tensor.shape[0]

    # Project data onto principal components
    pc_scores = data_tensor.reshape(B, -1) @ V_ablated

    # Zero out the specified components
    pc_scores[:, :num_pcs_to_ablate] = 0

    # Project back to original space
    deltas_ablated = pc_scores @ V_ablated.T

    return deltas_ablated

deltas_ablated = ablate_pcs(my_data.float(), V, 16)

fig = project_and_plot(deltas_ablated.float(), V.float())
# fig.show()# 

ablated_U, ablated_S, ablated_V = deltas_ablated.float().svd()
ablated_variance_explained, ablated_cumulative_variance = compute_variance_explained(ablated_S)
print("Variance explained by each component:")
for i, (var, cum_var) in enumerate(zip(ablated_variance_explained, ablated_cumulative_variance)):
    print(f"PC {i+1}: {var:.2f}% (Cumulative: {cum_var:.5f}%)")

project_and_plot(deltas_ablated.float(), ablated_V.float()).show()
# %%
import numpy as np
np.save(f"{model_name}-position-PCs.npy", V.cpu().numpy())
# %%
