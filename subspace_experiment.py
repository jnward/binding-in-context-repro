# %%
# %load_ext autoreload
# %autoreload 2
import torch
from transformer_lens import HookedTransformer, ActivationCache
from tasks.capitals import CAPITAL_MAP, NAMES, capitals_generator
from functools import partial
from tqdm import tqdm

device = "mps"
torch.set_grad_enabled(False)

import os
os.environ["HF_TOKEN"] = "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"

# %%
# Load the model
model = HookedTransformer.from_pretrained(
    "pythia-1B",
    # "meta-llama/Llama-3.2-1B",
    # "gemma-2-2b",
    device=device,
    dtype=torch.bfloat16,
)

# %%
# We want to get all the pre-transformer activations in the residual stream
hooks_of_interest = []
for hook in model.hook_dict.keys():
    if "hook_resid_pre" in hook:
        hooks_of_interest.append(hook)

LAYERS = [10]
hooks_of_interest = hooks_of_interest[LAYERS[0]:LAYERS[-1] + 1]

# N is the number of binding examples in each context
N = 13

# Need to keep track of where in the context each entity/attribute is.
# pythia and gemma 2
E_0_POS = 18
A_0_POS = 25

N_LENGTH = 9

CONTEXT_LENGTH = 18 + N_LENGTH * N

# llama 3.2
# E_0_POS = 17
# A_0_POS = 24

# N_LENGTH = 9

# CONTEXT_LENGTH = 17 + N_LENGTH * N

# %%
my_capitals_generator = capitals_generator(n=N)
capitals_examples = [next(my_capitals_generator) for _ in range(1000)]

print(capitals_examples[0].context)
print(capitals_examples[0].query_E_0)
print(capitals_examples[0].answers)

tokenized = model.tokenizer.encode(capitals_examples[0].context, return_tensors="pt")

for i, token in enumerate(tokenized[0]):
    print(i, model.tokenizer.decode(token))



# %%
def get_activation_difference(
    context: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Given some context, return the difference in activations between
    the zeroth instance of entities and attributes, and every other
    instance.

    Args:
        context (str): The context to query

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
            The activation differences. First is entities, second is the
            token after an entity's token, last is attributes.

            Each of the three lists has N-1 elements, where N is the number
            of binding examples in the context. Each element has shape
            [n_layers, d_model].
    """
    tokenized = model.tokenizer.encode(context, return_tensors="pt")
    tokenized = tokenized.to(device)
    _, cache = model.run_with_cache(tokenized)
    E_acts = [[] for _ in range(N)]
    E_next_acts = [[] for _ in range(N)]
    A_acts = [[] for _ in range(N)]

    for hook in hooks_of_interest:
        acts = cache[hook].squeeze()

        # normalize activations
        acts /= acts.norm(2, -1, keepdim=True)
        for i in range(N):
            E_idx = E_0_POS + i * N_LENGTH
            E_next_idx = E_0_POS + 1 + i * N_LENGTH
            A_idx = A_0_POS + i * N_LENGTH
            # E_idx = torch.randint(0, CONTEXT_LENGTH, (1,))
            # E_next_idx = torch.randint(0, CONTEXT_LENGTH, (1,))
            # A_idx = torch.randint(0, CONTEXT_LENGTH, (1,))

            E_acts[i].append(acts[E_idx])
            E_next_acts[i].append(acts[E_next_idx])
            A_acts[i].append(acts[A_idx])

    E_acts = torch.stack([torch.stack(acts) for acts in E_acts])
    E_next_acts = torch.stack([torch.stack(acts) for acts in E_next_acts])
    A_acts = torch.stack([torch.stack(acts) for acts in A_acts])

    E_diff = [E_acts[i] - E_acts[0] for i in range(1, N)]
    E_next_diff = [E_next_acts[i] - E_next_acts[0] for i in range(1, N)]
    A_diff = [A_acts[i] - A_acts[0] for i in range(1, N)]

    # C_A_0 + A_diffs[0] => A_0 bound to E_1
    # C_A_1 - A_diffs[0] => A_1 bound to E_0
    return E_diff, E_next_diff, A_diff


# %%
E_diffs = []
E_next_diffs = []
A_diffs = []

# Iterate thru all our examples so we can get mean differences
for example in tqdm(capitals_examples):
    ctx = example.context
    E_diff, E_next_diff, A_diff = get_activation_difference(ctx)
    # E_diff, E_next_diff, A_diff = get_activation_difference(example.context)
    E_diffs.append(E_diff)
    E_next_diffs.append(E_next_diff)
    A_diffs.append(A_diff)

# %%
# For N=13, we get 12 deltas
E_deltas = torch.stack([torch.stack(E_diff) for E_diff in E_diffs]).mean(0)
E_next_deltas = torch.stack(
    [torch.stack(E_next_diff) for E_next_diff in E_next_diffs]
).mean(0)
A_deltas = torch.stack([torch.stack(A_diff) for A_diff in A_diffs]).mean(0)


# %%
# Use my_data = delta_stack to look at each class of binding vector
delta_stack = torch.cat([E_deltas, E_next_deltas, A_deltas])
delta_stack.shape

# %%
my_data = delta_stack
my_data = my_data.reshape(my_data.shape[0], -1)

# %%
U, S, V = my_data.reshape(my_data.shape[0], -1).float().svd()

# %%
import plotly.express as px

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
    print(f"PC {i+1}: {var:.2f}% (Cumulative: {cum_var:.2f}%)")

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
    if V.shape[1] == 36:
        categories = ["E deltas"] * 12 + ["E+1 deltas"] * 12 + ["A deltas"] * 12
        labels = [str(i % 12 + 1) for i in range(36)]
    else:
        categories = ["Deltas"] * 12
        labels = [str(i + 1) for i in range(12)]
    
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

    if V.shape[1] == 36:
        categories = ["E deltas"] * 12 + ["E+1 deltas"] * 12 + ["A deltas"] * 12
        labels = [str(i % 12 + 1) for i in range(36)]
    else:
        categories = ["Deltas"] * 12
        labels = [str(i + 1) for i in range(12)]

    df = pd.DataFrame(
        {
            "Index": range(B),
            "PC1": proj_np[:, 0],
            "PC2": proj_np[:, 1],
            "Category": categories,
            "Label": labels,
        }
    )

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="Index",
        custom_data=["Index"],  # Include Index in hover data
        title="Data Projected onto First Two Principal Components",
        # set rainbow color scale
        color_continuous_scale=px.colors.sequential.Rainbow,
    )

    # fig.update_traces(textposition="top center", marker=dict(size=10))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    fig.show()


project_and_plot(my_data.float(), V.float())

# %%
position_V = np.load("position_PCs.npy")
position_V = torch.Tensor(position_V).to(device)

# %%
def ablate_pcs(data_tensor, V, num_pcs_to_ablate):
    """Remove only the components in the specified PCs from difference vectors"""
    projection = data_tensor @ V[:, :num_pcs_to_ablate] @ V[:, :num_pcs_to_ablate].T
    return data_tensor - projection

my_data_ablated = ablate_pcs(my_data, position_V, 24)
ablated_U, ablated_S, ablated_V = my_data_ablated.reshape(my_data_ablated.shape[0], -1).float().svd()

project_and_plot(my_data_ablated.float(), ablated_V.float())

# %%
# Only keep the non-zero components of the ablated space
basis = torch.cat([ablated_V[:, :3], position_V[:, :2]], dim=1)

# Verify orthogonality
print("Basis orthogonality check:")
print(torch.mm(basis.T, basis))

# Calculate explained variance
total_variance = (my_data**2).sum()
projections = my_data @ basis
variances = (projections**2).sum(dim=0)
explained_variance = (variances / total_variance * 100).cpu()

print("\nVariance explained by each basis vector:")
for i, var in enumerate(explained_variance):
    label = "Binding PC" if i < 2 else "Position PC"
    print(f"{label} {i+1}: {var:.2f}%")
print(f"\nTotal explained variance: {explained_variance.sum():.2f}%")

# %%
# %%
def project_and_plot_3d(data_tensor, V):
    B = data_tensor.shape[0]
    flattened = data_tensor.reshape(B, -1)

    mean = flattened.mean(dim=0, keepdim=True)
    centered = flattened - mean

    projections = centered @ V[:, :3]

    proj_np = projections.cpu().numpy()

    if V.shape[1] == 36:
        categories = ["E deltas"] * 12 + ["E+1 deltas"] * 12 + ["A deltas"] * 12
        labels = [str(i % 12 + 1) for i in range(36)]
    else:
        categories = ["Deltas"] * 12
        labels = [str(i + 1) for i in range(12)]

    df = pd.DataFrame(
        {
            "PC1": proj_np[:, 0],
            "PC2": proj_np[:, 1],
            "PC3": proj_np[:, 2],
            "Category": categories,
            "Label": labels,
        }
    )

    fig = px.scatter_3d(
        df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Category",
        text="Label",
        title="Data Projected onto First Three Principal Components",
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    fig.update_traces(marker=dict(size=8), textposition="top center")
    fig.update_layout(
        scene=dict(
            aspectmode="cube",  # Make the plot cubic
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    fig.show()

project_and_plot_3d(my_data.float(), ablated_V.float())

# %%
