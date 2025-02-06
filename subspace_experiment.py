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
    # "pythia-1B",
    # "meta-llama/Llama-3.2-1B",
    "gemma-2-2b",
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

# prepend zero
E_deltas = torch.cat([torch.zeros_like(E_deltas[:1]), E_deltas])
E_next_deltas = torch.cat([torch.zeros_like(E_next_deltas[:1]), E_next_deltas])
A_deltas = torch.cat([torch.zeros_like(A_deltas[:1]), A_deltas])


# %%
# Use my_data = delta_stack to look at each class of binding vector
delta_stack = torch.cat([E_deltas, E_next_deltas, A_deltas])
delta_stack.shape

# %%
torch.save(delta_stack, "gemma-2-2b-delta-stack.pt")

# %%
my_data = E_deltas
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
    
    # Project onto all PCs
    projections = flattened @ V
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

    projections = flattened @ V[:, :2]
    proj_np = projections.cpu().numpy()

    if V.shape[1] == 36:
        categories = ["E deltas"] * 12 + ["E+1 deltas"] * 12 + ["A deltas"] * 12
        labels = [str(i % 12 + 1) for i in range(36)]
    else:
        categories = ["Deltas"] * B
        labels = [str(i + 1) for i in range(B)]

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
position_V = np.load("gemma-2-2b-position-PCs.npy")
position_V = torch.Tensor(position_V).to(device)

# %%
def ablate_pcs(data_tensor, V, num_pcs_to_ablate):
    """Remove only the components in the specified PCs from difference vectors"""
    projection = data_tensor @ V[:, :num_pcs_to_ablate] @ V[:, :num_pcs_to_ablate].T
    return data_tensor - projection

my_data_ablated = ablate_pcs(my_data, position_V, 24)
ablated_U, ablated_S, ablated_V = my_data_ablated.reshape(my_data_ablated.shape[0], -1).float().svd()

project_and_plot(my_data_ablated.float(), ablated_V.float())

# print the variance explained by each PC in the ablated space
variance_explained, cumulative_variance = compute_variance_explained(ablated_S)
print("Variance explained by each component in ablated space:")
for i, (var, cum_var) in enumerate(zip(variance_explained, cumulative_variance)):
    print(f"PC {i+1}: {var:.2f}% (Cumulative: {cum_var:.2f}%)")

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
    label = "Binding PC" if basis.shape[1] - i > 2 else "Position PC"
    print(f"{label} {i+1}: {var:.2f}%")
print(f"\nTotal explained variance: {explained_variance.sum():.2f}%")

# %%
def project_and_plot_3d(data_tensor, V):
    B = data_tensor.shape[0]
    flattened = data_tensor.reshape(B, -1)

    projections = flattened @ V[:, :3]

    proj_np = projections.cpu().numpy()

    if V.shape[1] == 36:
        categories = ["E deltas"] * 12 + ["E+1 deltas"] * 12 + ["A deltas"] * 12
        labels = [str(i % 12 + 1) for i in range(36)]
    else:
        categories = ["Deltas"] * B
        labels = [str(i + 1) for i in range(B)]

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

def get_pos_acts(
    context: str,
    positions: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    tokenized = model.tokenizer.encode(context, return_tensors="pt")
    tokenized = tokenized.to(device)
    _, cache = model.run_with_cache(tokenized)
    all_acts = [[] for _ in positions]

    for hook in hooks_of_interest:
        acts = cache[hook].squeeze()

        # normalize activations
        acts /= acts.norm(2, -1, keepdim=True)

        for i, pos in enumerate(positions):
            all_acts[i].append(acts[pos])

    all_acts = torch.stack([torch.stack(acts) for acts in all_acts])  # shape: [n_positions, n_layers, d_model]

    return all_acts

# %%

all_acts = []
# all_E_next_acts = []
# all_A_acts = []

for example in tqdm(capitals_examples[:50]):
    acts = get_pos_acts(example.context, [E_0_POS, E_0_POS + 1, A_0_POS])
    all_acts.append(acts)
    # all_E_next_acts.append(E_next_acts)
    # all_A_acts.append(A_acts)

act_avg = torch.stack(all_acts).mean(0)
E_avg = act_avg[0]
# E_next_avg = torch.stack(all_E_next_acts).mean(0)
# A_avg = torch.stack(all_A_acts).mean(0)

# %%
# project avgs into binding subspace and add to deltas
binding_basis = ablated_V[:, :2]

# E_0_vec = E_avg @ binding_basis @ binding_basis.T
# E_vectors = E_deltas + E_0_vec[:, None, :]

E_vectors = E_deltas + E_avg
E_vectors = E_vectors @ binding_basis @ binding_basis.T

# plot E_vectors in 3D
project_and_plot(E_deltas.float(), binding_basis.float())
project_and_plot(E_vectors.float(), binding_basis.float())

# %%
def binding_id_scatter(id_vectors, other_points, basis):
    n_ids = id_vectors.shape[0]
    n_categories = other_points.shape[1]
    
    # Flatten and project ID vectors
    ids_flattened = id_vectors.reshape(n_ids, -1)
    ids_projections = ids_flattened @ basis[:, :2]
    ids_proj_np = ids_projections.cpu().numpy()

    # Create DataFrame for ID vectors
    df_ids = pd.DataFrame(
        {
            "Index": range(n_ids),
            "PC1": ids_proj_np[:, 0],
            "PC2": ids_proj_np[:, 1],
            "Label": [str(i) for i in range(n_ids)],
            "Type": "ID",
            "Category": [f"ID_{i}" for i in range(n_ids)]
        }
    )

    # Process and create traces for other points first
    category_traces = []
    for cat_idx in range(n_categories):
        # Select points for this category and flatten
        cat_points = other_points[:, cat_idx]
        cat_points_flattened = cat_points.reshape(cat_points.shape[0], -1)
        
        # Project points
        cat_projections = cat_points_flattened @ basis[:, :2]
        cat_proj_np = cat_projections.cpu().numpy()

        # Create trace for this category
        cat_trace = go.Scatter(
            x=cat_proj_np[:, 0],
            y=cat_proj_np[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                opacity=1.0,
                color=px.colors.qualitative.Set3[cat_idx % len(px.colors.qualitative.Set3)]
            ),
            name=f'E_{cat_idx}',
            legendgroup='categories',
            legendgrouptitle_text='Activations projected onto binding subspace',
        )
        category_traces.append(cat_trace)

    # Create separate traces for each ID with their respective colors
    id_traces = []
    colors = px.colors.sample_colorscale('rainbow_r', n_ids)
    
    for i in range(n_ids):
        id_trace = go.Scatter(
            x=[df_ids['PC1'][i]],
            y=[df_ids['PC2'][i]],
            mode='markers',
            marker=dict(
                color=colors[i],
                symbol='triangle-down',
                size=10,
            ),
            name=f'Extracted Entity ID Vector {i}',
            legendgroup='ids',
            legendgrouptitle_text='Extracted ID Vectors',
            showlegend=True  # Show each ID in legend
        )
        id_traces.append(id_trace)

    # Create figure and add all traces in the desired order
    fig = go.Figure()
    
    # Add category traces first (they'll be on bottom)
    for trace in category_traces:
        fig.add_trace(trace)
    
    # Add ID traces last (they'll be on top)
    for trace in id_traces:
        fig.add_trace(trace)

    # Update layout
    fig.update_layout(
        title="Extracted ID Vectors and Example Activations Projected onto Binding ID Subspace",
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title=None,
            groupclick="toggleitem",
            itemsizing='constant',
            itemwidth=30,
            entrywidth=70,
            font=dict(size=10)
        ),
        showlegend=True,
        width=900,  # Slightly wider to accommodate legend
        height=730,  # Taller to fit all legend items
        margin=dict(r=200)  # More right margin for legend
    )

    return fig

# %%
example_acts = []
positions = [E_0_POS + i * N_LENGTH for i in range(13)]
for example in tqdm(capitals_examples[:100]):
    # get activations for E_0
    acts = get_pos_acts(example.context, positions)
    example_acts.append(acts)

example_acts = torch.stack(example_acts)
example_acts.shape
# %%
fig = binding_id_scatter(E_vectors.float(), example_acts.float(), binding_basis)
fig.show()


# %%
fig.write_image("plots/gemma-2-2b_binding_id_scatter.png", scale=4)

# %%
def compute_2d_distances(query_activation, id_vectors, basis):
    """
    Compute Euclidean distances between a query activation and ID vectors in the 2D projected space.
    
    Args:
        query_activation: tensor of shape (n_layers, d_model) or (batch, n_layers, d_model)
        id_vectors: tensor of shape (n_ids, n_layers, d_model) - the ID vector centroids
        basis: tensor containing the projection basis vectors
    
    Returns:
        distances: tensor of shape (n_ids,) containing the Euclidean distance 
                  from the query point to each ID vector in the 2D space
    """
    # Convert to float
    query_activation = query_activation.float()
    id_vectors = id_vectors.float()
    basis = basis.float()
    
    # Handle different input shapes
    if query_activation.dim() == 2:  # (n_layers, d_model)
        query_activation = query_activation.unsqueeze(0)  # Add batch dim
    
    # Reshape query to (batch_size or 1, -1)
    batch_size = query_activation.shape[0]
    query_flattened = query_activation.flatten(start_dim=1)
    query_2d = query_flattened @ basis[:, :2]  # Shape: (batch_size, 2)
    
    # Flatten and project ID vectors
    ids_flattened = id_vectors.flatten(start_dim=1)  # Shape: (n_ids, -1)
    ids_2d = ids_flattened @ basis[:, :2]  # Shape: (n_ids, 2)
    
    # Compute distances using broadcasting
    # Reshape to allow broadcasting: (batch_size, n_ids, 2)
    query_2d = query_2d.unsqueeze(1)  # Shape: (batch_size, 1, 2)
    ids_2d = ids_2d.unsqueeze(0)      # Shape: (1, n_ids, 2)
    
    # Compute Euclidean distances in 2D space
    distances = torch.sqrt(torch.sum((ids_2d - query_2d) ** 2, dim=2))  # Shape: (batch_size, n_ids)
    
    # If input was single example, squeeze out batch dimension
    if distances.shape[0] == 1:
        distances = distances.squeeze(0)
        
    return distances

# %%
def distance2similarity(distances, alpha=0.1):
    return torch.nn.functional.relu(alpha - distances) / alpha

for i in range(10):
    try_act = example_acts[i, 1]
    dists = compute_2d_distances(try_act, E_vectors, binding_basis)
    sims = distance2similarity(dists, alpha=0.1)
    print(sims)
# %%
def get_all_position_acts(tokens):
    _, cache = model.run_with_cache(tokens)
    all_acts = [[] for _ in range(tokens.shape[-1])]
    for hook in hooks_of_interest:
        acts = cache[hook].squeeze()
        acts /= acts.norm(2, -1, keepdim=True)
        for i in range(tokens.shape[-1]):
            all_acts[i].append(acts[i])
    all_acts = torch.stack([torch.stack(acts) for acts in all_acts])
    return all_acts

# %%
# test_context = capitals_examples[0].context
test_context = """\
A Day at the Park

Emma has blonde hair that shines in the sunlight. She was waiting at the park for her friends. Thomas is a tall boy with curly hair. He always brings his blue backpack to the playground.

Maria has red sneakers that she wears every day. When she arrived at the park, she saw that Thomas was sitting on the rusty swing. The old oak tree has thick branches that stretch over the playground. 

The three friends played until the sky was orange and the air was cool."""

tokenized = model.tokenizer.encode(test_context, return_tensors="pt")
all_acts = get_all_position_acts(tokenized)  # ctx_len, n_layers, d_model
all_acts.shape
distances = compute_2d_distances(all_acts, E_vectors, binding_basis)
sims = distance2similarity(distances, alpha=0.1)

for i, token in enumerate(tokenized[0]):
    sim = sims[i]
    entity_hit = sim.sum() > 0
    entity_id = torch.argmax(sim).item() if entity_hit else -1
    entity_str = f"E_{entity_id} " if entity_hit else "    "
    print(f"{entity_str} {repr(model.tokenizer.decode(token))}")
# %%
