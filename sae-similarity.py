# %%
# %load_ext autoreload
# %autoreload 2
import torch
from sae_lens import SAE

device = "cuda"
model_name = "gemma-2-2b"
hook_name = "blocks.12.hook_resid_post"
release = "gemma-scope-2b-pt-res-canonical"

my_sae, _, _ = SAE.from_pretrained(
    release,
    sae_id = "layer_12/width_1m/canonical",
    device=device,
)

# %%
E_vectors = torch.load(f"{model_name}-E-vectors.pt")
E_vectors = E_vectors[:, 0]
E_vectors /= E_vectors.norm(2, -1, keepdim=True)

# %%
dictionary_vectors = my_sae.W_dec

# %%
#compute similarity matrix
similarity_matrix = E_vectors @ dictionary_vectors.T
similarity_matrix.shape
# %%
E_vectors.shape
# %%
similarity_matrix
# %%
max_vals, max_indices = similarity_matrix.max(dim=1)
max_vals
# %%
max_indices

found_features = dictionary_vectors[max_indices]

# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

    # Create a new figure
    fig = go.Figure()

    # Calculate color values for the rainbow scale
    colors = px.colors.sample_colorscale(
        px.colors.sequential.Rainbow, 
        [i/(B-1) for i in range(B)]
    )

    # Add arrows for each point
    for i in range(len(df)):
        fig.add_trace(go.Scatter(
            x=[0, df['PC1'].iloc[i]],
            y=[0, df['PC2'].iloc[i]],
            mode='lines+markers',
            name=f'Vector {i+1}',
            line=dict(color=colors[i], width=2),
            marker=dict(size=[0, 10]),  # Hide origin marker, show endpoint
            hoverinfo='text',
            hovertext=f'Index: {i}<br>PC1: {df["PC1"].iloc[i]:.3f}<br>PC2: {df["PC2"].iloc[i]:.3f}'
        ))

    fig.update_layout(
        title="Data Projected onto First Two Principal Components",
        showlegend=False,
        xaxis_title="PC1",
        yaxis_title="PC2",
        # Make the plot square
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    fig.show()

project_and_plot(E_vectors.float(), V.float())
project_and_plot(found_features.detach().float(), V.float())







# %%
import pandas as pd
import plotly.express as px

_, _, V = E_vectors.svd()

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


project_and_plot(E_vectors.float(), V.float())
project_and_plot(found_features.float(), V.float())
# %%
