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

# N is the number of binding examples in each context
N = 2

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
capitals_examples = [next(my_capitals_generator) for _ in range(16)]

print(capitals_examples[0].context)
print(capitals_examples[0].query_E_0)
print(capitals_examples[0].answers)

tokenized = model.tokenizer.encode(capitals_examples[0].context, return_tensors="pt")

for i, token in enumerate(tokenized[0]):
    print(i, repr(model.tokenizer.decode(token)))


# %%
answer_logits = []

for example in tqdm(capitals_examples[:16]):
    full_context = example.context + example.query_E_0
    tokenized = model.tokenizer.encode(full_context, return_tensors="pt").to(device)
    logits = model(tokenized)
    answer_token_ids = [
        model.tokenizer.encode(
            f" {a}", return_tensors="pt", add_special_tokens=False
        ).item()
        for a in example.answers
    ]
    answer_logits.append(logits[0, -1, answer_token_ids])

answer_logits = torch.stack(answer_logits)
print(answer_logits.mean(0))


# %%
answer_logits = []

for example in tqdm(capitals_examples):
    # full_context = (
    #     example.context
    #     + f"\nQuestion: Which city is the capital of {example.A_0}?\nAnswer: The capital of {example.A_0} is"
    # )
    # tokenized = model.tokenizer.encode(full_context, return_tensors="pt").to(device)
    tokenized, answer_token_ids = get_no_relation_tokens(example)
    logits = model(tokenized)
    # answer_token_ids = [
    #     model.tokenizer.encode(
    #         f" {a}", return_tensors="pt", add_special_tokens=False
    #     ).item()
    #     for a in example.answers
    # ]
    answer_logits.append(logits[0, -1, answer_token_ids])

answer_logits = torch.stack(answer_logits)
print(answer_logits.mean(0))

# %%
# calculate mean activation for each attention head

head_hooks = []
for hook in model.hook_dict.keys():
    if "attn_out" in hook:
        head_hooks.append(hook)

# %%
d_head = model.cfg.d_head


def ablate_head(target_ids: torch.Tensor, block_no, head_idx):
    def head_ablation_hook(activation: torch.Tensor, hook, head_idx):
        activation[:, :, head_idx * d_head : (head_idx + 1) * d_head] = 0.0
        return activation

    hook_name = head_hooks[block_no]
    logits = model.run_with_hooks(
        target_ids,
        fwd_hooks=[(hook_name, partial(head_ablation_hook, head_idx=head_idx))],
    )
    return logits


# %%
def get_no_relation_tokens(example):
    full_context = (
        example.context
        + f"\nQuestion: Which city is the capital of {example.A_0}?\nAnswer: The capital of {example.A_0} is"
    )
    token_ids = model.tokenizer.encode(full_context, return_tensors="pt").to(device)
    answer_token_ids = [
        model.tokenizer.encode(
            f" {a}", return_tensors="pt", add_special_tokens=False
        ).item()
        for a in example.answers
    ]
    return token_ids, answer_token_ids


def get_relation_tokens(example):
    full_context = example.context + example.query_E_0
    token_ids = model.tokenizer.encode(full_context, return_tensors="pt").to(device)
    answer_token_ids = [
        model.tokenizer.encode(
            f" {a}", return_tensors="pt", add_special_tokens=False
        ).item()
        for a in example.answers
    ]
    return token_ids, answer_token_ids


# %%
# baseline
answer_logits = []

for example in tqdm(capitals_examples):
    tokenized, answer_token_ids = get_no_relation_tokens(example)
    logits = model(tokenized)
    answer_logits.append(logits[0, -1, answer_token_ids])

answer_logits = torch.stack(answer_logits)
print(answer_logits.mean(0))

# %%
mean_logits = torch.zeros(model.cfg.n_layers, model.cfg.n_heads, N)

for block_no in range(model.cfg.n_layers):
    for head_idx in range(model.cfg.n_heads):
        print(f"Block {block_no}, Head {head_idx}", end=": ")
        answer_logits = []
        for example in capitals_examples:
            query_tokens, answer_tokens = get_relation_tokens(example)
            logits = ablate_head(query_tokens, block_no, head_idx)
            answer_logits.append(logits[0, -1, answer_tokens])
        mean_logits[block_no, head_idx] = torch.stack(answer_logits).mean(0)
        print(mean_logits[block_no, head_idx])

# %%
mean_logits.min()
# %%
import plotly.express as px
diffs = mean_logits[..., 0] - mean_logits[..., 1]
fig = px.imshow(mean_logits[..., 0].cpu().numpy())
fig.show()
# %%
mean_logits[..., 0].min()
# %%
