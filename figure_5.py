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
    # "pythia-6.9B",
    # "meta-llama/Llama-3.2-3B",
    "gemma-2-27b",
    device=device,
    dtype=torch.bfloat16
)

# %%
hooks_of_interest = []
for hook in model.hook_dict.keys():
    if "hook_resid_pre" in hook:
        hooks_of_interest.append(hook)

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
def get_activation_difference(context: str):
    tokenized = model.tokenizer.encode(context, return_tensors='pt')
    tokenized = tokenized.to(device)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokenized)
    E_0_acts = []
    E_0_next_acts = []
    E_1_acts = []
    E_1_next_acts = []
    A_0_acts = []
    A_1_acts = []

    for hook in hooks_of_interest:
        acts = cache[hook].squeeze()
        E_0_acts.append(acts[E_0_POS])
        E_0_next_acts.append(acts[E_0_POS + 1])
        E_1_acts.append(acts[E_1_POS])
        E_1_next_acts.append(acts[E_1_POS + 1])
        A_0_acts.append(acts[A_0_POS])
        A_1_acts.append(acts[A_1_POS])

    E_0_acts = torch.stack(E_0_acts)
    E_0_next_acts = torch.stack(E_0_next_acts)
    E_1_acts = torch.stack(E_1_acts)
    E_1_next_acts = torch.stack(E_1_next_acts)
    A_0_acts = torch.stack(A_0_acts)
    A_1_acts = torch.stack(A_1_acts)

    E_diff = E_1_acts - E_0_acts
    E_next_diff = E_1_next_acts - E_0_next_acts
    A_diff = A_1_acts - A_0_acts

    # C_A_0 + A_diff => A_0 bound to E_1
    # C_A_1 - A_diff => A_1 bound to E_0
    return E_diff, E_next_diff, A_diff

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

mean_E_diff = torch.stack(E_diffs).mean(dim=0)
mean_E_next_diff = torch.stack(E_next_diffs).mean(dim=0)
mean_A_diff = torch.stack(A_diffs).mean(dim=0)


# %%
E_diff_norms = torch.norm(mean_E_diff, dim=-1)
E_next_diff_norms = torch.norm(mean_E_next_diff, dim=-1)
A_diff_norms = torch.norm(mean_A_diff, dim=-1)

# %%
r_E = torch.randn(len(E_diff_norms), mean_E_diff.shape[-1], device=device, dtype=E_diff_norms.dtype)
r_E_next = torch.randn(len(E_next_diff_norms), mean_E_diff.shape[-1], device=device, dtype=E_diff_norms.dtype)
r_A = torch.randn(len(A_diff_norms), mean_E_diff.shape[-1], device=device, dtype=E_diff_norms.dtype)

r_E /= torch.norm(r_E, dim=1, keepdim=True)
r_E_next /= torch.norm(r_E_next, dim=1, keepdim=True)
r_A /= torch.norm(r_A, dim=1, keepdim=True)

r_E *= E_diff_norms.unsqueeze(1)
r_E_next *= E_next_diff_norms.unsqueeze(1)
r_A *= A_diff_norms.unsqueeze(1)

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

print(f"Control:\t{control_0 / n_examples:.2f}, {control_1 / n_examples:.2f}")
print(f"Entity:\t{entity_0 / n_examples:.2f}, {entity_1 / n_examples:.2f}")
print(f"Attribute:\t{attribute_0 / n_examples:.2f}, {attribute_1 / n_examples:.2f}")
print(f"Both:\t{both_0 / n_examples:.2f}, {both_1 / n_examples:.2f}")

# %%
