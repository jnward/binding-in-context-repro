# %%
%load_ext autoreload
%autoreload 2
import torch
from transformer_lens import HookedTransformer, ActivationCache
from tasks.capitals import CAPITAL_MAP, NAMES, capitals_generator
from functools import partial
from tqdm import tqdm

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

# %%
def permute_acts(target_ids: torch.Tensor, target_cache: ActivationCache, i: int, j: int):
    # move dim i to j
    def position_patch_hook(activation: torch.Tensor, hook, hook_name: str, i: int, j: int):
        target_acts = target_cache[hook_name]
        # if hook_name.split(".")[1] == "0":
            # activation[:, :CONTEXT_LENGTH] = move_elements(target_acts, i, j, dim=1)
        activation[:, :CONTEXT_LENGTH] = move_elements(target_acts, i, j, dim=1)
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


