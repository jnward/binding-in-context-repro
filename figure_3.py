# %%
%load_ext autoreload
%autoreload 2
import torch
from transformer_lens import HookedTransformer, ActivationCache
from tasks.capitals import CAPITAL_MAP, NAMES, capitals_generator
from functools import partial
from tqdm import tqdm

device = "mps"

# %%
model = HookedTransformer.from_pretrained_no_processing(
    "pythia-410M",
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

E_0_POS = 18
E_1_POS = 27
A_0_POS = 25
A_1_POS = 34

CONTEXT_LENGTH = 36  #target_context_ids.shape[-1]

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
    next(my_capitals_generator) for _ in range(16)
]

# %%
my_example = capitals_examples[1]

def get_logit_matrices(test_example, patch_positions):
    target_context = test_example.context
    source_context = test_example.context_p

    target_context_ids = model.tokenizer.encode(target_context, return_tensors="pt")
    source_context_ids = model.tokenizer.encode(source_context, return_tensors="pt")

    assert target_context_ids.shape[-1] == source_context_ids.shape[-1]

    _, target_cache = model.run_with_cache(target_context_ids)
    _, source_cache = model.run_with_cache(source_context_ids)

    queries = [test_example.query_E_0, test_example.query_E_1, test_example.query_E_0p, test_example.query_E_1p]
    answers = [test_example.answer_0, test_example.answer_1, test_example.answer_0p, test_example.answer_1p]

    query_token_ids = torch.stack([model.tokenizer.encode(query, return_tensors="pt", add_special_tokens=False).squeeze() for query in queries])
    answer_token_ids = torch.stack([model.tokenizer.encode(f" {answer}", return_tensors="pt", add_special_tokens=False).squeeze() for answer in answers])

    target_mask_ids = torch.ones_like(target_context_ids).squeeze() * 5
    full_query_ids = torch.cat([target_mask_ids[None, :].expand(4, -1), query_token_ids], dim=1)

    corrupt_logits = patch_all_acts_at_positions(full_query_ids, source_cache, target_cache, patch_positions)
    del source_cache, target_cache

    answer_probs = corrupt_logits[:, -1, answer_token_ids]
    return answer_probs

# %%
import gc
def cleanup():
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    
# %%
import plotly.express as px

clean_logits = []
a0_logits = []
e0_logits = []
a0e0_logits = []

for example in tqdm(capitals_examples):
    logit_matrix = get_logit_matrices(example, [])
    clean_logits.append(logit_matrix)
    cleanup()

clean_avg = torch.stack(clean_logits).mean(0)
px.imshow(clean_avg.detach().float().cpu().numpy(), title="Clean").show()

for example in tqdm(capitals_examples):
    logit_matrix = get_logit_matrices(example, [A_0_POS])
    a0_logits.append(logit_matrix)
    cleanup()

a0_avg = torch.stack(a0_logits).mean(0)
px.imshow(a0_avg.detach().float().cpu().numpy(), title="A_0").show()

for example in tqdm(capitals_examples):
    logit_matrix = get_logit_matrices(example, [E_0_POS])
    e0_logits.append(logit_matrix)
    cleanup()

e0_avg = torch.stack(e0_logits).mean(0)
px.imshow(e0_avg.detach().float().cpu().numpy(), title="E_0").show()

for example in tqdm(capitals_examples):
    logit_matrix = get_logit_matrices(example, [A_0_POS, E_0_POS])
    a0e0_logits.append(logit_matrix)
    cleanup()

a0e0_avg = torch.stack(a0e0_logits).mean(0)
px.imshow(a0e0_avg.detach().float().cpu().numpy(), title="A_0, E_0").show()

# %%
