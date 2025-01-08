# %%
%load_ext autoreload
%autoreload 2
import torch
from transformer_lens import HookedTransformer, ActivationCache
from tasks.capitals import CAPITAL_MAP, NAMES, capitals_generator
from functools import partial

# %%
model = HookedTransformer.from_pretrained(
    "pythia-2.8B",
    device="mps",
    dtype=torch.bfloat16
)

# %%
my_capitals_generator = capitals_generator()
capitals_examples = list(my_capitals_generator)

# %%
test_example = capitals_examples[1]

target_context = test_example.context
source_context = test_example.context_p

target_context_ids = model.tokenizer.encode(target_context, return_tensors="pt")
source_context_ids = model.tokenizer.encode(source_context, return_tensors="pt")

for pos, (s_id, t_id) in enumerate(zip(target_context_ids.squeeze(), source_context_ids.squeeze())):
    s_token = model.tokenizer.decode(s_id.item())
    t_token = model.tokenizer.decode(t_id.item())
    print(f"{pos}: {repr(s_token)}\t{repr(t_token)}")

assert target_context_ids.shape[-1] == source_context_ids.shape[-1]

# %%
E_0_POS = 18
E_1_POS = 27
A_0_POS = 25
A_1_POS = 34

CONTEXT_LENGTH = target_context_ids.shape[-1]

# %%
_, target_cache = model.run_with_cache(
    target_context_ids,
)

_, source_cache = model.run_with_cache(
    source_context_ids,
)

# %%
hooks_of_interest = {}
for hook in model.hook_dict.keys():
    if "block" in hook:
        if "resid" in hook:
            hooks_of_interest[hook] = source_cache[hook]
        else:
            continue
    # else:
    #     if source_cache[hook].shape[:2] == torch.Size([1, 16]):
    #         hooks_of_interest[hook] = source_cache[hook]

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
queries = [test_example.query_E_0, test_example.query_E_1, test_example.query_E_0p, test_example.query_E_1p]
answers = [test_example.answer_0, test_example.answer_1, test_example.answer_0p, test_example.answer_1p]

query_token_ids = torch.stack([model.tokenizer.encode(query, return_tensors="pt", add_special_tokens=False).squeeze() for query in queries])
answer_token_ids = torch.stack([model.tokenizer.encode(f" {answer}", return_tensors="pt", add_special_tokens=False).squeeze() for answer in answers])

# print(model.tokenizer.decode(target_context_ids.squeeze()))
# print(model.tokenizer.decode(query_token_ids[0].squeeze()))
# print(model.tokenizer.decode(answer_token_ids[0].squeeze()))

# %%
target_mask_ids = torch.ones_like(target_context_ids).squeeze() * 5
full_query_ids = torch.cat([target_mask_ids[None, :].expand(4, -1), query_token_ids], dim=1)

# %%
corrupt_logits = patch_all_acts_at_positions(full_query_ids, source_cache, target_cache, [E_0_POS, A_0_POS])

print(target_context)
for query_idx in range(4):
    answer_probs = corrupt_logits[query_idx, -1] - 15
    print(queries[query_idx])
    for token_id in answer_token_ids:
        print(f"{repr(model.tokenizer.decode(token_id))}: {answer_probs[token_id]:.3f}")

# %%
answer_probs = corrupt_logits[:, -1, answer_token_ids] - 15
answer_probs
# answer_probs[:, answer_token_ids]
# answer_probs.shape


# %%
print(test_example.E_0, test_example.E_1, test_example.E_0p, test_example.E_1p)

# %%
answer_probs = corrupt_logits[0, -1].softmax(0) #[answer_token_ids]
for token_id in answer_token_ids:
    print(f"{repr(model.tokenizer.decode(token_id))}: {answer_probs[token_id]:.3f}")


# %%
from functools import partial

expected_logits = model(source_ids)

# %%

_, target_cache = model.run_with_cache(
    target_ids,
    prepend_bos=False
)

corrupt_logits = patch_all_acts_at_pos(long_ids, source_cache, target_cache, 14)

weird_ids = long_ids.clone()
# set all of the context tokens to something weird so we can make sure
# we're patching in all of the activations correctly
weird_ids[0, :CONTEXT_LENGTH] = 3

weird_logits = patch_all_acts_at_pos(weird_ids, source_cache, target_cache, 14)

# This should be True if we're patching the context activations correctly;
# only the post-context tokens should have an affect on logits.
torch.allclose(corrupt_logits, weird_logits)

# %%
madrid_id = model.tokenizer.encode(" Madrid", return_tensors="pt").squeeze().item()
london_id = model.tokenizer.encode(" London", return_tensors="pt").squeeze().item()


# %%
for pos in range(source_ids.shape[-1]):
    corrupt_logits = patch_all_acts_at_pos(long_ids, source_cache, target_cache, pos)
    next_logits = corrupt_logits[0, -1]
    next_probs = next_logits.softmax(0)
    next_token_id = next_logits.argmax()
    print(f"Position {pos}: {model.tokenizer.decode(next_token_id)}")
    print(f"\tMadrid prob: {next_probs[madrid_id]:.3f}")
    print(f"\tLondon prob: {next_probs[london_id]:.3f}")


# %%
