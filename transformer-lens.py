# %%
import torch
from transformer_lens import HookedTransformer

# %%
model = HookedTransformer.from_pretrained("gpt2")

# %%
source_context = "<|endoftext|>The capital of France is the city of Paris\nThe capital of England is"
question = " the city of"

source_ids = model.tokenizer.encode(source_context, return_tensors="pt")

CONTEXT_LENGTH = source_ids.shape[-1]

print(source_ids.shape)
source_ids

# %%
for pos, input_id in enumerate(source_ids.squeeze()):
    token = model.tokenizer.decode(input_id.item())
    print(f"{pos}: {repr(token)}")
# %%
_, source_cache = model.run_with_cache(
    source_ids,
    prepend_bos=False
)

hooks_of_interest = {}
for hook in source_cache.keys():
    if "block" in hook:
        if "resid" in hook:
            hooks_of_interest[hook] = source_cache[hook]
        else:
            continue
    else:
        if source_cache[hook].shape[:2] == torch.Size([1, 16]):
            hooks_of_interest[hook] = source_cache[hook]

for hook, value in hooks_of_interest.items():
    print(f"{value.shape}\t{hook}")


# %%
torch.equal(source_cache['blocks.0.hook_resid_post'], source_cache['blocks.1.hook_resid_pre'])

# %%
target_ids = source_ids.clone()

target_ids[..., 14] = model.tokenizer.encode(" Spain", return_tensors="pt").squeeze()

model.tokenizer.decode(target_ids.squeeze())

# %%
question_ids = model.tokenizer.encode(question, return_tensors="pt")

long_ids = torch.cat([target_ids, question_ids], dim=1)

print(model.tokenizer.decode(long_ids.squeeze()))

# %%
from functools import partial

expected_logits = model(source_ids)

# %%

_, target_cache = model.run_with_cache(
    target_ids,
    prepend_bos=False
)

# we run a forward pass on the query sentence, patching in _all_ of the activations
# from the target context. We also patch in activations from the source context only
# at pos.
def patch_all_acts_at_pos(target_ids, source_cache, target_cache, pos):
    def position_patch_hook(activation, hook, hook_name, pos):

        source_acts = source_cache[hook_name]
        target_acts = target_cache[hook_name]
        activation[:, :pos, ...] = target_acts[:, :pos, ...]
        activation[:, pos, ...] = source_acts[:, pos, ...]
        activation[:, pos+1:CONTEXT_LENGTH, ...] = target_acts[:, pos+1:CONTEXT_LENGTH, ...]
        return activation
    
    corrupt_logits = model.run_with_hooks(
        target_ids,
        fwd_hooks=[
            (hook_name, partial(position_patch_hook, hook_name=hook_name, pos=pos)) for hook_name in hooks_of_interest
        ]
    )

    return corrupt_logits

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
