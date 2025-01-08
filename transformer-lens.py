# %%
import torch
from transformer_lens import HookedTransformer

# %%
model = HookedTransformer.from_pretrained("gpt2")

# %%
source_context = "<|endoftext|>The capital of France is the city of Paris\nThe capital of England is"
question = " the city of"

source_ids = model.tokenizer.encode(source_context, return_tensors="pt")

print(source_ids.shape)
source_ids

# %%
for pos, input_id in enumerate(source_ids.squeeze()):
    token = model.tokenizer.decode(input_id.item())
    print(f"{pos}: {repr(token)}")
# %%
_, cache = model.run_with_cache(
    source_ids,
    prepend_bos=False
)

hooks_of_interest = {}
for hook in cache.keys():
    if "block" in hook:
        if "resid" in hook:
            hooks_of_interest[hook] = cache[hook]
        else:
            continue
    else:
        if cache[hook].shape[:2] == torch.Size([1, 16]):
            hooks_of_interest[hook] = cache[hook]

for hook, value in hooks_of_interest.items():
    print(f"{value.shape}\t{hook}")


# %%
torch.equal(cache['blocks.0.hook_resid_post'], cache['blocks.1.hook_resid_pre'])

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

def patch_all_acts_at_pos(target_ids, source_cache, pos):
    def position_patch_hook(activation, hook, hook_name, pos):
        source_acts = source_cache[hook_name]
        activation[:, pos, ...] = source_acts[:, pos, ...]
        return activation
    
    corrupt_logits = model.run_with_hooks(
        target_ids,
        fwd_hooks=[
            (hook_name, partial(position_patch_hook, hook_name=hook_name, pos=pos)) for hook_name in hooks_of_interest
        ]
    )

    return corrupt_logits

corrupt_logits = patch_all_acts_at_pos(long_ids, cache, 14)
# %%
torch.allclose(corrupt_logits[:, :16], expected_logits)


# %%
madrid_id = model.tokenizer.encode(" Madrid", return_tensors="pt").squeeze().item()
london_id = model.tokenizer.encode(" London", return_tensors="pt").squeeze().item()


# %%
for pos in range(source_ids.shape[-1]):
    corrupt_logits = patch_all_acts_at_pos(long_ids, cache, pos)
    next_logits = corrupt_logits[0, -1]
    next_probs = next_logits.softmax(0)
    next_token_id = next_logits.argmax()
    print(f"Position {pos}: {model.tokenizer.decode(next_token_id)}")
    print(f"\tMadrid prob: {next_probs[madrid_id]:.3f}")
    print(f"\tLondon prob: {next_probs[london_id]:.3f}")


# %%
