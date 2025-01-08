# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel

# %%
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

assert isinstance(model, GPT2LMHeadModel)

# %%
target_context = "<|endoftext|>The capital of France is the city of Paris\nThe capital of England is"
question = " the city of"

target_ids = tokenizer.encode(target_context, return_tensors="pt")

for pos, input_id in enumerate(target_ids.squeeze()):
    token = tokenizer.decode(input_id.item())
    print(f"{pos}: {repr(token)}")
# %%
target_out = model.forward(target_ids)

target_logits = target_out.logits
print(tokenizer.decode(torch.argmax(target_logits.squeeze()[-1]).item()))

target_kv_cache = (
    target_out.past_key_values
)  # n_layers-tuple of (key, value) tensors with shape (batch_size, num_heads, seq_len, head_dim)


# %%
source_context = target_context.replace("England", "Spain")
source_ids = tokenizer.encode(source_context, return_tensors="pt")

source_out = model.forward(source_ids)

source_logits = source_out.logits
print(tokenizer.decode(torch.argmax(source_logits.squeeze()[-1]).item()))

source_kv_cache = source_out.past_key_values
print(len(source_kv_cache[0][0][0][0]))

# %%

n_tokens = len(target_ids.squeeze())


# %%
mask_ids = (
    torch.ones(n_tokens, dtype=torch.int) * 3
).tolist()  # this is just to assert that information from the source/target is not being used except for that which in the KV cache

# mask_ids[-1] = target_ids.squeeze()[-1].item()

question_ids = tokenizer.encode(question, return_tensors="pt").squeeze().tolist()

mask_ids = torch.tensor(mask_ids + question_ids, dtype=torch.int).unsqueeze(0)
mask_ids

# %%
madrid_id = tokenizer.encode(" Madrid", return_tensors="pt").squeeze().item()
london_id = tokenizer.encode(" London", return_tensors="pt").squeeze().item()

print(madrid_id, london_id)

# %%
for ts, kv_cache in zip(["target", "source"], [target_kv_cache, source_kv_cache]):
    corrupt_out = model.generate(
        mask_ids,
        past_key_values=kv_cache,
        use_cache=True,
        max_length=mask_ids.shape[-1] + 1,
        pad_token_id=tokenizer.eos_token_id,
    )

    print(f"Using {ts} activations: {tokenizer.decode(corrupt_out.squeeze())}")
# %%
def clone_cache(kv_cache):
    return tuple((key.clone(), value.clone()) for key, value in kv_cache)

for pos in range(n_tokens):
    corrupted_kv_cache = clone_cache(target_kv_cache)
    for layer in range(len(corrupted_kv_cache)):
        for kv in range(2):
            corrupted_kv_cache[layer][kv][:, :, pos, :] = source_kv_cache[layer][kv][:, :, pos, :]
    corrupt_out = model.generate(
        mask_ids,
        past_key_values=corrupted_kv_cache,
        use_cache=True,
        max_length=mask_ids.shape[-1] + 1,
        pad_token_id=tokenizer.eos_token_id,
        output_logits=True,
        return_dict_in_generate=True,
    )
    print(f"Corrupted target with source position {pos}: {tokenizer.decode(corrupt_out.sequences.squeeze())}")
    print(f"Madrid logit: {corrupt_out.logits[0][0, madrid_id]}")
    print(f"London logit: {corrupt_out.logits[0][0, london_id]}")

    temp_ids = torch.cat([mask_ids, corrupt_out.sequences[:, -1].unsqueeze(0)], dim=1)
    print(temp_ids)

    print(tokenizer.decode(temp_ids.squeeze()))

    forward_out = model(
        temp_ids,
        past_key_values=corrupted_kv_cache,
        use_cache=True,
    )

    print(f"Madrid logit: {forward_out.logits[0, -1, madrid_id]}")
    print(f"London logit: {forward_out.logits[0, -1, london_id]}")

    print(tokenizer.decode(forward_out.logits.squeeze()[-1].argmax()))
    probs = forward_out.logits[0, -1].softmax(0)
    print(f"Madrid prob: {probs[madrid_id]:.3f}")
    print(f"London logit: {probs[london_id]:.3f}")

# %%

corrupted_out = model(
    mask_ids,
    past_key_values=corrupted_kv_cache,
    use_cache=True,
)
print(tokenizer.decode(corrupted_out.logits.squeeze()[-1].argmax()))
# %%
mask_ids.shape
# %%
print(forward_out.logits[0, -1].softmax(0).max())
# %%
tokenizer.decode([3576])
# %%






# test_ids = (
#     torch.ones(n_tokens, dtype=torch.int) * 3
# ).tolist()  # this is just to assert that information from the source/target is not being used except for that which in the KV cache

test_ids = target_ids.squeeze().clone()

# mask_ids[-1] = target_ids.squeeze()[-1].item()

# test_ids[:1] = 3

test_ids = test_ids.tolist()

question_ids = tokenizer.encode(question, return_tensors="pt").squeeze().tolist()


test_ids = torch.tensor(test_ids + question_ids, dtype=torch.int).unsqueeze(0)

# temp_ids = torch.cat([mask_ids, corrupt_out.sequences[:, -1].unsqueeze(0)], dim=1)
# print(temp_ids)

# print(tokenizer.decode(test_ids.squeeze()))

forward_out = model(
    test_ids,
    past_key_values=corrupted_kv_cache,
    use_cache=True,
)

logits = forward_out.logits[0, -1]

print(tokenizer.decode(logits.argmax()))

logits
# %%
