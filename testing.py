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

question_ids = tokenizer.encode(question, return_tensors="pt").squeeze().tolist()

mask_ids = torch.tensor(mask_ids + question_ids, dtype=torch.int).unsqueeze(0)
mask_ids


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
    )
    print(f"Corrupted target with source position {pos}: {tokenizer.decode(corrupt_out.squeeze())}")
# %%
