from transformer_lens import HookedTransformer
import torch
import json
from tqdm import tqdm
import random

# 初始化模型（小模型以便运行更快）
model = HookedTransformer.from_pretrained("gpt2", device='mps')

def get_logit_diff(logits, correct_str=" True", incorrect_str=" False"):
    correct_token = model.to_single_token(correct_str)
    incorrect_token = model.to_single_token(incorrect_str)
    probs = torch.softmax(logits[0, -1], dim=-1)
    return probs[correct_token] - probs[incorrect_token]


def get_position(str_tokens: list, year: int):
    for i, token in enumerate(str_tokens):
        if str(year) in token:
            return (i,), (token, )
        elif ''.join([token, str_tokens[i+1]]).replace(' ', '') == str(year):
            return (i, i+1), (token, str_tokens[i+1])
    return None

def activation_patch(
    A: str,
    B: str,
    year_A: str,
    year_B: str,
    layer: int,
    label_A: bool,
    label_B: bool,
    token_pos = (3, ),
    correct_str=" True",
    incorrect_str=" False"
):
    tokens_A = model.to_tokens(A)
    logits_A = model(tokens_A)
    if label_A:
        log_diff_A = get_logit_diff(logits_A, correct_str, incorrect_str)
    else:
        log_diff_A = get_logit_diff(logits_A, incorrect_str, correct_str)
    if log_diff_A <= 0:
        # print("Activation of A may not contain correct information", log_diff_A)
        return None
    tokens_B = model.to_tokens(B)
    logits_clean = model(tokens_B)
    # if label_B:
    #     log_diff_B = get_logit_diff(logits_clean, correct_str, incorrect_str)
    # else:
    #     log_diff_B = get_logit_diff(logits_clean, incorrect_str, correct_str)
    # if log_diff_B <= 0:
    #     # print("Activation of B may not contain correct information", log_diff_B)
    #     return None
    
    # try:
    #     positions_A, year_tokens_A = get_position(model.to_str_tokens(tokens_A), year_A)
    #     positions_B, year_tokens_B = get_position(model.to_str_tokens(tokens_B), year_B)
    #     print(positions_A, year_tokens_A, positions_B, year_tokens_B)
    #     # Make sure alignment is the same
    #     assert positions_A == positions_B, (
    #     f"Token mismatch:\nA tokens: {model.to_str_tokens(model.to_tokens(A))}\n"
    #     f"B tokens: {model.to_str_tokens(model.to_tokens(B))}\n"
    #     f"A year tokens: {positions_A}, {year_tokens_A}, B year tokens: {positions_B}, {year_tokens_B}"
    #     )
    # except Exception as e:
    #     print(e)
    #     return None
    
    _, cache_A = model.run_with_cache(tokens_A)
    def patch_hook(act, hook):
        for pos in token_pos:
            act[:, pos, :] = cache_A[hook.name][:, pos, :]
        return act

    logits_patched = model.run_with_hooks(
        tokens_B,
        fwd_hooks=[
            (f'blocks.{layer}.hook_resid_post', patch_hook)
        ]
    )


    if label_A:
        log_diff_clean = get_logit_diff(logits_clean, correct_str, incorrect_str)
        log_diff_patched = get_logit_diff(logits_patched, correct_str, incorrect_str)
    else:
        log_diff_clean = get_logit_diff(logits_clean, incorrect_str, correct_str)
        log_diff_patched = get_logit_diff(logits_patched, incorrect_str, correct_str)

    delta = log_diff_patched - log_diff_clean

    # print(f"[Layer {layer}] Logit diff clean: {log_diff_clean:.4f} | patched: {log_diff_patched:.4f} | Δ: {delta:.4f}")
    return log_diff_clean.item(), log_diff_patched.item(), delta.item()



# def get_data_loader(data):
#     for token_len, years in data.items():
#         for year_A, digits in years.items():
#             for digit, replacements in digits.items():
#                 for year_B in replacements.keys():
#                     yield token_len, int(year_A), int(digit), int(year_B)





def get_prompt(year_A, year_B, min_year=1000, max_year=2077):
    
    def make_prompt(year_A, year_B, year_C, correct_A, correct_B, relation="earlier"):

        prompt_A = f"The year {year_A} is {relation} than the year {year_C}. True or False?\nA:"
        prompt_B = f"The year {year_B} is {relation} than the year {year_C}. True or False?\nA:"
        return (prompt_A, prompt_B, correct_A, correct_B)
    
    prompts = []
    if year_A < year_B:
        # A B C
        # prompts.append(make_prompt(year_A, year_B, random.choice(range(year_B, max_year)), True, True))
        # A C B
        prompts.append(make_prompt(year_A, year_B, random.choice(range(year_A, year_B)), True, False))
        # C A B
        # prompts.append(make_prompt(year_A, year_B, random.choice(range(min_year, year_A)), False, False))
    else:
        # B A C
        # prompts.append(make_prompt(year_A, year_B, random.choice(range(year_A, max_year)), True, True))
        # B C A
        prompts.append(make_prompt(year_A, year_B, random.choice(range(year_B, year_A)), False, True))
        # C B A
        # prompts.append(make_prompt(year_A, year_B, random.choice(range(min_year, year_B)), False, False))
    return prompts[0]
    reverse_prompts = [
        (
            prompt[0].replace("earlier", "later"),
            prompt[1].replace("earlier", "later"),
            not prompt[2],
            not prompt[3]
        )
        for prompt in prompts
    ]
    prompts += reverse_prompts
    
    return random.choice(prompts)
    return prompts + reverse_prompts


# def get_all_prompts(year_A, year_B, min=1000, max=2077):
#     prompts = []
#     if year_A < year_B:
#         # A B C
#         year_C = random.choice(range(year_B, max))
#         A = f"The year {year_A} is earlier than the year {year_C}. True or False?\nA:"
#         B = f"The year {year_B} is earlier than the year {year_C}. True or False?\nA:"
#         label_A = True
#         label_B = True
#         prompts.append((A, B, label_A, label_B))
#         # A C B
#         year_C = random.choice(range(year_A, year_B))
#         A = f"The year {year_A} is earlier than the year {year_C}. True or False?\nA:"
#         B = f"The year {year_B} is earlier than the year {year_C}. True or False?\nA:"
#         label_A = True
#         label_B = False
#         prompts.append((A, B, label_A, label_B))
#         # C A B
#         year_C = random.choice(range(min, year_A))
#         A = f"The year {year_A} is earlier than the year {year_C}. True or False?\nA:"
#         B = f"The year {year_B} is earlier than the year {year_C}. True or False?\nA:"
#         label_A = False
#         label_B = False
#         prompts.append((A, B, label_A, label_B))
#     else:
#         # B A C
#         year_C = random.choice(range(year_A, max))
#         A = f"The year {year_A} is earlier than the year {year_C}. True or False?\nA:"
#         B = f"The year {year_B} is earlier than the year {year_C}. True or False?\nA:"
#         label_A = True
#         label_B = True
#         prompts.append((A, B, label_A, label_B))
#         # B C A
#         year_C = random.choice(range(year_B, year_A))
#         A = f"The year {year_A} is earlier than the year {year_C}. True or False?\nA:"
#         B = f"The year {year_B} is earlier than the year {year_C}. True or False?\nA:"
#         label_A = False
#         label_B = True
#         prompts.append((A, B, label_A, label_B))
#         # C B A 
#         year_C = random.choice(range(min, year_B))
#         A = f"The year {year_A} is earlier than the year {year_C}. True or False?\nA:"
#         B = f"The year {year_B} is earlier than the year {year_C}. True or False?\nA:"
#         label_A = False
#         label_B = False
#         prompts.append((A, B, label_A, label_B))
#     reverse_prompts = []
#     for prompt in prompts:
#         reverse_prompts.append((prompt[0].replace('earlier', 'later'), prompt[1].replace('earlier', 'later'), not prompt[2], not prompt[3]))
#     prompts += reverse_prompts
#     return prompts


if __name__ == '__main__':
    with open('dataset_1.json', 'r') as f:
        data_loader = json.load(f)
    results = {i: {} for i in range(12)}
    min, max = 1950, 2050
    for i, item in enumerate(tqdm(data_loader)):
        token_len, year_A, digit, year_B = item
        # if year_A <= min or year_A >= max or year_B <= min or year_B >= max: continue
        prompt = get_prompt(year_A, year_B, min, max)
        A, B, label_A, label_B = prompt
        for layer in range(12):
            ret = activation_patch(
                            A,
                            B,
                            year_A=str(year_A),
                            year_B=str(year_B),
                            layer=layer,
                            label_A=label_A,
                            label_B=label_B)
            if ret is not None:
                log_diff_clean, log_diff_patched, delta = ret
                if token_len not in results[layer]:
                    results[layer][token_len] = {}
                if digit not in results[layer][token_len]:
                    results[layer][token_len][digit] = {
                        'log_diff_clean': [],
                        'log_diff_patched': [],
                        'delta': []
                    }
                results[layer][token_len][digit]['log_diff_clean'].append(log_diff_clean)
                results[layer][token_len][digit]['log_diff_patched'].append(log_diff_patched)
                results[layer][token_len][digit]['delta'].append(delta)
    with open('results_1.json', 'w') as f:
        json.dump(results, f, indent=4)

# demo
# A = "Q: The year 2023 is earlier than the year 2025. True or False?\nA:"
# B = "Q: The year 2026 is earlier than the year 2025. True or False?\nA:"
# for i in range(12):
#     _ = activation_patch(
#         A,
#         B,
#         year_A="2023",
#         year_B="2026",
#         layer=i,
#         label_A=True,
#         label_B=False)