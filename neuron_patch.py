from transformer_lens import HookedTransformer
import torch
import json
from tqdm import tqdm
import random
import numpy as np
import os

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


def run_neuron_patching(
    A: str,
    B: str,
    year_A: str,
    year_B: str,
    layer: int,
    label_A: bool,
    label_B: bool,
    token_pos = 3,
    correct_str=" True",
    incorrect_str=" False"
):
    tokens_A = model.to_tokens(A)
    tokens_B = model.to_tokens(B)

    # Check A/B are correctly classified
    logits_A = model(tokens_A)
    logits_B = model(tokens_B)

    if label_A:
        log_diff_A = get_logit_diff(logits_A, correct_str, incorrect_str)
    else:
        log_diff_A = get_logit_diff(logits_A, incorrect_str, correct_str)
    if log_diff_A <= 0:
        return None

    # if label_B:
    #     log_diff_B = get_logit_diff(logits_B, correct_str, incorrect_str)
    # else:
    #     log_diff_B = get_logit_diff(logits_B, incorrect_str, correct_str)
    # if log_diff_B <= 0:
    #     return None

    # Get activations from A
    _, cache_A = model.run_with_cache(tokens_A)
    resid_A = cache_A[f'blocks.{layer}.hook_resid_post']  # [1, seq_len, d_model]
    neuron_A = resid_A[0, token_pos, :]  # [d_model]
    d_model = neuron_A.shape[0]

    # Get clean B prediction for comparison
    logits_clean_B = logits_B
    if label_A:
        log_diff_clean = get_logit_diff(logits_clean_B, correct_str, incorrect_str)
    else:
        log_diff_clean = get_logit_diff(logits_clean_B, incorrect_str, correct_str)

    deltas = []
    
    for i in range(d_model):
        def patch_hook(act, hook):
            # act: [1, seq_len, d_model]
            act[:, token_pos, i] = neuron_A[i]
            return act

        logits_patched = model.run_with_hooks(
            tokens_B,
            fwd_hooks=[(f'blocks.{layer}.hook_resid_post', patch_hook)]
        )

        if label_A:
            log_diff_patched = get_logit_diff(logits_patched, correct_str, incorrect_str)
        else:
            log_diff_patched = get_logit_diff(logits_patched, incorrect_str, correct_str)

        delta = log_diff_patched - log_diff_clean
        deltas.append(delta.item())
    return deltas


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


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run neuron patching experiments.")
    parser.add_argument('--token_len', type=int, default=1)
    parser.add_argument('--digit', type=int, default=0)
    parser.add_argument('--token_pos', type=int, default=3)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    with open(f'dataset_{args.token_len}_{args.digit}.json', 'r') as f:
        data_loader = json.load(f)
    min, max = 1950, 2050
    for i, item in enumerate(tqdm(data_loader)):
        token_len, year_A, digit, year_B = item
        # if year_A <= min or year_A >= max or year_B <= min or year_B >= max: continue
        prompt = get_prompt(year_A, year_B, min, max)
        A, B, label_A, label_B = prompt
        for layer in range(5, 8):
            save_dir = f'neuron_patching_results/layer{layer}/tokenlen{token_len}/digit{digit}'
            os.makedirs(save_dir, exist_ok=True)
            delta = run_neuron_patching(
                            A,
                            B,
                            year_A=str(year_A),
                            year_B=str(year_B),
                            layer=layer,
                            token_pos=args.token_pos,
                            label_A=label_A,
                            label_B=label_B)
            
            if delta is not None:
                np.save(os.path.join(save_dir, f'{i}_{layer}_{token_len}_{digit}.npy'), delta)


