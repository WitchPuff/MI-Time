from transformer_lens import HookedTransformer
import torch
import json
from tqdm import tqdm
import random
import numpy as np
import os

# 初始化模型（小模型以便运行更快）
model = HookedTransformer.from_pretrained("gpt2", device='mps')

def get_logit_diff_batch(logits, correct_str=" True", incorrect_str=" False"):
    correct_token = model.to_single_token(correct_str)
    incorrect_token = model.to_single_token(incorrect_str)
    # logits shape: [batch, seq_len, vocab_size]
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    return probs[:, correct_token] - probs[:, incorrect_token]

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




def run_neuron_patching_batch(
    A: str,
    B: str,
    year_A: str,
    year_B: str,
    layer: int,
    label_A: bool,
    label_B: bool,
    token_pos=3,
    correct_str=" True",
    incorrect_str=" False"
):
    # Tokenize A/B
    tokens_A = model.to_tokens(A)  # shape: [1, seq_len]
    tokens_B = model.to_tokens(B)  # shape: [1, seq_len]

    # 检查 A 是否正确分类
    logits_A = model(tokens_A)
    if label_A:
        log_diff_A = get_logit_diff_batch(logits_A, correct_str, incorrect_str)
    else:
        log_diff_A = get_logit_diff_batch(logits_A, incorrect_str, correct_str)
    if log_diff_A <= 0:
        return None

    # 从 A 中获得指定位置（token_pos）在 layer 层的激活
    _, cache_A = model.run_with_cache(tokens_A)
    resid_A = cache_A[f'blocks.{layer}.hook_resid_post']  # [1, seq_len, d_model]
    neuron_A = resid_A[0, token_pos, :]  # [d_model]
    d_model = neuron_A.shape[0]

    # 获取 B 的干净预测，用于对比
    logits_clean_B = model(tokens_B)
    if label_A:
        log_diff_clean = get_logit_diff_batch(logits_clean_B, correct_str, incorrect_str)
    else:
        log_diff_clean = get_logit_diff_batch(logits_clean_B, incorrect_str, correct_str)

    # 复制 tokens_B 为批量输入：每个批次对应一个神经元维度
    tokens_B_batch = tokens_B.repeat(d_model, 1)  # [d_model, seq_len]

    # 定义 hook：对每个 batch 样本，在对应的神经元位置替换成 neuron_A 中的数值
    def patch_hook(act, hook):
        # act: [batch_size, seq_len, d_model]，其中 batch_size == d_model
        batch_indices = torch.arange(act.shape[0])
        act[batch_indices, token_pos, batch_indices] = neuron_A
        return act

    # 使用 hook 运行前向传播，批量计算所有神经元的打补丁效果
    logits_patched = model.run_with_hooks(
        tokens_B_batch,
        fwd_hooks=[(f'blocks.{layer}.hook_resid_post', patch_hook)]
    )

    # 对批量输出计算 logit 差值（每个批次对应一个被打补丁的神经元）
    if label_A:
        log_diff_patched = get_logit_diff_batch(logits_patched, correct_str, incorrect_str)
    else:
        log_diff_patched = get_logit_diff_batch(logits_patched, incorrect_str, correct_str)

    # delta 为打补丁后的 logit 差值与原始 clean 差值的差值（对每个神经元）
    delta = log_diff_patched - log_diff_clean  # 结果形状：[d_model]
    return delta.cpu().tolist()


if __name__ == '__main__':
    # 加载整个数据集（假设是一个数组）
    token_len = 2
    digit = 0
    token_pos = 3
    with open(f'dataset_{token_len}_{digit}.json', 'r') as f:
        data = json.load(f)
    
    batch_size = 256  # 根据需要调整 batch 大小
    min_year, max_year = 1950, 2050
    print(len(data))
    # 按 batch 读取数据集
    for batch_start in tqdm(range(0, len(data), batch_size), desc="Batch Processing"):
        batch = data[batch_start: batch_start + batch_size]
        for j, item in enumerate(batch):
            _, year_A, _, year_B = item
            prompt = get_prompt(year_A, year_B, min_year, max_year)
            A, B, label_A, label_B = prompt

            # 对每个层进行神经元打补丁操作
            for layer in range(5, 8):
                save_dir = f'neuron_patching_results/layer{layer}/tokenlen{token_len}/digit{digit}'
                os.makedirs(save_dir, exist_ok=True)
                delta = run_neuron_patching_batch(
                    A,
                    B,
                    year_A=str(year_A),
                    year_B=str(year_B),
                    layer=layer,
                    token_pos=token_pos,
                    label_A=label_A,
                    label_B=label_B
                )
                
                if delta is not None:
                    # 用全局索引 batch_start+j 保存结果
                    np.save(os.path.join(save_dir, f'{batch_start+j}_{layer}.npy'), delta)
