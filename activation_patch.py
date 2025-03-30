from transformer_lens import HookedTransformer
import torch

# 加载 GPT-2 small 模型（TransformerLens 格式）
model = HookedTransformer.from_pretrained("gpt2", device="mps")
# 找出 " yes" 和 " no" 的 token ID
yes_token = model.to_single_token(" Yes")
no_token = model.to_single_token(" No")
# 定义两个 prompt
prompt_clean = "Q: Is the year 1987 earlier than the year 2023?\nA:"
prompt_corrupt = "Q: Is the year 1987 earlier than the year 1823?\nA:"

# 将文本转换为模型的 token 表示
tokens_clean = model.to_tokens(prompt_clean)
tokens_corrupt = model.to_tokens(prompt_corrupt)

# 得到原始输出，便于比较
logits_clean = model(tokens_clean)
pred_clean = model.to_string(logits_clean.argmax(dim=-1)[0])
next_token_logits = logits_clean[0, -1]
topk_clean = model.to_string(torch.topk(next_token_logits, k=5).indices)
print("[Clean] Top-5 predictions:", topk_clean)
print("yes: ", next_token_logits[yes_token].item())
print("no:", next_token_logits[no_token].item())


logits_corrupt = model(tokens_corrupt)
pred_corrupt = model.to_string(logits_corrupt.argmax(dim=-1)[0])
next_token_logits = logits_corrupt[0, -1]
topk_corrupt = model.to_string(torch.topk(next_token_logits, k=5).indices)
print("[Corrupt] Top-5 predictions:", topk_corrupt)
print("yes: ", next_token_logits[yes_token].item())
print("no:", next_token_logits[no_token].item())

# 设置我们要 patch 的层和位置
# 这里我们选择第 8 层的 residual stream（patch 前置 hook）
layer = 10
hook_name = f"blocks.{layer}.hook_resid_pre"

# 用于保存干净句子在该层的激活
clean_activations = {}

def save_clean_hook(act, hook):
    clean_activations["act"] = act

# 运行干净 prompt 并保存激活
_ = model.run_with_hooks(tokens_clean, return_type=None, fwd_hooks=[(hook_name, save_clean_hook)])

# 定义 patch hook：将被污染句子在指定位置的激活替换为干净句子的激活
def patch_hook(act, hook):
    patched = act.clone()
    # 假设我们替换 tokens_clean 中位置 9 的激活（具体位置需根据 token 化结果确定）
    patched[0, 9] = clean_activations["act"][0, 9]
    return patched

# 使用 patch hook 运行被污染的 prompt
logits_patched = model.run_with_hooks(tokens_corrupt, return_type="logits", fwd_hooks=[(hook_name, patch_hook)])
pred_patched = model.to_string(logits_patched.argmax(dim=-1)[0])
next_token_logits = logits_patched[0, -1]
topk_patched = model.to_string(torch.topk(next_token_logits, k=5).indices)
print("[Patched] Top-5 Prediction:", topk_patched)
print("yes: ", next_token_logits[yes_token].item())
print("no:", next_token_logits[no_token].item())