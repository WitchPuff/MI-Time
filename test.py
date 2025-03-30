from transformers import GPT2Tokenizer, GPT2Model
import torch

# 加载模型
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# 编码字符串
text = "1"
tokens = tokenizer.tokenize(str(text))
print(tokens)
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]  # [1, seq_len]
print(inputs)
# 1. token embedding
token_emb = model.wte(input_ids)  # shape: [1, seq_len, hidden]
print(token_emb.shape)
# 2. position embedding
positions = torch.arange(0, input_ids.shape[1]).unsqueeze(0)  # [1, seq_len]
pos_emb = model.wpe(positions)  # [1, seq_len, hidden]
print(pos_emb.shape)
# 3. 加和：即 transformer 的输入表示
combined_emb = token_emb + pos_emb  # shape: [1, seq_len, hidden]

# 4. 拿到整体表示（可做平均、拼接、拿第一个位等）
rep_mean = combined_emb.mean(dim=1).squeeze()  # shape: [hidden]
rep_first = combined_emb[0, 0, :]  # '1' 的位置 + 位置信息

print("1xxx feature (mean pooling):", rep_mean.shape)