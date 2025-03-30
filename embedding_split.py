import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import torch 

# 加载 GPT-2 模型与 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

# 收集数据：只选 GPT-2 tokenizer 会把年份拆成 ['18', '87'] 这样的情况
def collect_token_pairs(start=1500, end=2050):
    pairs = []
    for year in range(start, end + 1):
        tokens = tokenizer.tokenize(str(year))
        if len(tokens) == 2:
            pairs.append((tokens, year))
    return pairs

def get_prompt(year, digit):
    d2w = {1: 'first', 2: 'second', 3: 'third', 4: 'last'}
    return f"The {d2w[digit]} digit of the year is {year[digit]}."

data = collect_token_pairs()
print(f"共找到 {len(data)} 个可以拆成两个 token 的年份样本。")

# 提取 embedding
def get_embedding(token):
    token_id = tokenizer.convert_tokens_to_ids(token)
    with torch.no_grad():
        embedding = model.wte.weight[token_id].detach().numpy()
    return embedding

X = []
y = []

for (tokens, year) in data:
    emb1 = get_embedding(tokens[0])
    emb2 = get_embedding(tokens[1])
    feature = np.concatenate([emb1, emb2])  # 拼接两个token的embedding
    X.append(feature)
    y.append(year)

X = np.array(X)
y = np.array(y)

# 训练线性回归模型
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)

# 可视化预测 vs 真实
plt.figure(figsize=(8,6))
plt.scatter(y, y_pred, alpha=0.6)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
plt.xlabel("真实年份")
plt.ylabel("预测年份")
plt.title("线性回归：使用 GPT-2 的 Token Embedding 预测年份")
plt.grid(True)
plt.show()

# 分析权重：将前一半和后一半分开看
coef = reg.coef_
dim = model.config.n_embd
w1 = coef[:dim]   # 高位 token
w2 = coef[dim:]   # 低位 token

print(f"高位 token 权重均值: {np.mean(np.abs(w1)):.4f}")
print(f"低位 token 权重均值: {np.mean(np.abs(w2)):.4f}")