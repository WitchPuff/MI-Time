import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 构造“高位变化”和“低位变化”任务
high_digit_pairs = [("1987", "2087"), ("1977", "2077"), ("1991", "2091"), ("1965", "2065")]
low_digit_pairs  = [("1987", "1989"), ("1977", "1978"), ("1991", "1992"), ("1965", "1968")]

labels = [1, 1, 1, 1] + [1, 1, 1, 1]  # always "year1 < year2"

# 提取 hidden state（最后一层）
def get_token_hidden_state(sentence, token_str, layer=-1):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states[layer][0]  # [seq_len, hidden_dim]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(tokens)
    try:
        token_index = tokens.index(token_str)
    except:
        print("token not found:", token_str)
        return None
    return hidden_states[token_index].cpu().numpy()

X_high, X_low, y = [], [], []

# 高位变化数据集
for y1, y2 in high_digit_pairs:
    sent = f"The year {y1} is earlier than the year {y2}."
    h1 = get_token_hidden_state(sent, y1)
    h2 = get_token_hidden_state(sent, y2)
    if h1 is not None and h2 is not None:
        # 只用 year1 的表示来预测是不是更早（就是预测 1）
        X_high.append(h1)
        y.append(1)

# 低位变化数据集
for y1, y2 in low_digit_pairs:
    sent = f"The year {y1} is earlier than the year {y2}."
    h1 = get_token_hidden_state(sent, y1)
    h2 = get_token_hidden_state(sent, y2)
    if h1 is not None and h2 is not None:
        X_low.append(h1)
        y.append(1)

X_high = np.array(X_high)
X_low = np.array(X_low)
y = np.array(y)

# 训练两个 probe
probe_high = LogisticRegression(max_iter=1000).fit(X_high, y[:len(X_high)])
probe_low = LogisticRegression(max_iter=1000).fit(X_low, y[len(X_high):])

acc_high = accuracy_score(y[:len(X_high)], probe_high.predict(X_high))
acc_low = accuracy_score(y[len(X_high):], probe_low.predict(X_low))

print(f"🔍 Probe trained on 高位变化样本：准确率 = {acc_high:.3f}")
print(f"🔍 Probe trained on 低位变化样本：准确率 = {acc_low:.3f}")