import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import torch 
from tqdm import tqdm

def get_prompt(year, digit=None):
    year = year if isinstance(year, str) else str(year)
    if digit == None:
        return f"The year is {year}."
    d2w = {0: 'first', 1: 'second', 2: 'third', 3: 'last'}
    return f"The {d2w[digit]} digit of the year is {year[digit]}."

def get_embedding(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden]
    return hidden_states.squeeze().mean(dim=0).detach().numpy()




def prob(X, y):
    
    X = np.array(X)
    y = np.array(y)

    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)

    plt.figure(figsize=(8,6))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
    plt.xlabel("Real Year")
    plt.ylabel("Predicted Year")
    plt.title("Linear Regression: Predict Year Using GPT-2 Prompt Embedding")
    plt.grid(True)
    plt.show()

    W = reg.coef_  # shape: (4*D,) / (D,)
    D = model.config.n_embd
    print("W shape:", W.shape)
    print("D:", D)

    for i in range((W.shape[0]//D)):
        Wi = W[i*D:(i+1)*D]
        wi_norm = np.linalg.norm(Wi)
        print(f"{i+1} digit Weight Norm: {wi_norm:.2f}")
        print(f"{i+1} digit token Weight Avg: {np.mean(np.abs(Wi)):.4f}")

    # W1 = W[:D]
    # W2 = W[D:2*D]
    # W3 = W[2*D:3*D]
    # W4 = W[3*D:]

    # w1_norm = np.linalg.norm(W1)
    # w2_norm = np.linalg.norm(W2)
    # w3_norm = np.linalg.norm(W3)
    # w4_norm = np.linalg.norm(W4)

    # print(f"1st digit Weight Norm: {w1_norm:.2f}")
    # print(f"2nd digit Weight Norm: {w2_norm:.2f}")
    # print(f"3rd digit Weight Norm: {w3_norm:.2f}")
    # print(f"4th digit Weight Norm: {w4_norm:.2f}")

    # print(f"1st digit token Weight Avg: {np.mean(np.abs(W1)):.4f}")
    # print(f"2nd digit token Weight Avg: {np.mean(np.abs(W2)):.4f}")
    # print(f"3rd digit token Weight Avg: {np.mean(np.abs(W3)):.4f}")
    # print(f"4th digit token Weight Avg: {np.mean(np.abs(W4)):.4f}")
    
    
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

years = list(range(1500, 2100))

split_digits = []
all_digits = []
y = []

for year in tqdm(years):
    embs = []
    for i in range(4):
        embs.append(get_embedding(get_prompt(year, i)))
    embs = np.concatenate(embs)
    split_digits.append(embs)
    all_digits.append(get_embedding(get_prompt(year)))
    y.append(int(year))

print('==== split digits ====')
prob(split_digits, y)
print('==== all digits ====')
prob(all_digits, y)

