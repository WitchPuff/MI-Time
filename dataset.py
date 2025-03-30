from transformer_lens import HookedTransformer
import json
from tqdm import tqdm

model = HookedTransformer.from_pretrained("gpt2", device='mps')

def get_position(tokens: list, year: int):
    for i, token in enumerate(tokens):
        if str(year) in token:
            return (i,), (token, )
        elif ''.join([token, tokens[i+1]]).replace(' ', '') == str(year):
            return (i, i+1), (token, tokens[i+1])
    return None

def get_dataset(year: int, max=2100, min=1000):
    assert year <= max and year >= min, f"Year Range: {min}-{max}, {year}"
    prompt = f"The year {year} is earlier than the year {min}."
    tokens = model.to_tokens(prompt)
    
    year_pos, year_tokens = get_position(model.to_str_tokens(tokens), year)
    safe_token_len = len(year_tokens)
    
    year = year if isinstance(year, str) else str(year)
    
    safe_replacements = {}
    
    for pos in range(4):  # each digit position
        safe_replacements[pos] = {}
        for new_digit in range(10):
            if str(new_digit) == year[pos]:
                continue  # skip same digit
            replaced = list(year)
            replaced[pos] = str(new_digit)
            new_year = ''.join(replaced)
            if int(new_year) > max or int(new_year) < min: continue
            new_year_tokens = model.to_tokens(f"The year {new_year} is earlier than the year {min}.")
            new_year_pos, new_year_tokens = get_position(model.to_str_tokens(new_year_tokens), int(new_year))
            if len(new_year_tokens) == safe_token_len:
                safe_replacements[pos].update({new_year: new_year_tokens})
            # else:
            #     print(f"Break at pos {pos}: replacing with {new_digit} → {new_year} → {tokenizer.tokenize(new_year)}")
            #     # break  # Once we split into multiple tokens, stop further changes at this position
    
    return safe_replacements, safe_token_len


# demo
years = list(range(1950, 2050))
dataset = {
    1: {},
    2: {}
}
for year in tqdm(years):
    safe, safe_token_len = get_dataset(year)
    dataset[safe_token_len][year] = safe

with open('dataset.json', 'w') as f:
    json.dump(dataset, f, indent=4)
    # print("\nSafe replacements:")
    # for pos, years in safe.items():
    #     print(f"  Digit position {pos} → {years}")