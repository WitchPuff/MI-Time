import json

with open("dataset.json") as f:
    raw_data = json.load(f)

flat_data = {i:[] for i in range(4)}
min, max = 1850, 2025
target_token_len = "2"
def flatten_data(raw_data, min_digit=3, max_digit=0):
    for token_len, years in raw_data.items():
        if token_len == target_token_len:
            for year_A, digits in years.items():
                for digit, replacements in digits.items():
                    for year_B, tokens_B in replacements.items():
                        year_A, year_B = int(year_A), int(year_B)
                        # if len(flat_data[int(digit)]) > 100 and year_A < 2000: continue
                        if len(flat_data[int(digit)]) > 200: continue
                        if digit == "2" or digit == "3":
                            if year_A <= min or year_A >= max or year_B <= min or year_B >= max: continue
                        flat_data[int(digit)].append({
                            "token_len": int(token_len),
                            "year_A": int(year_A),
                            "year_B": int(year_B),
                            "tokens_B": tokens_B
                        })

                        if min_digit > int(digit):
                            min_digit = int(digit)
                        if max_digit < int(digit):
                            max_digit = int(digit)
                        # if len(flat_data) > 250:
    print(min_digit, max_digit)
    print(len(flat_data[0]), len(flat_data[1]), len(flat_data[2]), len(flat_data[3]))
    return flat_data
    

# 保存为新文件
with open(f"dataset_token_len_{target_token_len}.json", "w") as f:
    json.dump(flatten_data(raw_data), f, indent=2)