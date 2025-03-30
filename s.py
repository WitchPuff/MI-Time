import json

dataset = {i: [] for i in range(4)}

for i in range(1, 3):
    with open(f"dataset_token_len_{i}.json") as f:
        data = json.load(f)
    for digit, items in data.items():
            for item in items:
                dataset[int(digit)].append((i, item['year_A'], int(digit), item['year_B']))

    for digit, data in dataset.items():
        with open(f'dataset_{i}_{digit}.json', 'w') as f:
            json.dump(dataset[digit], f)
            print("Done.")
