import json
import matplotlib.pyplot as plt
import numpy as np

# Load the results

results_files = [
    f"results_1.json",
    f"results_2_3.json",
    f"results_2_4.json",
    f"results_2_34.json"
]
for results_file in results_files:
    with open(results_file, 'r') as f:
        results = json.load(f)
    token_len = results_file.split("_")[1]
    patch_pos = results_file.split("_")[-1].split(".")[0]
    layers = sorted([int(k) for k in results.keys()])
    token_lens = sorted({int(tk) for l in results.values() for tk in l.keys()})
    digit_positions = sorted({int(d) for l in results.values() for tl in l.values() for d in tl.keys()})

    # Plot
    plt.figure(figsize=(10, 6))

    for token_len in token_lens:
        for digit in digit_positions:
            mean_deltas = []
            for layer in layers:
                try:
                    deltas = results[str(layer)][str(token_len)][str(digit)]['delta']
                    mean_delta = np.mean(deltas)
                except KeyError:
                    mean_delta = np.nan  # or 0
                mean_deltas.append(mean_delta)

            plt.plot(layers, mean_deltas, label=f"token_len={token_len}, digit={digit}")

    plt.xlabel("Layer")
    plt.ylabel("Mean Delta (logit_diff_patched - clean)")
    plt.title(f"Activation Patching Effect Across Layers (Patch Position {patch_pos})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results_{token_len}_{patch_pos}.png")
