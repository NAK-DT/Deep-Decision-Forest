import matplotlib.pyplot as plt

# Accuracy data for each model
accuracy_data = {
    "Mushroom": {
        "DT": 0.990,
        "RF": 0.998,
        "DF": 0.999,
        "DF (restricted)": 0.998,
        "DDF (Ours)": 0.998,
    },
    "Banknote": {
        "DT": 0.961,
        "RF": 0.996,
        "DF": 0.999,
        "DF (restricted)": 0.996,
        "DDF (Ours)": 0.990,
    },
    "Raisin": {
        "DT": 0.820,
        "RF": 0.826,
        "DF": 0.839,
        "DF (restricted)": 0.827,
        "DDF (Ours)": 0.841,
    },
    "Diabetes": {
        "DT": 0.699,
        "RF": 0.732,
        "DF": 0.739,
        "DF (restricted)": 0.732,
        "DDF (Ours)": 0.730,
    },
    "Heart disease": {
        "DT": 0.744,
        "RF": 0.849,
        "DF": 0.834,
        "DF (restricted)": 0.839,
        "DDF (Ours)": 0.845,
    },
    "Iris": {
        "DT": 0.956,
        "RF": 0.950,
        "DF": 0.956,
        "DF (restricted)": 0.950,
        "DDF (Ours)": 0.966,
    },
    "Maternal health": {
        "DT": 0.685,
        "RF": 0.745,
        "DF": 0.817,
        "DF (restricted)": 0.745,
        "DDF (Ours)": 0.765,
    },
}

# Redundancy scores
redundancy_scores = {
    "Mushroom": 0.250,
    "Banknote": 0.208,
    "Raisin": 0.524,
    "Diabetes": 0.000,
    "Heart disease": 0.038,
    "Iris": 0.500,
    "Maternal health": 0.117,
}

# Plotting
plt.figure(figsize=(10, 7))
markers = {
    "DT": "o",
    "RF": "s",
    "DF": "^",
    "DF (restricted)": "v",
    "DDF (Ours)": "D",
}
colors = {
    "DT": "#1f77b4",
    "RF": "#ff7f0e",
    "DF": "#2ca02c",
    "DF (restricted)": "#d62728",
    "DDF (Ours)": "#9467bd",
}

for dataset, scores in accuracy_data.items():
    redundancy = redundancy_scores[dataset]
    for model, acc in scores.items():
        plt.scatter(acc, redundancy, marker=markers[model], color=colors[model], label=model if dataset == "Mushroom" else "")
        plt.text(acc + 0.002, redundancy + 0.005, dataset, fontsize=8)

plt.xlabel("Accuracy", fontsize=12)
plt.ylabel("Redundancy Score", fontsize=12)
plt.title("Model Accuracy vs Redundancy Score (by Dataset)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
