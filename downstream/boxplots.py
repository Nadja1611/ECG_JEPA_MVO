import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch 
import os

def boxplot(tp_indices, fn_indices, lesion_volumes, data_dir=None):
    # Extract FN and TP lesion volumes
    fn_volumes = lesion_volumes[fn_indices]
    mean_fn = fn_volumes.mean().item()
    median_fn = fn_volumes.median().item()
    print('fn' , mean_fn, flush = True)
    tp_volumes = lesion_volumes[tp_indices]
    mean_tp = tp_volumes.mean().item()
    median_tp = tp_volumes.median().item()
    print('tp ' , mean_tp, flush = True)

    # Convert to Pandas DataFrame for Seaborn
    df = pd.DataFrame({
        "Lesion Volume": torch.cat([fn_volumes, tp_volumes]).numpy(),
        "Type": ["FN"] * len(fn_volumes) + ["TP"] * len(tp_volumes)
    })

    # Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Type", y="Lesion Volume", data=df, 
                palette={"FN": "#A1CAF1", "TP": "#FBC15E"})  # Pastel colors

    # Add annotations for mean and median
    plt.text(0, mean_fn, f"Mean: {mean_fn:.2f}", ha='center', va='bottom', fontsize=10, color="blue")
    plt.text(0, median_fn, f"Median: {median_fn:.2f}", ha='center', va='top', fontsize=10, color="blue")

    plt.text(1, mean_tp, f"Mean: {mean_tp:.2f}", ha='center', va='bottom', fontsize=10, color="darkorange")
    plt.text(1, median_tp, f"Median: {median_tp:.2f}", ha='center', va='top', fontsize=10, color="darkorange")

    # Labels & Title
    plt.title("Lesion Volume Distribution for FN and TP")
    plt.xlabel("Category")
    plt.ylabel("Lesion Volume")
    plt.grid(True)

    # Save figure
    plt.savefig('boxplot'+ os.path.basename(data_dir)+'.png', dpi=300)
    plt.show()
