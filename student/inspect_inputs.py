from pathlib import Path
import numpy as np

data_path = Path(__file__).resolve().parent.parent / "k1000_inputs.npz"
cell_lines = ("HCT116", "HEK293T", "K562_VIP", "K562_Replogle", "KOLF")
checkpoints = ("Knockdowns", "Observed genes", "Samples", "Control cells", "Median cells per knockdown")

def summarize_source(source):
    cell_line = str(source)
    with np.load(data_path) as data:
        results = {}
        n_knockdowns = len(data[f"{cell_line}__perts"])
        n_observed_genes = np.count_nonzero(data[f"{cell_line}__observed"])
        n_samples = len(data[f"{cell_line}__ctrl_n"])
        n_control_cells = int(data[f"{cell_line}__ctrl_n"].sum())  # without a thousand separator yet
        n_median_cells__per_knockdown = np.median(data[f"{cell_line}__pert_sample_n"].sum(axis=1))
        results[cell_line] = {
            "Knockdowns": n_knockdowns,
            "Observed genes": n_observed_genes,
            "Samples": n_samples,
            "Control cells": n_control_cells,
            "Median cells per knockdown": n_median_cells__per_knockdown,
        }

    return results



# Step 1: Load the data and look
def main():
    results = {}
    for cell_line in cell_lines:
        results.update(summarize_source(cell_line))

    header = f"{'Chekpoint':<28}" + "|" + "|".join(f"{cell_line:>14}" for cell_line in cell_lines)

    print(header)
    print("-" * len(header))

    for checkpoint in checkpoints:
        print(f"{checkpoint:<28}" + "|" + "|".join(f"{results[cell_line][checkpoint]:>14}" for cell_line in cell_lines))



if __name__ == "__main__":
    main()