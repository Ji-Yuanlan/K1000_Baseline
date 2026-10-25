from pathlib import Path
import numpy as np

data_path = Path(__file__).resolve().parent.parent / "k1000_inputs.npz"
sources = ("HCT116", "HEK293T", "K562_VIP", "K562_Replogle", "KOLF")
checkpoints = ("Knockdowns", "Observed genes", "Samples", "Control cells", "Median cells per knockdown")

def summarize_source(source):
    """Calculate five summary statistics for one source.

    Args:
        source: Source name in the input archive, such as "HCT116".

    Returns:
        A dictionary mapping the source name to its summary:
        knockdowns, observed genes, samples, total control cells,
        and median cells per knockdown.

    Reads the archive at data_path without modifying it.
    """
        
    source = str(source)
    with np.load(data_path) as data:
        results = {}
        n_knockdowns = len(data[f"{source}__perts"])
        n_observed_genes = np.count_nonzero(data[f"{source}__observed"])
        n_samples = len(data[f"{source}__ctrl_n"])
        n_control_cells = int(data[f"{source}__ctrl_n"].sum())
        n_median_cells__per_knockdown = np.median(data[f"{source}__pert_sample_n"].sum(axis=1))
        results[source] = {
            "Knockdowns": n_knockdowns,
            "Observed genes": n_observed_genes,
            "Samples": n_samples,
            "Control cells": n_control_cells,
            "Median cells per knockdown": n_median_cells__per_knockdown,
        }

    return results



def main():
    """Collect summaries for all five sources and print a comparison table."""

    results = {}
    for source in sources:
        results.update(summarize_source(source))

    header = f"{'Chekpoint':<28}" + "|" + "|".join(f"{source:>14}" for source in sources)

    print(header)
    print("-" * len(header))

    for checkpoint in checkpoints:
        print(f"{checkpoint:<28}" + "|" + "|".join(f"{results[source][checkpoint]:>14}" for source in sources))



if __name__ == "__main__":
    main()