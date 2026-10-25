from pathlib import Path
import numpy as np
from K1000_tools import estimate_response, pool_response, select_top_genes, build_lfc, stochastic_round, save_prediction_h5ad


data_path = Path(__file__).resolve().parent.parent / "k1000_inputs.npz"
sources = ("HCT116", "HEK293T", "K562_VIP", "K562_Replogle", "KOLF")


def build_predictions(data_path=data_path):
    """Build source responses, A/B/C LFCs, and predictions for all 300 perturbations.

    Args:
        data_path: Path to the input NPZ file with the 300-perturbation
            handbook panel. All sources must share the same gene column order.

    Returns:
        A dictionary containing:
            panel_perts: Output perturbation names, shape (P,).
            genes: Shared gene column names, shape (G,).
            source_perts: Source names mapped to perturbation names,
                each shaped (P_source,).
            source_observed: Source names mapped to observed-gene masks,
                each shaped (G,).
            source_responses: Source names mapped to X, C, r, se, and z arrays,
                each shaped (P_source, G) in that source's row order.
            pooled: r_pool, se_pool, z_pool, and n_src arrays, each
                shaped (P, G) in panel_perts row order.
            domains: A, B, and C mapped to Boolean masks of shape (G,),
                indicating target control CPM > 5.
            selected_indices: A, B, and C mapped to dictionaries of
                perturbation row index -> ranked original gene indices.
                Each list has 1,000 entries for the handbook inputs.
            built_lfc: A, B, and C mapped to natural-log fold-change
                arrays, each shaped (P, G) in panel_perts and genes order.
                Includes selected response genes and target genes in
                domain; all other entries are zero.
            control_cells: Nested dictionary, context A/B/C -> perturbation
                index i -> simulated control counts shaped (400, G). Within
                each context, all 300 indices reference the same sampled
                control array; cells are on axis 0 and genes on axis 1.
            final_predictions: Nested dictionary, context A/B/C -> perturbation
                index i -> rounded predicted counts shaped (400, G), dtype
                np.int64. Each array uses built_lfc[context][i] and predicts
                panel_perts[i], with columns in genes order.

    Side effects:
        Reads the input file. Does not write files or print checkpoints.
    """
    source_responses = {}
    source_perts = {}
    source_observed = {}
    domains = {}

    with np.load(data_path) as data:
        panel_perts = data["panel__perts"]
        genes = data["HCT116__genes"]

        # Step 2: calculate each source in its own perturbation row order.
        for source in sources:
            source_perts[source] = data[f"{source}__perts"]
            source_observed[source] = data[f"{source}__observed"]

            source_responses[source] = estimate_response(
                data[f"{source}__pert_sum"],
                data[f"{source}__pert_sample_n"],
                data[f"{source}__ctrl_sum"],
                data[f"{source}__ctrl_n"],
                source_observed[source],
            )

        # Step 3: align perturbations by name and pool all five sources once.
        pooled = pool_response(panel_perts, source_perts, source_responses)

        # Step 4: add target domains here while the input file is open.
        for i in ["A", "B", "C"]:
            domains[i] = data[f"target_{i}__control_cpm"] > 5

    # Step 5: select response genes separately for each target domain.
    selected_indices = {}
    for context in ["A", "B", "C"]:
        selected_indices[context] = select_top_genes(
            panel_perts, genes, pooled["z_pool"], domains[context]
        )

    # Step 6: build LFC separately for target contexts A, B, and C.
    # Reuse pooled r; each context has its own domain and response picks.
    # build_lfc also handles the target gene when it is in that domain.
    built_lfc = {}
    for context in ["A", "B", "C"]:
        built_lfc[context] = build_lfc(
            panel_perts, genes, pooled["r_pool"], domains[context], selected_indices[context]
        )

    # Step 7: reuse 400 simulated controls within each context to predict
    # all 300 perturbations, applying each perturbation's own LFC row.
    # Use one seeded RNG for control sampling and every rounding call.
    rng = np.random.default_rng(20261021)
    control_cells = {}
    final_predictions = {}
    with np.load(data_path) as data:
        for context in ["A", "B", "C"]:
            cpm = data[f"target_{context}__control_cpm"]              # shape (18533,)
            lam = cpm / 1e6 * 10000                                   # Poisson means per gene at about 10,000 counts per cell
            cells = rng.poisson(lam = lam, size = (400, len(lam)))    # shape (400, 18533)
            control_cells[context] = {}
            final_predictions[context] = {}
            for i in range(300):
                lfc = built_lfc[context][i]                               # shape (18533,)
                prediction = cells * np.exp(lfc)                          # broadcast, (400, 18533) * (18533,) -> (400, 18533)
                rounded_prediction = stochastic_round(prediction, rng)
                control_cells[context][i] = cells
                final_predictions[context][i] = rounded_prediction

    return {
        "panel_perts": panel_perts,
        "genes": genes,
        "source_perts": source_perts,
        "source_observed": source_observed,
        "source_responses": source_responses,
        "pooled": pooled,
        "domains": domains,
        "selected_indices": selected_indices,
        "built_lfc": built_lfc,
        "control_cells": control_cells,
        "final_predictions": final_predictions,
    }



def main():
    built = build_predictions()
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "results"

    with np.load(data_path) as data:
        final_predictions = built["final_predictions"]
        panel_perts = data["panel__perts"]
        genes = data["HCT116__genes"]

        output_path = save_prediction_h5ad(
                                        final_predictions,
                                        panel_perts,
                                        genes,
                                        output_dir
                                    )

    print(f"Results have been saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    main()
