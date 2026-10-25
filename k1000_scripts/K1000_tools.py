from pathlib import Path
import numpy as np
from scipy import sparse
import pandas as pd
import anndata as ad

sources = ("HCT116", "HEK293T", "K562_VIP", "K562_Replogle", "KOLF")



def estimate_response(pert_sum, pert_sample_n, ctrl_sum, ctrl_n, observed):
    """Calculate one source’s matched controls, r, se, and z.

    Args:
        pert_sum: Sum of perturbation counts.
        pert_sample_n: Number of perturbation samples.
        ctrl_sum: Sum of control counts.
        ctrl_n: Number of control samples.
        observed: Boolean array indicating observed genes.


    Returns:
        A dictionary mapping the source name to its summary:
        r, se, z, X, C.

    """

    per_sample_control_mean = ctrl_sum / ctrl_n[:,None]
    weights = pert_sample_n / pert_sample_n.sum(axis=1)[:,None]
    matched_control = weights @ per_sample_control_mean

    X = np.where(observed, pert_sum, 0)
    masked_control = np.where(observed, matched_control, 0)
    scaling_factor = X.sum(axis=1) / masked_control.sum(axis=1)
    C = masked_control * scaling_factor[:,None]

    r = np.log((X + 0.5) / (C + 0.5))
    se = np.sqrt(1 / (X + 0.5) + 1 / (C + 0.5))
    z = r / se

    z = np.where(observed, z, 0)
    se = np.where(observed,se, np.inf)

    return {
        "r": r,
        "se": se,
        "z": z,
        "X": X,
        "C": C,
    }



def pool_response(panel_perts, source_perts, response):
    """Pool source responses by perturbation name using inverse-variance weights.

    Args:
        panel_perts: Unique perturbation names, shape (P,), defining output
            row order.
        source_perts: Dictionary mapping each source name to its unique
            perturbation names, shape (P_source,), in response-array row
            order. Every name must appear exactly once in panel_perts.
        response: Non-empty dictionary mapping source names to dictionaries
            containing "r" and "se" arrays, each shaped (P_source, G).
            All sources must share the same gene column order. Responses
            must be finite; standard errors must be positive finite values
            or np.inf for unavailable estimates.

    Returns:
        A dictionary of arrays shaped (P, G), with rows in panel_perts order
        and columns in the shared gene order:
            r_pool: Weighted mean response, using weights 1 / se**2.
            se_pool: Pooled standard error, 1 / sqrt(sum of weights).
            z_pool: Pooled response divided by its standard error.
            n_src: Integer count of sources with finite SE at each position.
        Where no source contributes, r_pool and z_pool are 0, se_pool is
        np.inf, and n_src is 0.

    Raises:
        AssertionError: A source perturbation does not match exactly one
            panel entry.
    """
    results = {}
    for source in response:
        SE = response[f"{source}"]["se"]
        r = response[f"{source}"]["r"]

        wt = 1 / (SE ** 2)
        weighted_r = wt * r
        SE_isfinite = np.isfinite(SE)
        unit_n_src = np.where(SE_isfinite, 1, 0)

        results[source] = {
            "unit_sum_weight": wt,
            "unit_weighted_r": weighted_r,
            "unit_n_src": unit_n_src,
        }

    first_source = next(iter(response))

    row = len(panel_perts)
    col = response[first_source]["r"].shape[1]
    
    sum_weight = np.zeros((row, col), dtype = float)
    sum_weighted_r = np.zeros((row, col), dtype = float)
    sum_n_src = np.zeros((row, col), dtype = int)

    for source in response:
        matches = source_perts[source][:, None] == panel_perts[None, :]
        perts_row, perts_col = np.nonzero(matches)

        sum_weight[perts_col, :] += (
            results[source]["unit_sum_weight"][perts_row, :]
       )

        sum_weighted_r[perts_col, :] += (
            results[source]["unit_weighted_r"][perts_row, :]
       )

        sum_n_src[perts_col, :] += (
            results[source]["unit_n_src"][perts_row, :]
        )

        np.testing.assert_array_equal(
            perts_row,
            np.arange(len(source_perts[source]))
        )

    has_weight = sum_weight > 0
    r_pool = np.zeros_like(sum_weight)
    se_pool = np.full_like(sum_weight, np.inf)
    z_pool = np.zeros_like(sum_weight)



    r_pool[has_weight] = sum_weighted_r[has_weight] / sum_weight[has_weight]
    se_pool[has_weight] = 1 / np.sqrt(sum_weight[has_weight])
    z_pool[has_weight] = r_pool[has_weight] / se_pool[has_weight]

    return {
        "r_pool": r_pool,
        "se_pool": se_pool,
        "z_pool": z_pool,
        "n_src": sum_n_src,
    }



def select_top_genes(panel_perts, genes, z_pool, domain):
    """Select up to 1,000 response genes per perturbation for one context.

    Args:
        panel_perts: Perturbation names, shape (P,), in z_pool row order.
        genes: Gene names, shape (G,), in z_pool column order.
        z_pool: Finite pooled z-scores, shape (P, G). Rows are
            perturbations and columns are genes on the shared gene axis.
        domain: Boolean mask, shape (G,), already calculated as the
            target context's control CPM > 5.

    Returns:
        A dictionary mapping each perturbation row index p to a list of
        original gene-column indices. Candidates must be in domain and
        differ from panel_perts[p]. Each list is ordered by descending
        absolute z-score; ties retain the original gene order.
        The list has 1,000 entries when at least 1,000 candidates exist,
        as in the handbook inputs; otherwise it contains all candidates.
        Use genes[indices] for names or z_pool[p, indices] for scores.

    Side effects:
        None. Does not modify inputs, read or write files, or print.
    """
    results = {}                                           # unit is "pert_index": ["gene_index_1", "gene_index_2", ..., "gene_index_1000"] after sorted
    for p, pert in enumerate(panel_perts):
        pairs = []                                         # unit is (gene_index, abs_z_pool)
        intermediate = []                                  # unit is gene_index
        keep = (genes != pert) & domain
        genes_remaining = genes[keep]
        z_remaining = z_pool[p, keep]
        candidate_indices = np.flatnonzero(keep)

        assert genes_remaining.shape == z_remaining.shape

        for g, gene in enumerate(genes_remaining):
            original_index = candidate_indices[g]
            score = np.abs(z_remaining[g])
            pairs.append((original_index, score))
            
        pairs.sort(key=lambda x: x[1], reverse=True)   
        top_1000 = pairs[:1000]
        for i in top_1000:
            intermediate.append(i[0])

        results[p] = intermediate

    return results



def build_lfc(panel_perts, genes, r_pool, domain, selected_indices):
    """Build the Step 6 natural-log fold changes for one target context.

    Args:
        panel_perts: Perturbation names, shape (P,), in r_pool row order.
            Each name must match exactly one entry in genes.
        genes: Gene names, shape (G,), in r_pool column order.
        r_pool: Finite pooled responses, a floating-point array of
            shape (P, G).
        domain: Boolean mask, shape (G,), for this context's control
            CPM > 5.
        selected_indices: Dictionary mapping perturbation row index p
            to selected original gene-column indices. These genes must
            be in domain and exclude panel_perts[p]. Step 5 supplies
            1,000 indices per row for the handbook inputs.

    Returns:
        An array shaped (P, G), with the same floating dtype as r_pool.
        Rows follow panel_perts and columns follow genes. Selected
        response genes receive sign(v) * clip(abs(v), 0.2, 2.0), where
        v = 0.7 * r_pool; an exactly zero v remains zero. Each target
        gene in domain receives min(r_pool_target, -0.5), without the
        0.7 factor or clipping. All other entries remain zero.
        Applying these natural-log fold changes multiplies expression
        by exp(LFC).

    Side effects:
        None. Does not modify inputs, read or write files, or print.
    """
    results = np.zeros_like(r_pool)                          # shape (300, 18533)
    for p, pert in enumerate(panel_perts):
        selected_gene_indices = selected_indices[p]          # shape iteration -> (1000,)
        target_col = np.flatnonzero(genes == pert).item()    # the column index of the current pert, used to be excluded


        for gene_index in selected_gene_indices:
            selected_r_pool = r_pool[p, gene_index]          # scalar, not a tensor
            v = 0.7 * selected_r_pool
            v_abs = np.abs(v)
            LFC = np.sign(v) * np.clip(v_abs, 0.2, 2.0)
            results[p, gene_index] = LFC


        if domain[target_col]:
            excluded_r_pool = r_pool[p, target_col]          # scalar
            LFC_target = min(excluded_r_pool, -0.5)
            results[p, target_col] = LFC_target

    return results



def stochastic_round(values, rng):       
    """Round fractional counts to integers without bias in expectation.

    Args:
        values: NumPy array of finite, nonnegative counts. Any shape is
            supported; Step 7 uses (400, G), with cells on axis 0 and
            genes on axis 1. Counts must fit in np.int64 after rounding.
        rng: A NumPy Generator created by the caller. Uses its current
            state without resetting the seed.

    Returns:
        A new np.int64 array with the same shape as values. Each entry
        is floor(value) plus 1 with probability value - floor(value).
        Integer inputs remain unchanged; each output's expectation equals
        its input, but a single draw need not preserve the array's total.

    Side effects:
        Advances rng by one uniform draw per entry. Does not modify values.
    """
    interger_part = np.floor(values).astype(np.int64)        # shape (400, 18533), predictions of 400 cells' 18533 gene expression
    fractional_part = values - interger_part                 # shape (400, 18533), and all below in this block
    random_draw = rng.random(values.shape)
    add_one = random_draw < fractional_part
    rounded_value = interger_part + add_one

    return rounded_value



def save_prediction_h5ad(final_predictions, panel_perts, genes, output_dir):
    """Save A/B/C predicted cell counts and their labels in one H5AD file.

    Args:
        final_predictions: Nested dictionary, context -> perturbation row
            index -> count array. Contexts must include A, B, and C; each
            context must contain indices 0 through len(panel_perts) - 1.
            Each array has shape (n_cells, G), with cells on axis 0 and
            genes on axis 1, and contains finite, nonnegative integer-valued
            counts. The current builder supplies 400 cells per perturbation.
        panel_perts: Perturbation names, shape (P,). panel_perts[i] labels
            the cells in final_predictions[context][i].
        genes: Gene names, shape (G,), in the same column order as every
            count array. Used as the AnnData variable index.
        output_dir: String or Path naming the destination directory.

    Returns:
        A Path to output_dir / "K1000_predictions.h5ad". The file stores
        X as a CSR count matrix, obs with context and target_gene columns,
        and var indexed by genes. Rows are ordered by context A/B/C, then
        panel_perts order, then cell order within each prediction block.
        Cell identifiers are context_i_c. With 300 perturbations and 400
        cells each, X has shape (360000, G).

    Side effects:
        Creates the destination directory if needed and writes a gzip-
        compressed H5AD file, replacing an existing file at the same path.
        Does not modify the inputs or normalize the counts. All blocks are
        assembled in memory before writing; this is not a streaming export.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "K1000_predictions.h5ad"


    matrix_blocks = []
    obs_blocks = []

    for context in ["A", "B", "C"]:
        for i, pert in enumerate(panel_perts):
            counts = final_predictions[context][i]
            sparse_counts = sparse.csr_matrix(counts)
            matrix_blocks.append(sparse_counts)

            obs_block = pd.DataFrame(
                {"context": context, "target_gene": pert},
                index=[f"{context}_{i}_{c}" for c in range(counts.shape[0])]
            )
            obs_blocks.append(obs_block)

    X = sparse.vstack(matrix_blocks, format="csr")
    obs = pd.concat(obs_blocks)
    var = pd.DataFrame(index = genes)

    adata = ad.AnnData(X = X, obs = obs, var = var)
    adata.write_h5ad(output_path, compression = "gzip")

    return output_path
