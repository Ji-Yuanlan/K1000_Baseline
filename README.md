# K1000 Baseline

A Python implementation of the K1000 sparse transfer baseline for predicting gene-expression responses to genetic perturbations across cellular contexts. The pipeline combines five source screens, estimates perturbation effects and their uncertainty, selects 1,000 response genes per perturbation and target context, and exports predicted cell counts in AnnData format.

The project connects statistical estimation with a reproducible numerical workflow: sample-matched controls, inverse-variance pooling, stable gene selection, log fold-change construction, stochastic rounding, and annotated count-matrix export.

## Analysis overview

| Component | Configuration |
|---|---|
| Source screens | HCT116, HEK293T, K562_VIP, K562_Replogle, KOLF |
| Target contexts | A, B, C |
| Perturbation panel | 300 perturbations |
| Shared gene axis | 18,533 genes |
| Response selection | 1,000 genes per perturbation and context, plus the target gene when eligible |
| Predicted cells | 400 per perturbation and context; 360,000 across the full panel |
| Input | Preprocessed `k1000_inputs.npz` |
| Output | `results/K1000_predictions.h5ad` |
| Random seed | `20261021` |

## Method

### Response estimation

For each source, control expression is matched to each perturbation's sample composition. Unobserved genes are masked, and matched control counts are scaled to the perturbation's total observed count depth.

For perturbation counts `X` and matched control counts `C`, the response, standard error, and standardized response are:

```text
r  = ln((X + 0.5) / (C + 0.5))
SE = sqrt(1 / (X + 0.5) + 1 / (C + 0.5))
z  = r / SE
```

All logarithms are natural logarithms. Unobserved genes receive `r = 0`, `z = 0`, and `SE = inf`, giving them zero weight in pooling.

### Cross-source pooling

Perturbation rows are aligned by name before combining sources; gene columns share the same order. Inverse-variance weights combine the source responses:

```text
w       = 1 / SE²
r_pool  = sum(w * r) / sum(w)
se_pool = 1 / sqrt(sum(w))
z_pool  = r_pool / se_pool
```

`n_src` records how many sources contribute a finite standard error to each perturbation-gene pair. Positions with no contributing source retain zero response and infinite standard error.

### Gene selection and fold changes

Each target context defines an expression domain using control CPM > 5. For each perturbation, the pipeline excludes its own target gene from the response candidates and selects the 1,000 largest absolute pooled z-scores. Stable sorting preserves the original gene order for tied scores.

Selected response genes receive:

```text
v   = 0.7 * r_pool
LFC = sign(v) * clip(abs(v), 0.2, 2.0)
```

An exactly zero response remains zero. The perturbation's target gene is handled separately: when it belongs to the expression domain, its LFC is `min(r_pool_target, -0.5)`. All other genes retain zero LFC.

### Cell-count generation

For each target context, 400 control profiles are simulated from Poisson distributions with gene-wise means `control_cpm / 1e6 * 10000`. The same control profiles are reused across that context's perturbations, with each perturbation receiving its own LFC vector:

```text
predicted_counts = control_counts * exp(LFC)
```

Stochastic rounding converts fractional predictions to integer counts: a value `x` becomes `floor(x) + 1` with probability `x - floor(x)`, and `floor(x)` otherwise. This preserves the expected count, while genes with zero LFC remain exactly unchanged. Control sampling and rounding use a single seeded NumPy generator.

## Input data

Use the project's preprocessed **`k1000_inputs.npz` directly**. Place this approximately 60 MB archive in the project root, alongside `pyproject.toml`. It contains source count summaries, perturbation and gene identifiers, observed-gene masks, and target-context control CPM vectors ready for analysis.

The scripts locate the archive relative to the project directory. Data archives are kept separate from Git-tracked source files.

`P_source` is the number of perturbations in a source, `S` is its number of samples, and `G = 18533`.

| Field | Shape | Description |
|---|---|---|
| `panel__perts` | `(300,)` | Shared output perturbation order |
| `<source>__perts` | `(P_source,)` | Source perturbation names |
| `<source>__genes` | `(G,)` | Shared gene column order |
| `<source>__observed` | `(G,)` | Observed-gene mask |
| `<source>__pert_sum` | `(P_source, G)` | Summed perturbation counts |
| `<source>__pert_sample_n` | `(P_source, S)` | Perturbation cell counts by sample |
| `<source>__ctrl_sum` | `(S, G)` | Summed control counts by sample |
| `<source>__ctrl_n` | `(S,)` | Control cell counts by sample |
| `target_A__control_cpm` | `(G,)` | Mean per-cell control CPM for A |
| `target_B__control_cpm` | `(G,)` | Mean per-cell control CPM for B |
| `target_C__control_cpm` | `(G,)` | Mean per-cell control CPM for C |

`<source>` is one of the five source names listed above. The input contains 300 perturbations for HCT116, HEK293T, and K562_VIP; 272 for K562_Replogle; and 282 for KOLF. Name-based alignment places each source in the shared output panel.

## Running the analysis

Requirements: **Python >= 3.14**, **uv**, and a compute environment with sufficient memory for the full panel. The retained dense prediction arrays require approximately **49.7 GiB**; source statistics, temporary arrays, and export buffers require additional memory. Both prediction generation and the diagnostic entry point build these arrays.

Run the following commands from the project root.

Install the project dependencies:

```bash
uv sync --locked
```

Inspect the input summaries:

```bash
uv run python k1000_scripts/inspect_inputs.py
```

Run the numerical diagnostics:

```bash
uv run python k1000_scripts/checkpoints.py
```

Generate and export predictions:

```bash
uv run python k1000_scripts/build_predictions.py
```

The output directory is the project's `results/` folder. The exporter creates it as needed and writes `K1000_predictions.h5ad`, replacing an existing file with the same name.

## Verification results

Numerical verification on 2026-10-25 compared the core calculations against saved reference arrays and tested count generation and export.

| Verification | Result |
|---|---|
| Input shapes, gene order, perturbation identities, and count conditions | Passed for all five sources |
| Source `r`, `SE`, and `z` arrays | Matched references with `rtol=0`, `atol=1e-12` |
| Pooled responses, standard errors, z-scores, and source counts | Matched references with `rtol=0`, `atol=1e-12` |
| A/B/C LFC arrays | Matched references with `rtol=0`, `atol=1e-12`; identical nonzero masks |
| Stable ranking, original gene indices, target-gene handling, and stochastic rounding | Passed targeted checks |
| Real-data export and readback | Counts and labels verified for 1,200 cells and 18,533 genes |
| Synthetic integration test | All 900 prediction blocks verified across 360,000 cells and 300 synthetic genes |

The A/B/C LFC matrices contain 300,299, 300,295, and 300,293 nonzero entries, respectively, reflecting 1,000 selected response genes per perturbation plus eligible target genes.

## Output format

The full-panel output is a gzip-compressed AnnData `.h5ad` file:

| Component | Contents |
|---|---|
| `X` | CSR integer count matrix, shape `(360000, 18533)` |
| `obs["context"]` | Context label A, B, or C |
| `obs["target_gene"]` | Perturbation target for each predicted cell |
| `obs_names` | Unique cell identifiers of the form `context_i_c` |
| `var_names` | Gene identifiers aligned with matrix columns |

Rows follow context order A/B/C, then `panel__perts` order, then cell order within each perturbation. The matrix stores predicted counts without CPM or log normalization.

For programmatic use, `build_predictions()` returns the source responses, pooled statistics, domains, selected indices, LFC matrices, simulated controls, and final predictions. In `final_predictions[context][i]`, `i` refers to `panel_perts[i]`, and each count array has shape `(400, 18533)`.

## Core project files

```text
.
├── k1000_scripts/
│   ├── K1000_tools.py        # Statistical estimation, selection, rounding, and export
│   ├── build_predictions.py # Prediction pipeline and H5AD export entry point
│   ├── checkpoints.py       # Numerical diagnostics and consistency checks
│   └── inspect_inputs.py   # Source-level input summaries
├── pyproject.toml           # Python requirement and dependencies
├── uv.lock                  # Dependency lock file
├── .gitignore
├── README.md
├── k1000_inputs.npz         # Preprocessed analysis input, stored separately from Git
└── results/                # Generated prediction files
```

The repository tracks code, documentation, and dependency specifications. Input archives, generated outputs, virtual environments, Python caches, and `.env` are excluded by `.gitignore`.
