import numpy as np
from K1000_tools import pool_response
from build_predictions import build_predictions, sources
from inspect_inputs import summarize_source



def main():
    """

    Call build_predictions() once for responses, LFCs, controls, and predictions.
    Keep the source summaries in the returned results.
    Calculate and print the Step 2 diagnostic summaries.
    Calculate, verify, and print the handbook Step 3, 5, 6, and 7 checkpoints.
    Verify and print the Step 4 domain checks.

    Returns:
        results, pooled, step2_checkpoints, step3_checkpoints,
        step5_checkpoints, step6_checkpoints, step7_checkpoints. Step 6
        summaries are keyed by context A/B/C. Step 7 contains context A's
        ABCD1 group sizes, realised LFCs, and unchanged-gene checks.

    """

    built = build_predictions()
    panel_perts = built["panel_perts"]
    genes = built["genes"]
    source_perts = built["source_perts"]

    results = {}
    step2_checkpoints = {
        "response_count_percentiles": {},
        "on_target_fraction": {},
        "se_calibration": {},
    }
    step3_checkpoints = {}

    for source in sources:
        results[source] = summarize_source(source)[source]
        results[source].update(built["source_responses"][source])

        # Step 2: response_count_percentiles
        # Count observed genes with |z| > 3 for each perturbation, then
        # summarize those counts as min / Q25 / median / Q75 / max.
        z = results[source]["z"]
        abs_z = np.abs(z)
        observed = built["source_observed"][source]
        above_threshold = abs_z > 3
        checkpoint1_mask = above_threshold & observed
        checkpoint1_counts = np.count_nonzero(checkpoint1_mask, axis = 1)
        percentiles = np.percentile(checkpoint1_counts, [0, 25, 50, 75, 100])

        step2_checkpoints["response_count_percentiles"][source] = percentiles

        # Step 2: on_target_fraction
        # Match each perturbation to its own target gene by name.
        # Among observed targets, calculate the fraction with z < -3.
        perts = source_perts[source]

        matches = perts[:, None] == genes[None, :]
        perts_rows, targets_cols = np.nonzero(matches)

        np.testing.assert_array_equal(perts_rows, np.arange(len(perts)))

        target_z = z[perts_rows, targets_cols]
        target_observed = observed[targets_cols]
        observed_target_z = target_z[target_observed]
        fraction = np.mean(observed_target_z < -3)

        step2_checkpoints["on_target_fraction"][source] = fraction

        # Step 2: se_calibration
        # Compare robust SD of z using full SE versus X-only SE,
        # restricting both to observed genes with matched control C >= 20.
        z_full = z
        r = results[source]["r"]
        C = results[source]["C"]
        X = results[source]["X"]
        se_X = np.sqrt(1 / (X + 0.5))
        z_X = r / se_X

        calibration_mask = observed & (C >= 20)
        selected_z_full = z_full[calibration_mask]
        selected_z_X = z_X[calibration_mask]

        median_z_full = np.median(selected_z_full)
        median_z_X = np.median(selected_z_X)
        sd_full = 1.4826 * np.median(np.abs(selected_z_full - median_z_full))
        sd_X = 1.4826 * np.median(np.abs(selected_z_X - median_z_X))

        step2_checkpoints["se_calibration"][source] = (sd_full, sd_X)

    # Step 3: check the five-source result already built.
    pooled = built["pooled"]

    # Step 3: ABCD1 at its own target gene, using full-precision data.
    target_row = np.flatnonzero(panel_perts == "ABCD1").item()
    target_col = np.flatnonzero(genes == "ABCD1").item()
    target = {
        key: pooled[key][target_row, target_col]
        for key in ("r_pool", "se_pool", "z_pool", "n_src")
    }
    # Tolerances reflect the handbook's displayed decimal precision.
    np.testing.assert_allclose(target["r_pool"], -0.9658, rtol=0, atol=0.00005)
    np.testing.assert_allclose(target["se_pool"], 0.1803, rtol=0, atol=0.00005)
    np.testing.assert_allclose(target["z_pool"], -5.358, rtol=0, atol=0.0005)
    np.testing.assert_equal(target["n_src"], 4)

    # Repeat the handbook's hand calculation with its rounded four-source table.
    table_r = np.array([-1.4986, -1.0887, -0.7093, -0.2494])
    table_se = np.array([0.9031, 0.2166, 0.4190, 0.6288])
    table_weight = 1 / table_se**2
    hand_r = np.sum(table_weight * table_r) / table_weight.sum()
    hand_se = 1 / np.sqrt(table_weight.sum())
    hand_z = hand_r / hand_se
    np.testing.assert_allclose(hand_r, -0.9658, rtol=0, atol=0.00005)
    np.testing.assert_allclose(hand_se, 0.1803, rtol=0, atol=0.00005)
    np.testing.assert_allclose(hand_z, -5.357, rtol=0, atol=0.0005)
    target["rounded_table"] = (hand_r, hand_se, hand_z)
    target["hek293t_weight_share"] = table_weight[1] / table_weight.sum()
    step3_checkpoints["abcd1_target"] = target

    # Step 3: per-perturbation counts of genes with |pooled z| > 3.
    response_counts = np.count_nonzero(np.abs(pooled["z_pool"]) > 3, axis=1)
    percentiles = np.percentile(response_counts, [0, 25, 50, 75, 100])
    np.testing.assert_array_equal(percentiles, [15, 43, 64, 109, 781])
    step3_checkpoints["response_count_percentiles"] = percentiles

    # Step 3: coverage across all perturbation-gene positions.
    source_counts = np.bincount(pooled["n_src"].ravel(), minlength=6)
    np.testing.assert_array_equal(
        source_counts, [88108, 39992, 46, 281086, 3182728, 1967940]
    )
    step3_checkpoints["source_count_distribution"] = source_counts

    # Step 3: each source's share of the total inverse-variance weight.
    source_weights = {
        source: np.sum(1 / results[source]["se"]**2) for source in sources
    }
    total_weight = sum(source_weights.values())
    weight_shares = {
        source: weight / total_weight for source, weight in source_weights.items()
    }
    np.testing.assert_allclose(
        [weight_shares[source] for source in sources],
        [0.247, 0.415, 0.084, 0.145, 0.109], rtol=0, atol=0.0005,
    )
    step3_checkpoints["source_weight_shares"] = weight_shares

    # Step 3: duplicate HCT116 without changing the five-source pooled result.
    # Use HCT116's own row order for a direct comparison with its input arrays.
    duplicate_perts = {
        "HCT116": source_perts["HCT116"],
        "HCT116_copy": source_perts["HCT116"],
    }
    duplicate_responses = {
        "HCT116": results["HCT116"], "HCT116_copy": results["HCT116"]
    }
    duplicate_pooled = pool_response(
        source_perts["HCT116"], duplicate_perts, duplicate_responses
    )
    original_response = results["HCT116"]
    np.testing.assert_allclose(
        duplicate_pooled["r_pool"], original_response["r"], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        duplicate_pooled["se_pool"], original_response["se"] / np.sqrt(2),
        rtol=0, atol=1e-12,
    )
    np.testing.assert_allclose(
        duplicate_pooled["z_pool"], original_response["z"] * np.sqrt(2),
        rtol=0, atol=1e-12,
    )
    np.testing.assert_array_equal(
        duplicate_pooled["n_src"], 2 * np.isfinite(original_response["se"])
    )
    step3_checkpoints["duplicate_source_check"] = {"passed": True}
    del duplicate_pooled

    # Step 4: validate the target-context domains returned by the builder.
    # Each Boolean mask represents genes with target control CPM > 5.
    # Check its shape against the shared gene axis, its Boolean dtype,
    # and its eligible-gene count against the handbook for A, B, and C.
    # Excluding each perturbation's own target gene belongs to Step 5.
    domains = built["domains"]
    expected_counts_domain = {"A": 9929, "B": 9626, "C": 10124}

    for context, mask in domains.items():
        assert mask.shape == genes.shape
        assert mask.dtype == np.bool_
        assert np.count_nonzero(mask) == expected_counts_domain[context]

    # Step 5: check the builder's selections without calling the selector again.
    # Convert each row-index dictionary to an array for validation only:
    # axis 0 is panel_perts order; axis 1 is rank among the 1,000 picks.
    selected_indices = built["selected_indices"]
    assert set(selected_indices) == {"A", "B", "C"}
    selected_arrays = {}
    for context in ["A", "B", "C"]:
        selected = selected_indices[context]
        assert set(selected) == set(range(len(panel_perts)))
        indices = np.asarray([selected[p] for p in range(len(panel_perts))])
        assert indices.shape == (len(panel_perts), 1000)
        assert np.issubdtype(indices.dtype, np.integer)
        assert np.all((indices >= 0) & (indices < len(genes)))
        for p, pert in enumerate(panel_perts):
            assert len(np.unique(indices[p])) == 1000
            assert np.all(domains[context][indices[p]])
            assert np.all(genes[indices[p]] != pert)
            # Independently verify both top-1,000 membership and stable order.
            candidates = np.flatnonzero(domains[context] & (genes != pert))
            order = np.argsort(-np.abs(pooled["z_pool"][p, candidates]), kind="stable")
            np.testing.assert_array_equal(indices[p], candidates[order[:1000]])
        selected_arrays[context] = indices

    # A: ABCD1's first three picks, including their signed responses and z-scores.
    first_three = selected_arrays["A"][target_row, :3]
    first_genes = genes[first_three]
    first_r = pooled["r_pool"][target_row, first_three]
    first_z = pooled["z_pool"][target_row, first_three]
    np.testing.assert_array_equal(first_genes, ["PCDH9", "MT-ATP6", "VIM"])
    np.testing.assert_allclose(first_r, [0.2934, -0.0488, 0.1321], rtol=0, atol=0.00005)
    np.testing.assert_allclose(first_z, [7.42, -5.89, 5.43], rtol=0, atol=0.005)

    # Gather each row's own picks, preserving the (perturbation, rank) axes.
    selected_abs_z = np.abs(np.take_along_axis(
        pooled["z_pool"], selected_arrays["A"], axis=1
    ))
    rank_1000_percentiles = np.percentile(selected_abs_z[:, -1], [0, 25, 50, 75, 100])
    np.testing.assert_allclose(
        rank_1000_percentiles, [1.19, 1.32, 1.38, 1.47, 2.29], rtol=0, atol=0.005
    )
    median_above_3 = np.median(np.count_nonzero(selected_abs_z > 3, axis=1))
    np.testing.assert_equal(median_above_3, 35)
    overlap_counts = [
        np.intersect1d(selected_arrays["A"][p], selected_arrays["B"][p]).size
        for p in range(len(panel_perts))
    ]
    median_overlap = np.median(overlap_counts)
    np.testing.assert_equal(median_overlap, 861)
    step5_checkpoints = {
        "abcd1_first_three_genes": first_genes,
        "abcd1_first_three_r": first_r,
        "abcd1_first_three_z": first_z,
        "rank_1000_abs_z_percentiles": rank_1000_percentiles,
        "median_picks_above_3": median_above_3,
        "median_overlap_A_B": median_overlap,
    }

    # Step 6: check the builder's LFCs without recomputing them.
    # Response statistics use only the 300 x 1,000 Step 5 positions.
    # Target statistics use only targets inside the current domain.
    built_lfc = built["built_lfc"]
    assert set(built_lfc) == {"A", "B", "C"}
    panel_rows, target_cols = np.nonzero(panel_perts[:, None] == genes[None, :])
    np.testing.assert_array_equal(panel_rows, np.arange(len(panel_perts)))
    expected_step6 = {
        # Expected values are ordered A, B, C, as in the handbook table.
        "nonzero_entries": [300299, 300295, 300293],
        "rows_with_1001_nonzeros": [299, 295, 293],
        "response_floor_fraction": [0.9339, 0.9360, 0.9302],
        "response_cap_count": [0, 0, 0],
        "max_abs_response_lfc": [1.7478, 1.7478, 1.7478],
        "response_up_fraction": [0.4845, 0.4844, 0.4860],
        "median_target_lfc": [-0.8815, -0.8818, -0.8815],
        "lfc_sum": [-2793.258, -2692.535, -2516.288],
    }
    step6_checkpoints = {}
    for context_index, context in enumerate(["A", "B", "C"]):
        lfc = built_lfc[context]
        assert lfc.shape == (len(panel_perts), len(genes))
        assert np.issubdtype(lfc.dtype, np.floating)
        assert np.isfinite(lfc).all()
        indices = selected_arrays[context]
        response_lfc = np.take_along_axis(lfc, indices, axis=1)
        abs_response_lfc = np.abs(response_lfc)
        target_in_domain = domains[context][target_cols]
        target_lfc = lfc[panel_rows, target_cols]

        # Only response picks and in-domain targets may have nonzero LFC.
        called = np.zeros(lfc.shape, dtype=bool)
        called[panel_rows[:, None], indices] = True
        called[panel_rows[target_in_domain], target_cols[target_in_domain]] = True
        assert np.all(lfc[~called] == 0)
        row_nonzero_counts = np.count_nonzero(lfc, axis=1)
        np.testing.assert_array_equal(
            row_nonzero_counts, 1000 + target_in_domain.astype(int)
        )
        np.testing.assert_allclose(
            target_lfc[target_in_domain],
            np.minimum(pooled["r_pool"][panel_rows, target_cols][target_in_domain], -0.5),
            rtol=0, atol=1e-12,
        )

        summary = {
            "nonzero_entries": int(np.count_nonzero(lfc)),
            "rows_with_1001_nonzeros": int(np.count_nonzero(row_nonzero_counts == 1001)),
            "response_floor_fraction": np.mean(abs_response_lfc == 0.2),
            "response_cap_count": int(np.count_nonzero(abs_response_lfc == 2.0)),
            "max_abs_response_lfc": np.max(abs_response_lfc),
            "response_up_fraction": np.mean(response_lfc > 0),
            "median_target_lfc": np.median(target_lfc[target_in_domain]),
            "lfc_sum": lfc.sum(),
        }
        for key in ("nonzero_entries", "rows_with_1001_nonzeros", "response_cap_count"):
            np.testing.assert_equal(summary[key], expected_step6[key][context_index])
        # Rounded fractions and four-decimal values allow half a last digit.
        # For example, 93.39% is 0.9339, with tolerance 0.00005.
        for key in (
            "response_floor_fraction", "max_abs_response_lfc",
            "response_up_fraction", "median_target_lfc",
        ):
            np.testing.assert_allclose(
                summary[key], expected_step6[key][context_index], rtol=0, atol=0.00005,
                err_msg=f"Step 6, {context}: {key}",
            )
        np.testing.assert_allclose(
            summary["lfc_sum"], expected_step6["lfc_sum"][context_index],
            rtol=0, atol=0.0005,
        )
        step6_checkpoints[context] = summary

    # A: illustrate why many weak pooled responses end up at the 0.2 floor.
    mt_atp6_col = np.flatnonzero(genes == "MT-ATP6").item()
    assert mt_atp6_col in selected_arrays["A"][target_row]
    mt_atp6_r = pooled["r_pool"][target_row, mt_atp6_col]
    mt_atp6_lfc = built_lfc["A"][target_row, mt_atp6_col]
    median_abs_response_r = np.median(np.abs(np.take_along_axis(
        pooled["r_pool"], selected_arrays["A"], axis=1
    )))
    np.testing.assert_allclose(mt_atp6_r, -0.0488, rtol=0, atol=0.00005)
    np.testing.assert_equal(mt_atp6_lfc, -0.2)
    np.testing.assert_allclose(median_abs_response_r, 0.1034, rtol=0, atol=0.00005)
    step6_checkpoints["A"].update({
        "abcd1_mt_atp6_r": mt_atp6_r,
        "abcd1_mt_atp6_lfc": mt_atp6_lfc,
        "median_abs_response_r": median_abs_response_r,
    })

    # Step 7: compare rounding on the SAME context A controls and ABCD1 LFC.
    # Reconstruct fractional counts without drawing new cells or rerounding
    # the builder's stochastic prediction, so the seeded experiment is kept.
    np.testing.assert_equal(panel_perts[0], "ABCD1")
    cells = built["control_cells"]["A"][target_row]
    lfc_one = built_lfc["A"][target_row]
    prediction = cells * np.exp(lfc_one)
    assert cells.shape == (400, len(genes))
    assert np.all(cells >= 0)
    up = lfc_one == 0.2
    down = lfc_one == -0.2
    unchanged = lfc_one == 0
    group_counts = (int(up.sum()), int(down.sum()))
    np.testing.assert_array_equal(group_counts, [438, 514])
    expected_step7 = {
        "exact": (0.200, -0.200),
        "round": (0.156, -0.121),
        "floor": (0.072, -0.770),
        "stochastic": (0.200, -0.202),
    }
    step7_checkpoints = {
        "group_counts": {"up": group_counts[0], "down": group_counts[1]},
        "realised_lfc": {},
        "unchanged_genes": {},
    }
    for method, expected in expected_step7.items():
        if method == "exact":
            rounded = prediction
        elif method == "round":
            rounded = np.round(prediction)
        elif method == "floor":
            rounded = np.floor(prediction)
        else:
            rounded = built["final_predictions"]["A"][target_row]
            assert rounded.dtype == np.int64
            assert np.all(
                (rounded == np.floor(prediction)) | (rounded == np.ceil(prediction))
            )
        assert rounded.shape == cells.shape
        assert np.isfinite(rounded).all() and np.all(rounded >= 0)
        if method != "exact":
            assert np.all(rounded == np.floor(rounded))
        unchanged_equal = np.array_equal(rounded[:, unchanged], cells[:, unchanged])
        assert unchanged_equal, f"Step 7, {method}: zero-LFC genes changed"
        # Sum over BOTH axes: one ratio of totals for each gene group,
        # not a mean of per-cell or per-gene log ratios.
        realised_up = np.log(rounded[:, up].sum() / cells[:, up].sum())
        realised_down = np.log(rounded[:, down].sum() / cells[:, down].sum())
        # Exact scaling is algebraic; the handbook's other values are approximate.
        tolerance = 1e-12 if method == "exact" else 0.001
        np.testing.assert_allclose(
            [realised_up, realised_down], expected, rtol=0, atol=tolerance,
            err_msg=f"Step 7, {method}: realised LFC",
        )
        step7_checkpoints["realised_lfc"][method] = {
            "up": realised_up, "down": realised_down,
        }
        step7_checkpoints["unchanged_genes"][method] = bool(unchanged_equal)

    print("Step 2: response-count percentiles")
    for source, percentiles in step2_checkpoints["response_count_percentiles"].items():
        print(source, ":", percentiles)

    print("")
    print("Step 2: on-target fraction")
    for source, fraction in step2_checkpoints["on_target_fraction"].items():
        print(f"{source}: {fraction:.3f}")

    print("")
    print("Step 2: SE calibration")
    header = f"{'Source':<18} | {'SD full':>14} | {'SD X':>14}"
    print(header)
    print("-" * len(header))
    for source in sources:
        print(f"{source:<18} | {step2_checkpoints["se_calibration"][source][0]:>14.3f} | {step2_checkpoints["se_calibration"][source][1]:>14.3f}")
    
    print("\nStep 3 checkpoints")
    target = step3_checkpoints["abcd1_target"]
    print(f"ABCD1: r={target['r_pool']:.4f}, SE={target['se_pool']:.4f}, "
          f"z={target['z_pool']:.3f}, n_src={target['n_src']}")
    hand_r, hand_se, hand_z = target["rounded_table"]
    print(f"Rounded-table calculation: r={hand_r:.4f}, SE={hand_se:.4f}, "
          f"z={hand_z:.3f}; HEK293T weight={target['hek293t_weight_share']:.1%}")
    print("Response-count percentiles:", step3_checkpoints["response_count_percentiles"])
    print("Source counts (0 through 5):", step3_checkpoints["source_count_distribution"])
    print("Weight shares:", ", ".join(
        f"{source}={share:.3f}"
        for source, share in step3_checkpoints["source_weight_shares"].items()
    ))
    print("Duplicate HCT116: PASS (r unchanged; SE / sqrt(2); z * sqrt(2))")

    print("\nStep 4 checkpoints")
    print("Domain sizes:", ", ".join(
        f"{domain}={int(mask.sum())}" for domain, mask in domains.items()
    ))
    print("Domain has the same shape with gene:", end = "")
    if (domains["A"].shape == genes.shape) & (domains["B"].shape == genes.shape) & (domains["C"].shape == genes.shape):
        print("True")
    else:
        print("False")

    print("\nStep 5 checkpoints")
    print("A/B/C: PASS (1,000 unique eligible genes per perturbation; stable ranking)")
    print("A, ABCD1 first three picks:")
    for gene, r, z in zip(first_genes, first_r, first_z):
        print(f"  {gene}: r={r:+.4f}, z={z:+.2f}")
    print("A, 1,000th-pick |z| percentiles:", np.round(rank_1000_percentiles, 2))
    print(f"A, median picks with |z| > 3: {median_above_3:.0f}")
    print(f"A/B, median shared picks: {median_overlap:.0f}")

    print("\nStep 6 checkpoints")
    print("A/B/C: PASS (LFC shape, finite values, unchanged positions, target rule)")
    step6_display = [
        ("Non-zero LFC entries", "nonzero_entries", ",d"),
        ("Rows with 1,001 non-zeros", "rows_with_1001_nonzeros", "d"),
        ("Response calls at |LFC| = 0.2", "response_floor_fraction", ".2%"),
        ("Response calls at |LFC| = 2.0", "response_cap_count", "d"),
        ("Largest response |LFC|", "max_abs_response_lfc", ".4f"),
        ("Response up fraction", "response_up_fraction", ".4f"),
        ("Median in-domain target LFC", "median_target_lfc", ".4f"),
        ("Sum of all LFC", "lfc_sum", ".3f"),
    ]
    header = f"{'Checkpoint':<32} | {'A':>12} | {'B':>12} | {'C':>12}"
    print(header)
    print("-" * len(header))
    for label, key, spec in step6_display:
        values = [format(step6_checkpoints[context][key], spec) for context in ["A", "B", "C"]]
        print(f"{label:<32} | " + " | ".join(f"{value:>12}" for value in values))
    print(f"A, ABCD1 / MT-ATP6: pooled r={mt_atp6_r:+.4f}; LFC={mt_atp6_lfc:+.1f}")
    print(f"A, median response |r|={median_abs_response_r:.4f}; "
          f"after x0.7={0.7 * median_abs_response_r:.4f}, below the 0.2 floor")

    print("\nStep 7 checkpoints: context A, ABCD1, 400 simulated cells")
    print(f"Gene groups: +0.2 = {group_counts[0]}; -0.2 = {group_counts[1]}")
    header = f"{'Method':<14} | {'+0.2 realised':>14} | {'-0.2 realised':>14}"
    print(header)
    print("-" * len(header))
    for method, shifts in step7_checkpoints["realised_lfc"].items():
        print(f"{method:<14} | {shifts['up']:>14.3f} | {shifts['down']:>14.3f}")
    print("PASS: zero-LFC genes are exactly unchanged for all four methods.")
    print("Ordinary rounding can erase small count changes, while flooring always "
          "rounds downward and therefore biases counts downward.")
    print("Stochastic rounding adds 1 with probability equal to the fractional part, "
          "so E[floor(x) + Bernoulli(x - floor(x))] = x and counts are preserved "
          "in expectation.")

    return (
        results, pooled, step2_checkpoints, step3_checkpoints,
        step5_checkpoints, step6_checkpoints, step7_checkpoints,
    )



if __name__ == "__main__":
    main()
