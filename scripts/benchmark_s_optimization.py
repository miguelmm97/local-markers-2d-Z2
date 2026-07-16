"""Lightweight benchmark for the local-S optimizer.

The timing values printed by this script are intended for performance
inspection, not strict regression thresholds.  They include Python, NumPy, and
SciPy overhead and will vary by machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.s_optimization import generate_tr_invariant_projector, optimize_local_s


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--site-dependent", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_single_benchmark(
    *,
    n_sites: int,
    restarts: int,
    maxiter: int,
    seed: int,
    translation_invariant: bool,
) -> dict[str, Any]:
    """Run one benchmark case and return a machine-readable summary."""

    rho = generate_tr_invariant_projector(n_sites=n_sites, seed=seed + n_sites)

    # The total timing wraps projector-independent optimization only.  The
    # optimizer diagnostics split the warm-start and exact-stage timings.
    start_time = time.perf_counter()
    result = optimize_local_s(
        rho,
        translation_invariant=translation_invariant,
        n_restarts=restarts,
        random_seed=seed,
        maxiter=maxiter,
        tolerance=1e-8,
    )
    total_runtime = time.perf_counter() - start_time

    residuals = result.constraint_residuals.as_dict()
    return {
        "n_sites": n_sites,
        "matrix_dimension": int(rho.shape[0]),
        "translation_invariant": translation_invariant,
        "parameter_count": int(result.parameters.size),
        "number_of_restarts": restarts,
        "warm_start_runtime": float(result.diagnostics.get("warm_start_runtime", 0.0)),
        "exact_runtime": float(result.diagnostics.get("exact_runtime", 0.0)),
        "total_runtime": float(total_runtime),
        "initial_objective": float(result.diagnostics.get("initial_objective", np.nan)),
        "final_objective": float(result.objective_value),
        "initial_gap": float(result.diagnostics.get("initial_gap", np.nan)),
        "final_gap": float(result.spectral_gap),
        "iterations": int(result.n_iterations),
        "function_evaluations": int(result.n_function_evaluations),
        "gradient_evaluations": int(result.n_gradient_evaluations),
        "maximum_constraint_residual": float(max(residuals.values())),
        "success": bool(result.success),
    }


def format_results(results: list[dict[str, Any]]) -> str:
    """Format benchmark results as a readable fixed-width table."""

    columns = [
        ("n_sites", "sites"),
        ("matrix_dimension", "dim"),
        ("translation_invariant", "TI"),
        ("parameter_count", "params"),
        ("number_of_restarts", "starts"),
        ("warm_start_runtime", "warm_s"),
        ("exact_runtime", "exact_s"),
        ("total_runtime", "total_s"),
        ("initial_objective", "obj_i"),
        ("final_objective", "obj_f"),
        ("initial_gap", "gap_i"),
        ("final_gap", "gap_f"),
        ("iterations", "iter"),
        ("function_evaluations", "fev"),
        ("gradient_evaluations", "gev"),
        ("maximum_constraint_residual", "max_res"),
        ("success", "ok"),
    ]
    widths = {header: len(header) for _, header in columns}
    rows: list[dict[str, str]] = []

    for result in results:
        row: dict[str, str] = {}
        for key, header in columns:
            value = result[key]
            if isinstance(value, float):
                row[header] = f"{value:.4g}"
            else:
                row[header] = str(value)
            widths[header] = max(widths[header], len(row[header]))
        rows.append(row)

    header_line = "  ".join(header.rjust(widths[header]) for _, header in columns)
    separator = "  ".join("-" * widths[header] for _, header in columns)
    body = [
        "  ".join(row[header].rjust(widths[header]) for _, header in columns)
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def save_results(results: list[dict[str, Any]], output: Path) -> None:
    """Save benchmark results as JSON."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    """Run all requested benchmark cases."""

    args = parse_arguments()
    translation_invariant = not args.site_dependent
    results = [
        run_single_benchmark(
            n_sites=n_sites,
            restarts=args.restarts,
            maxiter=args.maxiter,
            seed=args.seed,
            translation_invariant=translation_invariant,
        )
        for n_sites in args.sites
    ]

    print(format_results(results))
    if args.output is not None:
        save_results(results, args.output)


if __name__ == "__main__":
    main()

