"""F02: the F01 statistical comparator with chained imputation removed.

The unchanged forest_main module supplies encoding, counts, direct forests and
the original respondent split. This entry point never selects target neighbors,
fits conditional forests, samples target answers or performs imputation sweeps.
"""
import time

_STARTED = time.monotonic()  # Include helper/dependency imports in the shared clock.

import numpy as np
from . import forest_main as base


CONFIG = dict(base.CONFIG)
MIXTURES = base.MIXTURES[base.MIXTURES[:, 3] == 0].copy()
LAST_DIAGNOSTICS = {}


def _select_mixture(priors, counts, direct, labels, config):
    """Score only the seven original F01 candidates with zero chained weight."""
    scores = np.zeros(len(MIXTURES), dtype=float)
    for j in range(labels.shape[1]):
        valid = labels[:, j] >= 0
        if not np.any(valid):
            continue
        y = labels[valid, j]
        components = np.column_stack((priors[j][y], counts[j][valid, y],
                                      direct[j][valid, y], direct[j][valid, y]))
        probability = components @ MIXTURES.T
        probability = config["scoring_floor"] + (1 - len(priors[j]) * config["scoring_floor"]) * probability
        scores += np.log(probability).sum(axis=0)
    return MIXTURES[int(np.argmax(scores))].copy()


def predict(frame, schema):
    """Return blank-cell probabilities in frame-row then schema-target order."""
    global LAST_DIAGNOSTICS
    started = time.monotonic()
    cfg = CONFIG
    LAST_DIAGNOSTICS = {"method": "direct_probabilistic_forest", "budget_backoff": False,
                        "validation_ran": False, "forest_refit_ran": False, "chained_refit_ran": False}
    given, targets, options, x, y = base._encode(frame, schema)
    missing = y < 0
    query_rows = np.flatnonzero(np.any(missing, axis=1))
    if not len(query_rows):
        return []
    widths_x, widths_y = [len(options[n]) for n in given], [len(options[n]) for n in targets]
    visible = np.flatnonzero(np.any(y >= 0, axis=1))
    priors, counts = base._count_predictions(x[visible], y[visible], x[query_rows], widths_x, widths_y, given, cfg)
    probabilities = counts
    fitting, validation = base._split_visible(x, y, given, cfg)
    deadline = min(_STARTED + cfg["shared_work_budget_seconds"], started + 40 + 1.65 * len(targets))
    try:
        base._check_budget(deadline)
        if len(validation):
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                vpriors, vcounts = base._count_predictions(x[fitting], y[fitting], x[validation], widths_x, widths_y, given, cfg)
                models = base._fit_forests(x[fitting], y[fitting], targets, schema["items"], widths_x, widths_y, cfg, deadline, use_chains=False)
                # Target values, including routing parents, cannot enter inference.
                hidden_validation = np.full((len(validation), len(targets)), -1, dtype=np.int32)
                direct, _ = base._forest_predictions(models, x[validation], hidden_validation, vpriors, widths_x, widths_y, cfg, deadline, use_chains=False)
                weights = _select_mixture(vpriors, vcounts, direct, y[validation], cfg)
                LAST_DIAGNOSTICS["validation_ran"] = True
                del models, direct
                direct = None
                if weights[2]:
                    models = base._fit_forests(x[visible], y[visible], targets, schema["items"], widths_x, widths_y, cfg, deadline, use_chains=False)
                    hidden_query = np.full((len(query_rows), len(targets)), -1, dtype=np.int32)
                    direct, _ = base._forest_predictions(models, x[query_rows], hidden_query, priors, widths_x, widths_y, cfg, deadline, use_chains=False)
                    LAST_DIAGNOSTICS["forest_refit_ran"] = True
                probabilities = base._combine(priors, counts, direct, None, weights)
    except base.WorkBudgetReached:
        LAST_DIAGNOSTICS["budget_backoff"] = True
        probabilities = counts
    output = []
    for position, row in enumerate(query_rows):
        for j in range(len(targets)):
            if missing[row, j]:
                output.append(probabilities[j][position].tolist())
    LAST_DIAGNOSTICS["elapsed_seconds"] = time.monotonic() - started
    return output
