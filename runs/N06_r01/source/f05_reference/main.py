"""Compare a statistical challenger with frozen F02 on disjoint respondents."""
import time

_STARTED = time.monotonic()
from . import direct_forest_main as f02

_LAST_F02_FINISHED = time.monotonic()
import json
from pathlib import Path

import numpy as np
from . import candidate  # The builder packages precisely one provider under this name.

CONFIG = json.loads(Path(__file__).with_name('challenger_config.json').read_text(encoding='utf-8'))
LAST_DIAGNOSTICS = {}
_CANDIDATE_SECONDS = 0.0


def _baseline(frame, schema):
    """Charge F02's actual work and impose the wrapper's absolute wall deadline."""
    global _LAST_F02_FINISHED
    work_start = f02._STARTED + max(0.0, time.monotonic() - _LAST_F02_FINISHED)
    # Frozen F02 checks _STARTED + its own shared limit inside each forest fit.
    # An earlier effective start can only shorten this call's available budget.
    wall_deadline = _STARTED + CONFIG['overall_work_budget_seconds']
    f02._STARTED = min(work_start, wall_deadline - f02.CONFIG['shared_work_budget_seconds'])
    try:
        output = f02.predict(frame, schema)
        return output, dict(f02.LAST_DIAGNOSTICS)
    finally:
        f02._STARTED = work_start
        _LAST_F02_FINISHED = time.monotonic()


def _candidate_deadline():
    return min(_STARTED + CONFIG['overall_work_budget_seconds'],
               time.monotonic() + CONFIG['candidate_work_budget_seconds'] - _CANDIDATE_SECONDS)


def _candidate(x_fit, y_fit, x_query, given, targets, widths_x, widths_y):
    global _CANDIDATE_SECONDS
    started = time.monotonic()
    deadline = _candidate_deadline()
    try:
        f02.base._check_budget(deadline)
        raw = candidate.fit_predict(x_fit, y_fit, x_query, given, targets, widths_x, widths_y, CONFIG, deadline)
        if len(raw) != len(widths_y):
            raise ValueError('Candidate must return one probability matrix per target')
        matrices = []
        for values, width in zip(raw, widths_y):
            values = np.asarray(values, dtype=float)
            if (values.shape != (len(x_query), width) or not np.isfinite(values).all()
                    or (values < 0).any() or not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
                raise ValueError('Invalid candidate support, probabilities or row sums')
            matrices.append(values)
        f02.base._check_budget(deadline)
        return matrices
    finally:
        _CANDIDATE_SECONDS += time.monotonic() - started


def _select_weight(baseline, challenger, labels, weights, floor):
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 1 or not len(weights) or weights[0] != 0 or not np.isfinite(weights).all()
            or (weights > 1).any() or (np.diff(weights) <= 0).any()):
        raise ValueError('Candidate weights must start at zero and increase to at most one')
    scores = np.zeros(len(weights))
    for j, (reference, alternate) in enumerate(zip(baseline, challenger)):
        rows = np.flatnonzero(labels[:, j] >= 0)
        if not len(rows):
            continue
        classes = labels[rows, j]
        p = reference[rows, classes, None] * (1 - weights) + alternate[rows, classes, None] * weights
        scores += np.log(floor + (1 - reference.shape[1] * floor) * p).sum(axis=0)
    # Floating-point summation should not turn an exact tie into a positive contribution.
    return float(weights[np.flatnonzero(scores >= scores.max() - 1e-10)[0]])


def _selection_fitting_rows(fitting, x, given, config):
    """Subsample only outer FIT using GIVEN strata and a separate random stream."""
    cap = config.get('max_selection_fit_rows')
    if cap is None:
        return fitting
    if type(cap) is not int or cap < config['min_fitting_rows']:
        raise ValueError('max_selection_fit_rows must be an integer at least min_fitting_rows')
    if len(fitting) <= cap:
        return fitting
    columns = f02.base._context_columns(given)
    groups = {}
    for row in fitting:
        groups.setdefault(tuple(x[row, columns]), []).append(row)
    strata = [groups[key] for key in sorted(groups)]
    sizes = np.array([len(rows) for rows in strata], dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], 4002]))
    # Allocate proportionally with largest remainders; break equal remainders
    # independently of labels. One stratum gives ordinary uniform sampling.
    quotas, remainders = np.divmod(sizes * cap, len(fitting))
    order = np.lexsort((rng.random(len(strata)), -remainders))
    quotas[order[:cap - int(quotas.sum())]] += 1
    draws = [rng.choice(rows, size=int(n), replace=False) for rows, n in zip(strata, quotas) if n]
    return np.sort(np.concatenate(draws))


def _validation_baseline(frame, schema, targets, fitting, validation, labels):
    indices = np.concatenate((fitting, validation))
    masked = frame.iloc[indices].copy().reset_index(drop=True)
    masked.loc[len(fitting):, targets] = np.nan
    predictions, state = _baseline(masked, schema)
    # F02 returns all blank cells, including any originally blank FIT cells.
    skip = int((labels[fitting] < 0).sum())
    if len(predictions) != skip + len(validation) * len(targets):
        raise ValueError('Unexpected number of F02 validation predictions')
    by_item = [np.asarray(predictions[skip + j::len(targets)], dtype=float) for j in range(len(targets))]
    return by_item, state


def _predict(frame, schema, diagnostics):
    # These final predictions are never consulted by the validation selector.
    reference, final_state = _baseline(frame, schema)
    diagnostics['final_statistical_status'] = final_state
    if not reference:
        diagnostics['fallback_reason'] = 'no_query_cells'
        return reference
    if final_state['budget_backoff']:
        diagnostics.update(budget_backoff=True, fallback_reason='final_F02_budget')
        return reference

    given, targets, options, x, y = f02.base._encode(frame, schema)
    widths_x = [len(options[k]) for k in given]
    widths_y = [len(options[k]) for k in targets]
    fitting, validation = f02.base._split_visible(x, y, given, CONFIG)
    diagnostics.update(fitting_respondents=len(fitting), validation_respondents=len(validation))
    if not len(validation):
        diagnostics['fallback_reason'] = 'insufficient_visible_respondents'
        return reference

    diagnostics['available_fitting_respondents'] = len(fitting)
    fitting = _selection_fitting_rows(fitting, x, given, CONFIG)
    diagnostics['fitting_respondents'] = len(fitting)

    stage = 'candidate_validation'
    try:
        alternate = _candidate(x[fitting], y[fitting], x[validation], given, targets, widths_x, widths_y)
        diagnostics['candidate_validation_ran'] = True
        stage = 'F02_validation'
        comparison, validation_state = _validation_baseline(frame, schema, targets, fitting, validation, y)
        diagnostics['validation_statistical_status'] = validation_state
        if validation_state['budget_backoff'] or not validation_state['validation_ran']:
            diagnostics.update(budget_backoff=validation_state['budget_backoff'], fallback_reason='incomplete_F02_validation')
            return reference
        f02.base._check_budget(_STARTED + CONFIG['overall_work_budget_seconds'])
        weight = _select_weight(comparison, alternate, y[validation], CONFIG['weights'], f02.CONFIG['scoring_floor'])
        diagnostics.update(blend_calibrated=True, validation_ran=True, selected_contribution=weight,
                           validation_scored_cells=int((y[validation] >= 0).sum()))
        if weight == 0:
            diagnostics['fallback_reason'] = 'zero_selected'
            return reference
        stage = 'candidate_refit'
        visible = np.flatnonzero(np.any(y >= 0, axis=1))
        query = np.flatnonzero(np.any(y < 0, axis=1))
        alternate = _candidate(x[visible], y[visible], x[query], given, targets, widths_x, widths_y)
        diagnostics['candidate_refit_ran'] = True
        output = []
        for pos, row in enumerate(query):
            for j in np.flatnonzero(y[row] < 0):
                base_p = np.asarray(reference[len(output)], dtype=float)
                p = (1 - weight) * base_p + weight * alternate[j][pos]
                output.append((p / p.sum()).tolist())
        if len(output) != len(reference):
            raise ValueError('Unexpected number of blended query predictions')
        diagnostics['applied_contribution'] = weight
        return output
    except f02.base.WorkBudgetReached:
        diagnostics.update(budget_backoff=True, fallback_reason=stage + '_budget')
        return reference


def predict(frame, schema):
    """Return the competition's ordered probabilities for every blank PREDICT cell."""
    global LAST_DIAGNOSTICS
    started = time.monotonic()
    LAST_DIAGNOSTICS = {'method': 'statistical_challenger', 'provider': CONFIG['provider'],
                        'budget_backoff': False, 'candidate_validation_ran': False,
                        'validation_ran': False, 'blend_calibrated': False, 'candidate_refit_ran': False,
                        'selected_contribution': None, 'applied_contribution': 0.0, 'fallback_reason': None}
    try:
        return _predict(frame, schema, LAST_DIAGNOSTICS)
    finally:
        LAST_DIAGNOSTICS.update(elapsed_seconds=time.monotonic() - started,
                                cumulative_candidate_seconds=_CANDIDATE_SECONDS)
