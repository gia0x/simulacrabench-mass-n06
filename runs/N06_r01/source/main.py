"""Bounded neural survey experiments with a separately clocked frozen F05."""
import time

_STARTED = time.monotonic()
from f05_reference import main as f05

_LAST_F05_FINISHED = time.monotonic()
import importlib
import json
from pathlib import Path

import numpy as np

CONFIG = json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
if CONFIG['provider'] not in ('neural_candidate', 'tabicl_candidate', 'mlp_candidate', 'semantic_candidate'):
    raise ValueError('Unsupported neural provider')
if CONFIG['combination'] not in ('standalone', 'hybrid'):
    raise ValueError('Unsupported neural combination')
provider = importlib.import_module(CONFIG['provider'])
if hasattr(provider, 'initialize'):
    provider.initialize(CONFIG)
LAST_DIAGNOSTICS = {}
_PROVIDER_SECONDS = 0.0


def _check_budget(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Neural experiment work deadline reached')


def _instrument(given):
    names = {name.lower() for name in given}
    if {'country', 'q13'}.issubset(names):
        return 'unicef'
    if {'group', 'user.country'}.issubset(names):
        return 'world_bank'
    if {'wave', 's1_2_country_asylum_lbl'}.issubset(names):
        return 'unhcr'
    return 'other'


def _baseline(frame, schema, comparison=False):
    """Keep comparison work and external neural work off F05's native clocks.

    Both still count toward the common absolute wall deadline. The reference
    model's active-work counters persist across its three final survey calls.
    An outer selection comparison gets independent counters and restores the
    reference counters in finally, even when it raises.
    """
    global _LAST_F05_FINISHED
    now = time.monotonic()
    wall_deadline = _STARTED + CONFIG['overall_work_budget_seconds']
    _check_budget(wall_deadline)
    snapshot = (f05._STARTED, f05._CANDIDATE_SECONDS,
                f05._LAST_F02_FINISHED, f05.f02._STARTED)
    if comparison:
        work_start = now
        f05._CANDIDATE_SECONDS = 0.0
        f05._LAST_F02_FINISHED = now
        f05.f02._STARTED = now
    else:
        work_start = f05._STARTED + max(0.0, now - _LAST_F05_FINISHED)
    f05._STARTED = min(work_start, wall_deadline - f05.CONFIG['overall_work_budget_seconds'])
    try:
        output = f05.predict(frame, schema)
        # Copy nested diagnostics, which subsequent reference calls replace.
        return output, json.loads(json.dumps(f05.LAST_DIAGNOSTICS))
    finally:
        if comparison:
            (f05._STARTED, f05._CANDIDATE_SECONDS,
             f05._LAST_F02_FINISHED, f05.f02._STARTED) = snapshot
        else:
            f05._STARTED = work_start
            _LAST_F05_FINISHED = time.monotonic()


def _provider(x_fit, y_fit, x_query, given, targets, widths_x, widths_y, schema):
    global _PROVIDER_SECONDS
    started = time.monotonic()
    deadline = min(_STARTED + CONFIG['overall_work_budget_seconds'],
                   started + CONFIG['provider_work_budget_seconds'] - _PROVIDER_SECONDS)
    try:
        _check_budget(deadline)
        cfg = {**CONFIG, 'runtime_schema': schema} if CONFIG['provider'] == 'semantic_candidate' else CONFIG
        raw = provider.fit_predict(x_fit, y_fit, x_query, given, targets,
                                   widths_x, widths_y, cfg, deadline)
        if len(raw) != len(widths_y):
            raise ValueError('Provider must return one probability matrix per target')
        matrices = []
        for values, width in zip(raw, widths_y):
            values = np.asarray(values, dtype=float)
            if (values.shape != (len(x_query), width) or not np.isfinite(values).all()
                    or (values < 0).any()
                    or not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
                raise ValueError('Invalid neural probability shape, support, or row sums')
            matrices.append(values)
        _check_budget(deadline)
        return matrices, dict(getattr(provider, 'LAST_DIAGNOSTICS', {}))
    finally:
        _PROVIDER_SECONDS += time.monotonic() - started


def _marginal_matrices(y_fit, n_query, widths):
    alpha = CONFIG.get('marginal_pseudocount', .5)
    matrices = []
    for j, width in enumerate(widths):
        values = y_fit[:, j]
        counts = np.bincount(values[values >= 0], minlength=width).astype(float) + alpha
        matrices.append(np.broadcast_to(counts / counts.sum(), (n_query, width)))
    return matrices


def _vectors(matrices, labels, query, reference=None, weight=1.0):
    if reference is not None and weight == 0:
        return reference
    output = []
    for pos, row in enumerate(query):
        for j in np.flatnonzero(labels[row] < 0):
            values = matrices[j][pos]
            if reference is not None:
                values = (1 - weight) * np.asarray(reference[len(output)], dtype=float) + weight * values
            output.append((values / values.sum()).tolist())
    if reference is not None and len(output) != len(reference):
        raise ValueError('Reference and neural prediction counts differ')
    return output


def _validation_baseline(frame, schema, targets, fitting, validation, labels):
    indices = np.concatenate((fitting, validation))
    masked = frame.iloc[indices].copy().reset_index(drop=True)
    masked.loc[len(fitting):, targets] = np.nan
    predictions, state = _baseline(masked, schema, comparison=True)
    skip = int((labels[fitting] < 0).sum())
    if len(predictions) != skip + len(validation) * len(targets):
        raise ValueError('Unexpected number of F05 validation predictions')
    matrices = [np.asarray(predictions[skip + j::len(targets)], dtype=float)
                for j in range(len(targets))]
    return matrices, state


def _outer_split(x, y, given, visible, cfg):
    """Choose the held-out people whose observed answers will select the blend."""
    if 'outer_validation_sampling' not in cfg:
        return f05.f02.base._split_visible(x, y, given, cfg)
    if cfg['outer_validation_sampling'] != 'proportional_context':
        raise ValueError('Unsupported outer validation sampling method')
    count = min(int(len(visible) * cfg['validation_fraction']), cfg['validation_cap'],
                len(visible) - cfg['min_fitting_rows'])
    if count < cfg['min_validation_rows']:
        return visible, np.empty(0, dtype=int)
    # Reuse F05's largest-remainder sampler. Its minimum and cap refer here
    # to the sample being drawn (the holdout), and its seed is independent
    # of the subsequent cap on fitting rows. Answers never determine quotas.
    draw = {'seed': cfg['seed'] + 1, 'min_fitting_rows': cfg['min_validation_rows'],
            'max_selection_fit_rows': count}
    held = f05._selection_fitting_rows(visible, x, given, draw)
    return np.setdiff1d(visible, held), held


def _predict(frame, schema, diagnostics):
    given, targets, options, x, y = f05.f02.base._encode(frame, schema)
    widths_x, widths_y = [len(options[k]) for k in given], [len(options[k]) for k in targets]
    query = np.flatnonzero(np.any(y < 0, axis=1))
    visible = np.flatnonzero(np.any(y >= 0, axis=1))
    instrument = _instrument(given)
    diagnostics.update(instrument=instrument, visible_respondents=len(visible),
                       query_respondents=len(query))
    if not len(query):
        diagnostics['fallback_reason'] = 'no_query_cells'
        return []

    marginal = _vectors(_marginal_matrices(y[visible], len(query), widths_y), y, query)
    hybrid = CONFIG['combination'] == 'hybrid'
    reference = None
    stage = 'final_F05'
    try:
        if hybrid:
            reference, state = _baseline(frame, schema)
            diagnostics['final_statistical_status'] = state
            if state.get('budget_backoff', False):
                diagnostics.update(budget_backoff=True, fallback_reason='incomplete_final_F05')
                return reference
        active = CONFIG.get('active_instruments')
        if active is not None and instrument not in active:
            if reference is None:
                raise ValueError('An inactive survey requires a statistical reference')
            diagnostics['fallback_reason'] = 'prespecified_reference_survey'
            return reference

        fitting, validation = _outer_split(x, y, given, visible, CONFIG)
        diagnostics.update(available_fitting_respondents=len(fitting),
                           validation_respondents=len(validation))
        fitting = f05._selection_fitting_rows(fitting, x, given, CONFIG)
        diagnostics['fitting_respondents'] = len(fitting)
        weight = 1.0 if not hybrid else 0.0
        if len(validation):
            stage = 'provider_validation'
            alternate, state = _provider(x[fitting], y[fitting], x[validation],
                                         given, targets, widths_x, widths_y, schema)
            diagnostics.update(provider_validation_ran=True,
                               candidate_validation_ran=True,
                               provider_validation_status=state)
            if hybrid:
                stage = 'F05_validation'
                comparison, state = _validation_baseline(frame, schema, targets,
                                                        fitting, validation, y)
                diagnostics['validation_statistical_status'] = state
                if state.get('budget_backoff', False):
                    diagnostics.update(budget_backoff=True, fallback_reason='incomplete_F05_validation')
                    return reference
                weight = f05._select_weight(comparison, alternate, y[validation],
                                           CONFIG['weights'], CONFIG.get('scoring_floor', .001))
                diagnostics['blend_calibrated'] = True
            diagnostics.update(validation_ran=True, selected_contribution=weight,
                               validation_scored_cells=int((y[validation] >= 0).sum()))
            del alternate
        elif hybrid:
            diagnostics['fallback_reason'] = 'insufficient_blend_respondents'
        else:
            diagnostics['selected_contribution'] = 1.0

        # Both N01 and N02 always run the same final all-visible provider fit,
        # including when N02 selected zero. This makes its method testable.
        stage = 'provider_refit'
        alternate, state = _provider(x[visible], y[visible], x[query],
                                     given, targets, widths_x, widths_y, schema)
        diagnostics.update(provider_refit_ran=True, candidate_refit_ran=True,
                           provider_refit_status=state)
        _check_budget(_STARTED + CONFIG['overall_work_budget_seconds'])
        result = _vectors(alternate, y, query, reference=reference, weight=weight)
        diagnostics['applied_contribution'] = weight
        if hybrid and weight == 0 and len(validation):
            diagnostics['fallback_reason'] = 'zero_selected'
        return result
    except (TimeoutError, f05.f02.base.WorkBudgetReached):
        diagnostics.update(budget_backoff=True, fallback_reason=stage + '_budget')
        return reference if reference is not None else marginal


def predict(frame, schema):
    """Return canonical blank-cell distributions, using GIVEN inputs only."""
    global LAST_DIAGNOSTICS
    started = time.monotonic()
    LAST_DIAGNOSTICS = {
        'method': 'neural_survey', 'provider': CONFIG['provider'],
        'combination': CONFIG['combination'], 'budget_backoff': False,
        'validation_ran': False, 'blend_calibrated': False,
        'provider_validation_ran': False, 'provider_refit_ran': False,
        'candidate_validation_ran': False, 'candidate_refit_ran': False,
        'selected_contribution': None, 'applied_contribution': 0.0,
        'fallback_reason': None,
    }
    try:
        return _predict(frame, schema, LAST_DIAGNOSTICS)
    finally:
        LAST_DIAGNOSTICS.update(elapsed_seconds=time.monotonic() - started,
                                cumulative_provider_seconds=_PROVIDER_SECONDS)
