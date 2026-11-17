"""F01: validation-selected probabilistic forest imputation.

See README.md in the submission for the numbered algorithm and limitations.
Only permitted visible responses fit the model. No survey data is written out.
"""
import time

_STARTED = time.monotonic()  # Shared across the three predict() calls.

import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONFIG = {
    "experiment_id": "F01",
    "family": "probabilistic_forest_imputation",
    "seed": 777,
    "n_estimators": 48,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "min_forest_rows": 64,
    "max_target_neighbors": 3,
    "min_neighbor_information": 0.01,
    "chains": 3,
    "sweeps": 3,
    "burn_in": 1,
    "forest_prior_share": 0.02,
    "validation_fraction": 0.2,
    "validation_cap": 512,
    "min_validation_rows": 32,
    "min_fitting_rows": 128,
    "count_prior_strength": 30.0,
    "marginal_pseudocount": 0.5,
    "scoring_floor": 0.001,
    "shared_work_budget_seconds": 660.0,
}
_CONFIG_PATH = Path(__file__).with_name("config.json")
CONFIG = dict(DEFAULT_CONFIG)
if _CONFIG_PATH.exists():
    CONFIG.update(json.loads(_CONFIG_PATH.read_text(encoding="utf-8")))

# Candidate order resolves exact numerical ties conservatively. Components are
# [whole-instrument marginal, conditional counts, direct forest, chained forest].
MIXTURES = np.asarray([
    [1, 0, 0, 0], [0, 1, 0, 0],
    [0, .75, .25, 0], [0, .5, .5, 0], [0, .25, .75, 0], [0, 0, 1, 0],
    [.25, 0, .75, 0],
    [0, .75, 0, .25], [0, .5, 0, .5], [0, .25, 0, .75], [0, 0, 0, 1],
    [0, 0, .5, .5], [0, .25, .375, .375], [.25, 0, 0, .75],
], dtype=float)
LAST_DIAGNOSTICS = {}  # Technical status only; never labels, fitted weights or scores.


class WorkBudgetReached(Exception):
    """Internal, declared guard; only this exception triggers count backoff."""


def _check_budget(deadline):
    if time.monotonic() >= deadline:
        raise WorkBudgetReached()


def _encode(frame, schema):
    """Step 1: schema-defined supports; IDs and EXCLUDE columns are unused."""
    records = schema["items"]
    given = [n for n, r in records.items() if r["class"] == "GIVEN"]
    targets = [n for n, r in records.items() if r["class"] == "PREDICT"]
    names = given + targets
    options, encoded = {}, {}
    for name in names:
        record = records[name]
        values = list(record["values"])
        if record.get("gate"):
            values.append(schema["gated_value"])
        if not values or len(set(values)) != len(values):
            raise ValueError("Schema categories must be nonempty and distinct")
        options[name] = values
        codes = pd.Categorical(frame[name], categories=values).codes.astype(np.int32)
        missing = frame[name].isna().to_numpy()
        if np.any((codes < 0) & ~missing):
            raise ValueError("An input category is outside the supplied schema")
        if name in given and np.any(missing):
            raise ValueError("GIVEN fields must be observed")
        encoded[name] = codes
    x = np.column_stack([encoded[n] for n in given]) if given else np.empty((len(frame), 0), dtype=np.int32)
    y = np.column_stack([encoded[n] for n in targets]) if targets else np.empty((len(frame), 0), dtype=np.int32)
    return given, targets, options, x, y


def _one_hot(codes, widths):
    """Unordered categories become binary indicators, not ordinal thresholds."""
    n = len(codes)
    output = np.zeros((n, max(1, sum(widths))), dtype=np.float32)
    start = 0
    for column, width in enumerate(widths):
        valid = codes[:, column] >= 0
        output[np.flatnonzero(valid), start + codes[valid, column]] = 1.0
        start += width
    return output


def _context_columns(given):
    lookup = {name.lower(): i for i, name in enumerate(given)}
    for pair in (("wave", "s1_2_country_asylum_lbl"), ("group", "user.country"), ("country", "q13")):
        if all(name in lookup for name in pair):
            return [lookup[name] for name in pair]
    return []


def _split_visible(x, y, given, config):
    """Step 2: whole-respondent holdout, using GIVEN-only stratification."""
    eligible = np.flatnonzero(np.any(y >= 0, axis=1))
    rng = np.random.default_rng(config["seed"])
    lookup = {name.lower(): i for i, name in enumerate(given)}
    strata = []
    if "wave" in lookup and "s1_2_country_asylum_lbl" in lookup:
        strata = [lookup["wave"], lookup["s1_2_country_asylum_lbl"]]
    elif "country" in lookup:
        strata = [lookup["country"]]
    grouped = {}
    for row in eligible:
        key = tuple(x[row, strata]) if strata else ()
        grouped.setdefault(key, []).append(row)
    held = []
    for key in sorted(grouped):
        group = np.asarray(grouped[key], dtype=int)
        rng.shuffle(group)
        # Tiny strata remain available for fitting.
        count = int(len(group) * config["validation_fraction"]) if len(group) >= 5 else 0
        held.extend(group[:min(count, max(0, len(group) - 2))])
    held = np.asarray(held, dtype=int)
    rng.shuffle(held)
    held = np.sort(held[:config["validation_cap"]])
    fitting = np.setdiff1d(eligible, held)
    if len(held) < config["min_validation_rows"] or len(fitting) < config["min_fitting_rows"]:
        return eligible, np.empty(0, dtype=int)
    return fitting, held


def _count_predictions(x_fit, y_fit, x_query, widths_x, widths_y, given, config):
    """Step 3: partial-pooling counts and strictly positive marginal priors."""
    context = set(_context_columns(given))
    features = [(x_fit[:, i], x_query[:, i], width, 3.0 if i in context else 1.0)
                for i, width in enumerate(widths_x)]
    pair = _context_columns(given)
    if pair:
        a, b = pair
        features.append((x_fit[:, a] * widths_x[b] + x_fit[:, b],
                         x_query[:, a] * widths_x[b] + x_query[:, b],
                         widths_x[a] * widths_x[b], 3.0))
    priors, counts = [], []
    for j, width in enumerate(widths_y):
        valid = y_fit[:, j] >= 0
        prior = np.bincount(y_fit[valid, j], minlength=width).astype(float) + config["marginal_pseudocount"]
        prior /= prior.sum()
        average = np.zeros((len(x_query), width), dtype=float)
        weight_sum = 0.0
        for train_group, query_group, groups, weight in features:
            table = np.bincount(train_group[valid] * width + y_fit[valid, j], minlength=groups * width).reshape(groups, width).astype(float)
            table += config["count_prior_strength"] * prior
            table /= table.sum(axis=1, keepdims=True)
            average += weight * table[query_group]
            weight_sum += weight
        p = .25 * prior + .75 * average / weight_sum if weight_sum else np.tile(prior, (len(x_query), 1))
        priors.append(prior)
        counts.append(p)
    return priors, counts


def _neighbors(y_fit, targets, records, widths_y, config, deadline):
    """Step 4: fit-only target associations; at most three target neighbors."""
    neighbors = []
    target_index = {name: j for j, name in enumerate(targets)}
    for j, name in enumerate(targets):
        _check_budget(deadline)
        scores = []
        for k in range(len(targets)):
            if k == j:
                continue
            valid = (y_fit[:, j] >= 0) & (y_fit[:, k] >= 0)
            n = int(valid.sum())
            if n < config["min_forest_rows"]:
                continue
            table = np.bincount(y_fit[valid, j] * widths_y[k] + y_fit[valid, k], minlength=widths_y[j] * widths_y[k]).reshape(widths_y[j], widths_y[k]).astype(float)
            joint = table / n
            a, b = joint.sum(axis=1), joint.sum(axis=0)
            expected = a[:, None] * b[None, :]
            positive = joint > 0
            mutual = np.sum(joint[positive] * np.log(joint[positive] / expected[positive]))
            # Finite-sample correction discourages spurious high-cardinality links.
            mutual = max(0.0, mutual - ((a > 0).sum() - 1) * ((b > 0).sum() - 1) / (2 * n))
            entropy = min(-np.sum(a[a > 0] * np.log(a[a > 0])), -np.sum(b[b > 0] * np.log(b[b > 0])))
            strength = mutual / entropy if entropy > 0 else 0.0
            if strength >= config["min_neighbor_information"]:
                scores.append((-strength, k))
        parent = (records[name].get("gate") or {}).get("parent")
        chosen = [target_index[parent]] if parent in target_index and target_index[parent] != j else []
        for _, k in sorted(scores):
            if k not in chosen and len(chosen) < config["max_target_neighbors"]:
                chosen.append(k)
        neighbors.append(chosen[:config["max_target_neighbors"]])
    return neighbors


def _fit_forest(features, labels, width, config, seed, deadline):
    _check_budget(deadline)
    valid = labels >= 0
    if valid.sum() < config["min_forest_rows"] or len(np.unique(labels[valid])) < 2:
        return None
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"], max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"], max_features="sqrt",
        bootstrap=True, n_jobs=1, random_state=seed, class_weight=None,
    )
    model.fit(features[valid], labels[valid])
    _check_budget(deadline)
    return model


def _forest_probability(model, features, prior, config):
    if model is None:
        return np.tile(prior, (len(features), 1))
    p = np.zeros((len(features), len(prior)), dtype=float)
    p[:, model.classes_.astype(int)] = model.predict_proba(features)
    weight = config["forest_prior_share"]
    return (1 - weight) * p + weight * prior


def _fit_forests(x_fit, y_fit, targets, records, widths_x, widths_y, config, deadline, use_chains=True):
    """Step 5: fit forests once; never refit during imputation sweeps."""
    background = _one_hot(x_fit, widths_x)
    links = _neighbors(y_fit, targets, records, widths_y, config, deadline) if use_chains else [[] for _ in targets]
    direct, conditional = [], []
    for j in range(len(targets)):
        seed = config["seed"] + j * 17
        direct.append(_fit_forest(background, y_fit[:, j], widths_y[j], config, seed, deadline))
        if links[j]:
            auxiliary = _one_hot(y_fit[:, links[j]], [widths_y[k] for k in links[j]])
            features = np.concatenate((background, auxiliary), axis=1)
            conditional.append(_fit_forest(features, y_fit[:, j], widths_y[j], config, seed + 1, deadline))
        else:
            conditional.append(None)
    return direct, conditional, links


def _draw(probabilities, rng):
    uniforms = rng.random(len(probabilities))
    return np.minimum((uniforms[:, None] > probabilities.cumsum(axis=1)).sum(axis=1), probabilities.shape[1] - 1)


def _forest_predictions(models, x_query, known_y, priors, widths_x, widths_y, config, deadline, use_chains=True):
    """Step 6: average conditional probabilities over stochastic chain states."""
    direct_models, conditional_models, links = models
    background = _one_hot(x_query, widths_x)
    direct = []
    for j, model in enumerate(direct_models):
        _check_budget(deadline)
        direct.append(_forest_probability(model, background, priors[j], config))
    if not use_chains or not any(links):
        return direct, direct
    rng = np.random.default_rng(config["seed"] + 991)
    total = [np.zeros_like(p) for p in direct]
    retained = 0
    for _ in range(config["chains"]):
        state = np.column_stack([_draw(p, rng) for p in direct])
        observed = known_y >= 0
        state[observed] = known_y[observed]
        for sweep in range(config["sweeps"]):
            for j in rng.permutation(len(widths_y)):
                _check_budget(deadline)
                if links[j] and conditional_models[j] is not None:
                    auxiliary = _one_hot(state[:, links[j]], [widths_y[k] for k in links[j]])
                    features = np.concatenate((background, auxiliary), axis=1)
                    p = _forest_probability(conditional_models[j], features, priors[j], config)
                else:
                    p = direct[j]
                if sweep >= config["burn_in"]:
                    total[j] += p
                hidden = known_y[:, j] < 0
                sampled = _draw(p, rng)
                state[hidden, j] = sampled[hidden]
            if sweep >= config["burn_in"]:
                retained += 1
    return direct, [p / retained for p in total]


def _select_mixture(priors, counts, direct, chained, labels, config):
    """Step 7: the competition's floor mixture is used for validation only."""
    scores = np.zeros(len(MIXTURES), dtype=float)
    for j in range(labels.shape[1]):
        valid = labels[:, j] >= 0
        if not np.any(valid):
            continue
        y = labels[valid, j]
        components = np.column_stack((priors[j][y], counts[j][valid, y], direct[j][valid, y], chained[j][valid, y]))
        probability = components @ MIXTURES.T
        probability = config["scoring_floor"] + (1 - len(priors[j]) * config["scoring_floor"]) * probability
        scores += np.log(probability).sum(axis=0)
    return MIXTURES[int(np.argmax(scores))].copy()


def _combine(priors, counts, direct, chained, weights):
    result = []
    for j, count in enumerate(counts):
        p = weights[0] * priors[j] + weights[1] * count
        if weights[2]:
            p = p + weights[2] * direct[j]
        if weights[3]:
            p = p + weights[3] * chained[j]
        result.append(p / p.sum(axis=1, keepdims=True))
    return result


def predict(frame, schema):
    """Return one schema-ordered probability vector for each genuinely blank cell."""
    global LAST_DIAGNOSTICS
    started = time.monotonic()
    cfg = CONFIG
    LAST_DIAGNOSTICS = {"method": "probabilistic_forest_imputation", "budget_backoff": False,
                        "validation_ran": False, "forest_refit_ran": False, "chained_refit_ran": False}
    given, targets, options, x, y = _encode(frame, schema)
    missing = y < 0
    query_rows = np.flatnonzero(np.any(missing, axis=1))
    if not len(query_rows):
        return []
    widths_x, widths_y = [len(options[n]) for n in given], [len(options[n]) for n in targets]
    visible = np.flatnonzero(np.any(y >= 0, axis=1))
    priors, counts = _count_predictions(x[visible], y[visible], x[query_rows], widths_x, widths_y, given, cfg)
    probabilities = counts  # Declared small-data/budget backoff, always available.
    fitting, validation = _split_visible(x, y, given, cfg)
    # Per-call allocation prevents an early instrument consuming the whole run.
    deadline = min(_STARTED + cfg["shared_work_budget_seconds"], started + 40 + 1.65 * len(targets))
    try:
        _check_budget(deadline)
        if len(validation):
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                vpriors, vcounts = _count_predictions(x[fitting], y[fitting], x[validation], widths_x, widths_y, given, cfg)
                models = _fit_forests(x[fitting], y[fitting], targets, schema["items"], widths_x, widths_y, cfg, deadline)
                # No validation answer, including parent answers, enters inference.
                hidden_validation = np.full((len(validation), len(targets)), -1, dtype=np.int32)
                direct, chained = _forest_predictions(models, x[validation], hidden_validation, vpriors, widths_x, widths_y, cfg, deadline)
                weights = _select_mixture(vpriors, vcounts, direct, chained, y[validation], cfg)
                LAST_DIAGNOSTICS["validation_ran"] = True
                del models, direct, chained
                # Step 8: refit selected components on every permitted visible row.
                direct, chained = None, None
                if weights[2] or weights[3]:
                    models = _fit_forests(x[visible], y[visible], targets, schema["items"], widths_x, widths_y, cfg, deadline, use_chains=bool(weights[3]))
                    direct, chained = _forest_predictions(models, x[query_rows], y[query_rows].copy(), priors, widths_x, widths_y, cfg, deadline, use_chains=bool(weights[3]))
                    LAST_DIAGNOSTICS["forest_refit_ran"] = True
                    LAST_DIAGNOSTICS["chained_refit_ran"] = bool(weights[3])
                probabilities = _combine(priors, counts, direct, chained, weights)
    except WorkBudgetReached:
        LAST_DIAGNOSTICS["budget_backoff"] = True
        probabilities = counts
    # Step 9: frame row order, then schema PREDICT order; sentinel is last only
    # where declared. The grader will apply its own scoring floor exactly once.
    output = []
    for position, row in enumerate(query_rows):
        for j in range(len(targets)):
            if missing[row, j]:
                output.append(probabilities[j][position].tolist())
    LAST_DIAGNOSTICS["elapsed_seconds"] = time.monotonic() - started
    return output
