import time

_WALL_DEADLINE = time.monotonic() + 825

import numpy as np
import pandas as pd

import model

_remaining_fit_seconds = 400.0


def _encode(frame, schema, role):
    names = [name for name, item in schema["items"].items() if item["class"] == role]
    widths = []
    matrix = np.empty((len(frame), len(names)), dtype=np.int32)
    for column, name in enumerate(names):
        item = schema["items"][name]
        options = list(item["values"])
        if item.get("gate"):
            options.append(schema["gated_value"])
        if not options or len(set(options)) != len(options):
            raise ValueError(f"Invalid category list for {name}")
        series = frame[name]
        codes = pd.Categorical(series, categories=options).codes
        missing = series.isna().to_numpy()
        if np.any((codes < 0) & ~missing):
            raise ValueError(f"Unknown category in {name}")
        if role == "GIVEN" and missing.any():
            raise ValueError(f"Missing background value in {name}")
        matrix[:, column] = codes
        widths.append(len(options))
    return names, widths, matrix


def predict(frame, schema):
    global _remaining_fit_seconds
    _, input_widths, inputs = _encode(frame, schema, "GIVEN")
    targets, output_widths, labels = _encode(frame, schema, "PREDICT")
    query = np.flatnonzero(np.any(labels < 0, axis=1))
    if not len(query):
        return []
    visible = np.any(labels >= 0, axis=1)
    fitting_labels = labels[visible]
    started = time.monotonic()
    deadline = min(_WALL_DEADLINE, started + _remaining_fit_seconds)
    try:
        if started >= deadline:
            raise TimeoutError
        matrices = model.fit_predict(inputs[visible], fitting_labels, inputs[query],
                                     targets, input_widths, output_widths, schema, deadline)
        if time.monotonic() >= deadline:
            raise TimeoutError
    except TimeoutError:
        priors, _ = model.item_priors(fitting_labels, output_widths)
        matrices = [np.broadcast_to(prior, (len(query), len(prior))) for prior in priors]
    finally:
        _remaining_fit_seconds -= time.monotonic() - started

    output = []
    for position, row in enumerate(query):
        for column in np.flatnonzero(labels[row] < 0):
            values = matrices[column][position]
            output.append((values / values.sum()).tolist())
    return output
