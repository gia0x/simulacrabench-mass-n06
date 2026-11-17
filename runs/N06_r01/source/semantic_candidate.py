"""N06. Add a trainable shared wording term to N01's independent output heads."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import neural_candidate as base


LAST_DIAGNOSTICS = {}
_BANK = {}
FEATURE_DIMENSION = 64


def feature_key(name, question, value, gated):
    """A stable identity that distinguishes an ordinary label from a routed skip."""
    payload = json.dumps([name, question, value, gated], ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def initialize(cfg):
    """Read and verify the small public-metadata asset once during module import."""
    global _BANK
    path = Path(__file__).with_name('semantic_vectors.npz')
    if hashlib.sha256(path.read_bytes()).hexdigest() != cfg['semantic_asset_sha256']:
        raise ValueError('Semantic vectors differ from the frozen metadata asset')
    with np.load(path, allow_pickle=False) as asset:
        keys = asset['keys'].tolist()
        vectors = asset['vectors'].astype(np.float32)
    if (len(set(keys)) != len(keys) or vectors.shape != (len(keys), FEATURE_DIMENSION)
            or len(keys) != cfg['semantic_asset_rows'] or not np.isfinite(vectors).all()):
        raise ValueError('Invalid semantic vector asset')
    _BANK = dict(zip(keys, vectors))


def _ordered_features(schema, targets, widths):
    """Align public text vectors with the categories actually supplied by the grader."""
    if not _BANK:
        raise RuntimeError('Semantic vectors must be initialized at module import')
    features = []
    matched = 0
    for name, width in zip(targets, widths):
        item = schema['items'][name]
        question = item.get('question') or ''
        keys = [feature_key(name, question, value, False) for value in item['values']]
        if item.get('gate'):
            keys.append(feature_key(name, question, schema['gated_value'], True))
        if len(keys) != width:
            raise ValueError('Semantic head width differs from the declared schema')
        if all(key in _BANK for key in keys):
            rows = np.stack([_BANK[key] for key in keys])
            # A common offset across all allowed answers cancels in softmax.
            features.append(rows - rows.mean(axis=0))
            matched += 1
        else:
            # Unknown/changed metadata uses the ordinary N01 head for this item.
            features.append(np.zeros((width, FEATURE_DIMENSION), dtype=np.float32))
    if features:
        return np.concatenate(features), matched
    return np.zeros((0, FEATURE_DIMENSION), dtype=np.float32), matched


class WordingReadout(nn.Module):
    def __init__(self, independent, vectors, strength):
        super().__init__()
        self.independent = independent
        self.register_buffer('vectors', torch.from_numpy(vectors))
        self.projection = nn.Parameter(torch.zeros(independent.in_features, vectors.shape[1]))
        self.strength = strength

    def forward(self, summary):
        # Zero initialization gives exactly the ordinary N01 logits before training.
        latent = summary @ self.projection
        return self.independent(summary) + self.strength * (latent @ self.vectors.T)


def fit_predict(x_fit, y_fit, x_query, given, targets, widths_x, widths_y, cfg, deadline):
    global LAST_DIAGNOSTICS
    vectors, matched = _ordered_features(cfg['runtime_schema'], targets, widths_y)
    strength = float(cfg['semantic_strength'])
    if not 0 < strength <= 1:
        raise ValueError('Invalid semantic correction scale')
    latest_readout = None

    def factory(input_widths, output_widths, priors, settings):
        nonlocal latest_readout
        model = base.SurveyTransformer(input_widths, output_widths, priors, settings)
        model.output = WordingReadout(model.output, vectors, strength)
        latest_readout = model.output
        return model

    try:
        return base.fit_predict(x_fit, y_fit, x_query, given, targets, widths_x, widths_y,
                                cfg, deadline, model_factory=factory,
                                method='shared_survey_transformer_with_wording')
    finally:
        LAST_DIAGNOSTICS = dict(base.LAST_DIAGNOSTICS, semantic_matched_targets=matched,
                                semantic_total_targets=len(targets),
                                semantic_feature_dimension=FEATURE_DIMENSION)
        if latest_readout is not None and base.LAST_DIAGNOSTICS.get('refit_completed'):
            LAST_DIAGNOSTICS['semantic_projection_norm'] = latest_readout.projection.detach().norm().item()
