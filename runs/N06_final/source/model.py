import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


FEATURE_DIMENSION = 64
ASSET_SHA256 = 'a6a973cc91ae131f5e6fb2a7673cbde8c91f3fd094b9ca019fc60a255ca0dff0'
SEMANTIC_STRENGTH = 0.5
MODEL_WIDTH = 64
LAYERS = 2
HEADS = 4
DROPOUT = 0.1
MAX_EPOCHS = 16
PATIENCE = 3
BATCH_SIZE = 256
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.01
HOLDOUT_FRACTION = 0.15
MAX_HOLDOUT_ROWS = 256
MIN_TRAINING_ROWS = 32
TORCH_THREADS = 2
PRIOR_SHARE = 0.02
PSEUDOCOUNT = 0.5
SCORING_FLOOR = 0.001
SEED = 1777


def _feature_key(name, question, value, gated):
    payload = json.dumps([name, question, value, gated], ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _load_vectors():
    # Survey wording: https://github.com/SituatedEvals/public/tree/aa03186f8f083c7785a1a566c802c2077925e265/data
    # Encoder (Apache-2.0): https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf
    path = Path(__file__).with_name('semantic_vectors.npz')
    if hashlib.sha256(path.read_bytes()).hexdigest() != ASSET_SHA256:
        raise ValueError('Semantic vectors differ from the frozen metadata asset')
    with np.load(path, allow_pickle=False) as asset:
        keys = asset['keys'].tolist()
        vectors = asset['vectors'].astype(np.float32)
    if (len(set(keys)) != len(keys) or vectors.shape != (len(keys), FEATURE_DIMENSION)
            or len(keys) != 1550 or not np.isfinite(vectors).all()):
        raise ValueError('Invalid semantic vector asset')
    return dict(zip(keys, vectors))


_BANK = _load_vectors()


def _ordered_features(schema, targets, widths):
    features = []
    for name, width in zip(targets, widths):
        item = schema['items'][name]
        question = item.get('question') or ''
        keys = [_feature_key(name, question, value, False) for value in item['values']]
        if item.get('gate'):
            keys.append(_feature_key(name, question, schema['gated_value'], True))
        if len(keys) != width:
            raise ValueError('Semantic head width differs from the declared schema')
        if all(key in _BANK for key in keys):
            rows = np.stack([_BANK[key] for key in keys])
            features.append(rows - rows.mean(axis=0))
        else:
            # Use only the independent head when the wording or answer set changes.
            features.append(np.zeros((width, FEATURE_DIMENSION), dtype=np.float32))
    if features:
        return np.concatenate(features)
    return np.zeros((0, FEATURE_DIMENSION), dtype=np.float32)


class WordingReadout(nn.Module):
    def __init__(self, independent, vectors):
        super().__init__()
        self.independent = independent
        self.register_buffer('vectors', torch.from_numpy(vectors))
        self.projection = nn.Parameter(torch.zeros(independent.in_features, vectors.shape[1]))

    def forward(self, summary):
        latent = summary @ self.projection
        return self.independent(summary) + SEMANTIC_STRENGTH * (latent @ self.vectors.T)


def _check(deadline, device=None):
    if device is not None and device.type == 'cuda':
        torch.cuda.synchronize(device)
    if time.monotonic() >= deadline:
        raise TimeoutError('Neural provider work deadline exceeded')


def item_priors(labels, widths):
    priors, active = [], []
    for j, width in enumerate(widths):
        observed = labels[labels[:, j] >= 0, j]
        counts = np.bincount(observed, minlength=width).astype(np.float64)
        counts += PSEUDOCOUNT
        priors.append(counts / counts.sum())
        active.append(len(np.unique(observed)) >= 2)
    return priors, np.asarray(active, dtype=bool)


def _categorical_codes(codes, widths):
    """Each feature has its own embedding range and an unknown/missing slot."""
    encoded = np.empty(codes.shape, dtype=np.int64)
    offset = 0
    for j, width in enumerate(widths):
        col = codes[:, j]
        valid = (col >= 0) & (col < width)
        encoded[:, j] = np.where(valid, col, width) + offset
        offset += width + 1
    return encoded


class SurveyTransformer(nn.Module):
    def __init__(self, widths_x, widths_y, priors, vectors):
        super().__init__()
        self.embedding = nn.Embedding(sum(width + 1 for width in widths_x), MODEL_WIDTH)
        self.feature_embedding = nn.Parameter(torch.empty(1, len(widths_x), MODEL_WIDTH))
        self.summary = nn.Parameter(torch.empty(1, 1, MODEL_WIDTH))
        nn.init.normal_(self.feature_embedding, std=0.02)
        nn.init.normal_(self.summary, std=0.02)
        self.input_norm = nn.LayerNorm(MODEL_WIDTH)
        layer = nn.TransformerEncoderLayer(
            d_model=MODEL_WIDTH, nhead=HEADS, dim_feedforward=MODEL_WIDTH * 2,
            dropout=DROPOUT, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=LAYERS,
                                            enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(MODEL_WIDTH)
        independent = nn.Linear(MODEL_WIDTH, sum(widths_y))
        nn.init.normal_(independent.weight, std=0.01)
        with torch.no_grad():
            independent.bias.copy_(torch.tensor(np.log(np.concatenate(priors)), dtype=torch.float32))
        self.output = WordingReadout(independent, vectors)
        gather = np.zeros((len(widths_y), max(widths_y)), dtype=np.int64)
        support = np.zeros_like(gather, dtype=bool)
        offset = 0
        for j, width in enumerate(widths_y):
            gather[j, :width] = np.arange(offset, offset + width)
            support[j, :width] = True
            offset += width
        self.register_buffer('gather', torch.from_numpy(gather))
        self.register_buffer('support', torch.from_numpy(support))

    def forward(self, codes):
        tokens = self.input_norm(self.embedding(codes) + self.feature_embedding)
        tokens = torch.cat((self.summary.expand(len(codes), -1, -1), tokens), dim=1)
        summary = self.encoder(tokens)[:, 0]
        packed = self.output(self.output_norm(summary))
        logits = packed[:, self.gather]
        return logits.masked_fill(~self.support, -1e4)


def _loss(logits, labels, active):
    observed = (labels >= 0) & active[None, :]
    log_probs = logits.log_softmax(dim=-1)
    selected = log_probs.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return -(selected * observed).sum() / observed.sum().clamp_min(1)


def _probabilities(model, x, priors, active, widths, deadline, device):
    model.eval()
    by_item = [[] for _ in widths]
    with torch.inference_mode():
        for start in range(0, len(x), BATCH_SIZE):
            _check(deadline)
            values = model(x[start:start + BATCH_SIZE]).softmax(dim=-1).cpu().numpy()
            for j, width in enumerate(widths):
                p = values[:, j, :width].astype(np.float64)
                p = ((1 - PRIOR_SHARE) * p + PRIOR_SHARE * priors[j]
                     if active[j] else np.broadcast_to(priors[j], p.shape).copy())
                p /= p.sum(axis=1, keepdims=True)
                by_item[j].append(p)
    _check(deadline, device)
    return [np.concatenate(parts) if parts else np.empty((0, width), dtype=np.float64)
            for parts, width in zip(by_item, widths)]


def _score(probabilities, labels):
    total, count = 0.0, 0
    for j, p in enumerate(probabilities):
        rows = np.flatnonzero(labels[:, j] >= 0)
        if len(rows):
            adjusted = SCORING_FLOOR + (1 - p.shape[1] * SCORING_FLOOR) * p[rows, labels[rows, j]]
            total += np.log(adjusted).sum()
            count += len(rows)
    return total / count if count else -math.inf


def _epoch(model, optimizer, x, y, active, rng, deadline, device):
    model.train()
    permutation = rng.permutation(len(x))
    for start in range(0, len(x), BATCH_SIZE):
        _check(deadline)
        indices = torch.as_tensor(permutation[start:start + BATCH_SIZE], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model(x[indices]), y[indices], active)
        if not torch.isfinite(loss).item():
            raise ValueError('Nonfinite neural training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    _check(deadline, device)


def _select_epochs(x, y, labels, fit, hold, widths_x, widths_y, vectors, deadline, device):
    priors, active = item_priors(labels[fit], widths_y)
    if not active.any():
        return 1
    model = SurveyTransformer(widths_x, widths_y, priors, vectors).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    active_tensor = torch.as_tensor(active, device=device)
    train_x, train_y, hold_x = x[fit], y[fit], x[hold]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 9022]))
    best_score, best_epoch, stale = -math.inf, 1, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        _epoch(model, optimizer, train_x, train_y, active_tensor, rng, deadline, device)
        probabilities = _probabilities(model, hold_x, priors, active, widths_y, deadline, device)
        score = _score(probabilities, labels[hold])
        if score > best_score + 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    return best_epoch


def fit_predict(x_fit, y_fit, x_query, targets, widths_x, widths_y, schema, deadline):
    vectors = _ordered_features(schema, targets, widths_y)
    _check(deadline)
    x_fit = np.asarray(x_fit, dtype=np.int64)
    y_fit = np.asarray(y_fit, dtype=np.int64)
    x_query = np.asarray(x_query, dtype=np.int64)
    widths_x, widths_y = list(map(int, widths_x)), list(map(int, widths_y))
    if (x_fit.ndim != 2 or y_fit.ndim != 2 or x_query.ndim != 2
            or len(x_fit) != len(y_fit) or x_fit.shape[1] != len(widths_x)
            or x_query.shape[1] != len(widths_x) or y_fit.shape[1] != len(widths_y)
            or len(targets) != len(widths_y)
            or any(width < 1 for width in widths_x + widths_y)):
        raise ValueError('Invalid encoded neural shapes or supports')
    for j, width in enumerate(widths_y):
        if ((y_fit[:, j] < -1) | (y_fit[:, j] >= width)).any():
            raise ValueError('Target code outside declared support')
        if width * SCORING_FLOOR >= 1:
            raise ValueError('Scoring floor incompatible with target support')
    if not widths_y:
        return []
    priors, active = item_priors(y_fit, widths_y)
    if not widths_x or len(x_fit) < MIN_TRAINING_ROWS or not active.any():
        return [np.tile(prior, (len(x_query), 1)) for prior in priors]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_threads = torch.get_num_threads()
    torch.set_num_threads(TORCH_THREADS)
    try:
        split_rng = np.random.default_rng(np.random.SeedSequence([SEED, 9021]))
        permutation = split_rng.permutation(len(x_fit))
        n_hold = min(MAX_HOLDOUT_ROWS, max(1, int(len(x_fit) * HOLDOUT_FRACTION)))
        n_hold = min(n_hold, len(x_fit) - 2)
        hold, fit = np.sort(permutation[:n_hold]), np.sort(permutation[n_hold:])
        train_tensor = torch.as_tensor(_categorical_codes(x_fit, widths_x), device=device)
        label_tensor = torch.as_tensor(y_fit, device=device)
        query_tensor = torch.as_tensor(_categorical_codes(x_query, widths_x), device=device)
        cuda_devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(SEED)
            epochs = _select_epochs(train_tensor, label_tensor, y_fit, fit, hold, widths_x,
                                    widths_y, vectors, deadline, device)
            torch.manual_seed(SEED)
            model = SurveyTransformer(widths_x, widths_y, priors, vectors).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            active_tensor = torch.as_tensor(active, device=device)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 9023]))
            for _ in range(epochs):
                _epoch(model, optimizer, train_tensor, label_tensor, active_tensor, rng, deadline, device)
            result = _probabilities(model, query_tensor, priors, active, widths_y, deadline, device)
        _check(deadline, device)
        return result
    finally:
        torch.set_num_threads(original_threads)
