"""Compact, shared tabular transformer fitted only to supplied visible labels.

One encoder per survey shares information across targets. Target answers are
never encoder inputs. The inner early-stopping rows come only from x_fit/y_fit;
the caller owns the separate outer blend-selection split.
"""
import math
import time

import numpy as np
import torch
from torch import nn


LAST_DIAGNOSTICS = {}


def _check(deadline, device=None):
    if device is not None and device.type == 'cuda':
        torch.cuda.synchronize(device)
    if time.monotonic() >= deadline:
        raise TimeoutError('Neural provider work deadline exceeded')


def _settings(cfg):
    settings = {
        'width': int(cfg.get('neural_width', 64)),
        'layers': int(cfg.get('neural_layers', 2)),
        'heads': int(cfg.get('neural_heads', 4)),
        'dropout': float(cfg.get('neural_dropout', 0.1)),
        'epochs': int(cfg.get('neural_max_epochs', 16)),
        'patience': int(cfg.get('neural_patience', 3)),
        'batch': int(cfg.get('neural_batch_size', 256)),
        'lr': float(cfg.get('neural_learning_rate', 0.002)),
        'decay': float(cfg.get('neural_weight_decay', 0.01)),
        'fraction': float(cfg.get('neural_early_stopping_fraction', 0.15)),
        'cap': int(cfg.get('neural_early_stopping_cap', 256)),
        'min_rows': int(cfg.get('neural_min_training_rows', 32)),
        'threads': int(cfg.get('neural_threads', 2)),
        'prior_share': float(cfg.get('prior_share', 0.02)),
        'pseudocount': float(cfg.get('marginal_pseudocount', 0.5)),
        'floor': float(cfg.get('scoring_floor', 0.001)),
        'seed': int(cfg.get('seed', 1777)),
    }
    if (settings['width'] < 4 or settings['heads'] < 1
            or settings['width'] % settings['heads'] or settings['layers'] < 1
            or settings['epochs'] < 1 or settings['patience'] < 1
            or settings['batch'] < 1 or settings['threads'] < 1
            or not 0 <= settings['dropout'] < 1
            or not 0 < settings['fraction'] < 0.5 or settings['cap'] < 1
            or settings['min_rows'] < 2 or not 0 <= settings['prior_share'] <= 1
            or settings['pseudocount'] <= 0 or settings['lr'] <= 0
            or settings['decay'] < 0 or settings['floor'] < 0):
        raise ValueError('Invalid compact-transformer configuration')
    return settings


def _priors(labels, widths, pseudocount):
    priors, active = [], []
    for j, width in enumerate(widths):
        observed = labels[labels[:, j] >= 0, j]
        counts = np.bincount(observed, minlength=width).astype(np.float64)
        counts += pseudocount
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
    def __init__(self, widths_x, widths_y, priors, settings):
        super().__init__()
        dimension = settings['width']
        self.embedding = nn.Embedding(sum(width + 1 for width in widths_x), dimension)
        self.feature_embedding = nn.Parameter(torch.empty(1, len(widths_x), dimension))
        self.summary = nn.Parameter(torch.empty(1, 1, dimension))
        nn.init.normal_(self.feature_embedding, std=0.02)
        nn.init.normal_(self.summary, std=0.02)
        self.input_norm = nn.LayerNorm(dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension, nhead=settings['heads'], dim_feedforward=dimension * 2,
            dropout=settings['dropout'], activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=settings['layers'],
                                            enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(dimension)
        self.output = nn.Linear(dimension, sum(widths_y))
        nn.init.normal_(self.output.weight, std=0.01)
        with torch.no_grad():
            self.output.bias.copy_(torch.tensor(np.log(np.concatenate(priors)), dtype=torch.float32))
        # Gather the packed target heads into a padded tensor for one vectorized
        # categorical loss. Padding is never a permitted output category.
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
    # Multiplying by a mask is safe: padded logits use finite -1e4 rather than
    # -inf, and missing labels gather legal index zero.
    return -(selected * observed).sum() / observed.sum().clamp_min(1)


def _probabilities(model, x, priors, active, widths, settings, deadline, device):
    model.eval()
    by_item = [[] for _ in widths]
    with torch.inference_mode():
        for start in range(0, len(x), settings['batch']):
            _check(deadline)
            values = model(x[start:start + settings['batch']]).softmax(dim=-1).cpu().numpy()
            for j, width in enumerate(widths):
                p = values[:, j, :width].astype(np.float64)
                p = ((1 - settings['prior_share']) * p + settings['prior_share'] * priors[j]
                     if active[j] else np.broadcast_to(priors[j], p.shape).copy())
                p /= p.sum(axis=1, keepdims=True)
                by_item[j].append(p)
    _check(deadline, device)
    return [np.concatenate(parts) if parts else np.empty((0, width), dtype=np.float64)
            for parts, width in zip(by_item, widths)]


def _score(probabilities, labels, floor):
    total, count = 0.0, 0
    for j, p in enumerate(probabilities):
        rows = np.flatnonzero(labels[:, j] >= 0)
        if len(rows):
            adjusted = floor + (1 - p.shape[1] * floor) * p[rows, labels[rows, j]]
            total += np.log(adjusted).sum()
            count += len(rows)
    return total / count if count else -math.inf


def _epoch(model, optimizer, x, y, active, settings, rng, deadline, device):
    model.train()
    permutation = rng.permutation(len(x))
    for start in range(0, len(x), settings['batch']):
        _check(deadline)
        indices = torch.as_tensor(permutation[start:start + settings['batch']], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model(x[indices]), y[indices], active)
        if not torch.isfinite(loss).item():
            raise ValueError('Nonfinite neural training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    _check(deadline, device)


def fit_predict(x_fit, y_fit, x_query, given, targets, widths_x, widths_y, cfg, deadline,
                *, model_factory=SurveyTransformer, method='shared_survey_transformer'):
    """Return one n_query x support probability matrix per declared target."""
    global LAST_DIAGNOSTICS
    started = time.monotonic()
    LAST_DIAGNOSTICS = {
        'method': method, 'training_ran': False,
        'early_stopping_completed': False, 'refit_completed': False,
        'selected_epochs': 0, 'selection_epochs_run': 0, 'device': None,
        'budget_backoff': False, 'fallback_reason': None,
    }
    original_threads = torch.get_num_threads()
    device = None
    try:
        _check(deadline)
        settings = _settings(cfg)
        x_fit = np.asarray(x_fit, dtype=np.int64)
        y_fit = np.asarray(y_fit, dtype=np.int64)
        x_query = np.asarray(x_query, dtype=np.int64)
        widths_x, widths_y = list(map(int, widths_x)), list(map(int, widths_y))
        if (x_fit.ndim != 2 or y_fit.ndim != 2 or x_query.ndim != 2
                or len(x_fit) != len(y_fit) or x_fit.shape[1] != len(widths_x)
                or x_query.shape[1] != len(widths_x) or y_fit.shape[1] != len(widths_y)
                or len(given) != len(widths_x) or len(targets) != len(widths_y)
                or any(width < 1 for width in widths_x + widths_y)):
            raise ValueError('Invalid encoded neural shapes or supports')
        for j, width in enumerate(widths_y):
            if ((y_fit[:, j] < -1) | (y_fit[:, j] >= width)).any():
                raise ValueError('Target code outside declared support')
            if width * settings['floor'] >= 1:
                raise ValueError('Scoring floor incompatible with target support')
        if not widths_y:
            LAST_DIAGNOSTICS['fallback_reason'] = 'no_targets'
            return []
        observed = np.flatnonzero(np.any(y_fit >= 0, axis=1))
        x_fit, y_fit = x_fit[observed], y_fit[observed]
        priors, active = _priors(y_fit, widths_y, settings['pseudocount'])
        LAST_DIAGNOSTICS.update(fitting_respondents=len(y_fit), targets=len(widths_y),
                                trainable_targets=int(active.sum()))
        if not len(widths_x) or len(x_fit) < settings['min_rows'] or not active.any():
            LAST_DIAGNOSTICS.update(fallback_reason='insufficient_trainable_data', refit_completed=True)
            return [np.tile(prior, (len(x_query), 1)) for prior in priors]
        requested = cfg.get('neural_device', 'auto')
        device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                              if requested == 'auto' else requested)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Only CPU or CUDA neural devices are supported')
        LAST_DIAGNOSTICS['device'] = str(device)
        torch.set_num_threads(settings['threads'])
        # This split and its shuffling stream are independent of the caller's
        # outer fitting/blend split. It contains no rows outside x_fit/y_fit.
        split_rng = np.random.default_rng(np.random.SeedSequence([settings['seed'], 9021]))
        permutation = split_rng.permutation(len(x_fit))
        n_hold = min(settings['cap'], max(1, int(len(x_fit) * settings['fraction'])))
        n_hold = min(n_hold, len(x_fit) - 2)
        hold, fit = np.sort(permutation[:n_hold]), np.sort(permutation[n_hold:])
        LAST_DIAGNOSTICS.update(inner_training_respondents=len(fit),
                                inner_stopping_respondents=len(hold))
        inner_priors, inner_active = _priors(y_fit[fit], widths_y, settings['pseudocount'])
        encoded_fit = _categorical_codes(x_fit, widths_x)
        encoded_query = _categorical_codes(x_query, widths_x)
        train_tensor = torch.as_tensor(encoded_fit, device=device)
        label_tensor = torch.as_tensor(y_fit, device=device)
        query_tensor = torch.as_tensor(encoded_query, device=device)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(settings['seed'])
            if inner_active.any():
                model = model_factory(widths_x, widths_y, inner_priors, settings).to(device)
                LAST_DIAGNOSTICS['parameter_count'] = sum(p.numel() for p in model.parameters())
                optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'],
                                              weight_decay=settings['decay'])
                active_tensor = torch.as_tensor(inner_active, device=device)
                train_x, train_y = train_tensor[fit], label_tensor[fit]
                hold_x = train_tensor[hold]
                train_rng = np.random.default_rng(np.random.SeedSequence([settings['seed'], 9022]))
                best, best_epoch, stale = -math.inf, 1, 0
                for epoch in range(1, settings['epochs'] + 1):
                    _epoch(model, optimizer, train_x, train_y, active_tensor,
                           settings, train_rng, deadline, device)
                    probabilities = _probabilities(model, hold_x, inner_priors, inner_active,
                                                   widths_y, settings, deadline, device)
                    score = _score(probabilities, y_fit[hold], settings['floor'])
                    LAST_DIAGNOSTICS.update(training_ran=True, selection_epochs_run=epoch)
                    if score > best + 1e-5:
                        best, best_epoch, stale = score, epoch, 0
                    else:
                        stale += 1
                    if stale >= settings['patience']:
                        break
                LAST_DIAGNOSTICS['inner_stopping_log_score'] = best if math.isfinite(best) else None
                del model, optimizer, train_x, train_y, hold_x
            else:
                best_epoch = 1
            LAST_DIAGNOSTICS.update(early_stopping_completed=True, selected_epochs=best_epoch)
            # Reinitialize and fit all supplied respondents for the chosen count.
            # No retained training weights or labels cross the provider boundary.
            torch.manual_seed(settings['seed'])
            final_model = model_factory(widths_x, widths_y, priors, settings).to(device)
            LAST_DIAGNOSTICS['parameter_count'] = sum(p.numel() for p in final_model.parameters())
            optimizer = torch.optim.AdamW(final_model.parameters(), lr=settings['lr'],
                                          weight_decay=settings['decay'])
            active_tensor = torch.as_tensor(active, device=device)
            train_rng = np.random.default_rng(np.random.SeedSequence([settings['seed'], 9023]))
            for _ in range(best_epoch):
                _epoch(final_model, optimizer, train_tensor, label_tensor, active_tensor,
                       settings, train_rng, deadline, device)
            LAST_DIAGNOSTICS.update(training_ran=True, refit_completed=True)
            result = _probabilities(final_model, query_tensor, priors, active, widths_y,
                                    settings, deadline, device)
        _check(deadline, device)
        return result
    except TimeoutError:
        LAST_DIAGNOSTICS['budget_backoff'] = True
        raise
    finally:
        torch.set_num_threads(original_threads)
        LAST_DIAGNOSTICS['elapsed_seconds'] = time.monotonic() - started
