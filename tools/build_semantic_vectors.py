"""Build frozen semantic feature table from public metadata on CPU.

Run with a local copy of the pinned model, the public data/ schema directory
and an output directory. Survey inputs are the three public schema JSON files
and the competition configuration that defines the shared gate sentinel.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
MODEL_WEIGHT_HASH = '53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_key(name, question, value, gated):
    payload = json.dumps([name, question, value, gated], ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def encode(texts, tokenizer, model):
    """Mean-pool non-padding tokens and L2 normalize, following the model card."""
    unique = sorted(set(texts))
    parts = []
    lengths = []
    with torch.inference_mode():
        for start in range(0, len(unique), 32):
            batch = unique[start:start + 32]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=256,
                               return_tensors='pt')
            lengths.extend(tokens['attention_mask'].sum(dim=1).tolist())
            states = model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1)
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            parts.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
    encoded = dict(zip(unique, np.concatenate(parts)))
    return np.stack([encoded[text] for text in texts]), max(lengths)


def compress(rows, spans):
    """Keep the first 32 orthogonal directions of variation between options."""
    centered = rows.astype(np.float64)
    for span in spans:
        centered[span] -= centered[span].mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    basis = axes[:32].copy()
    # Orient each component deterministically by its largest loading.
    signs = np.sign(basis[np.arange(len(basis)), np.abs(basis).argmax(axis=1)])
    projected = centered @ (basis * signs[:, None]).T
    for span in spans:
        # One shared scale per question preserves relative distances among levels.
        rms = np.sqrt(np.mean(np.sum(projected[span] ** 2, axis=1)))
        projected[span] /= max(rms, 1e-6)
    return projected.astype(np.float32)


def build(model_dir, schema_dir, destination):
    torch.set_num_threads(2)
    hashes = {name: digest(model_dir / name) for name in (
        'model.safetensors', 'config.json', 'vocab.txt', 'tokenizer_config.json',
        'tokenizer.json', 'special_tokens_map.json')}
    if hashes['model.safetensors'] != MODEL_WEIGHT_HASH:
        raise ValueError('Encoder weights do not match the pinned release')
    model = AutoModel.from_pretrained(model_dir, local_files_only=True,
                                      use_safetensors=True).eval().cpu()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    keys, option_texts, pair_texts, spans = [], [], [], []
    schemas = {}
    config_path = schema_dir.parent / 'config.yml'
    gated_value = yaml.safe_load(config_path.read_text(encoding='utf-8'))['gated_value']
    for survey in ('unicef', 'world_bank', 'unhcr'):
        path = schema_dir / (survey + '.json')
        schema = json.loads(path.read_text(encoding='utf-8'))
        target_count = option_count = gate_count = missing_question_count = 0
        for name, item in schema['items'].items():
            if item['class'] != 'PREDICT':
                continue
            target_count += 1
            question = item.get('question') or ''
            missing_question_count += int(not question)
            options = [(value, False) for value in item['values']]
            if item.get('gate'):
                options.append((gated_value, True))
                gate_count += 1
            start = len(keys)
            for value, gated in options:
                keys.append(feature_key(name, question, value, gated))
                # Preserve numeric codes and genuine item nonresponse verbatim.
                wording = 'Not asked because of survey routing' if gated else str(value)
                option_texts.append(wording)
                pair_texts.append(f'Answer: {wording}. Question: {question}')
            spans.append(slice(start, len(keys)))
            option_count += len(options)
        schemas[survey] = dict(schema_sha256=digest(path), targets=target_count,
                               options_including_gate=option_count, gated_targets=gate_count,
                               missing_question_text=missing_question_count)
    if len(set(keys)) != len(keys):
        raise ValueError('Ambiguous metadata identity across surveys')
    option_vectors, option_length = encode(option_texts, tokenizer, model)
    pair_vectors, pair_length = encode(pair_texts, tokenizer, model)
    features = np.column_stack((compress(option_vectors, spans), compress(pair_vectors, spans)))
    features *= np.float32(2 ** -0.5)
    destination.mkdir(parents=True, exist_ok=True)
    asset = destination / 'semantic_vectors.npz'
    np.savez_compressed(asset, keys=np.array(keys), vectors=features.astype(np.float16))
    report = dict(model_id=MODEL_ID, model_revision=MODEL_REVISION, model_license='Apache-2.0',
                  model_inputs_sha256=hashes, public_schema_sources=schemas, gate_sentinel=gated_value,
                  public_config_sha256=digest(config_path), asset_sha256=digest(asset),
                  asset_bytes=asset.stat().st_size, rows=len(keys), dimensions=64,
                  max_option_tokens_used=option_length, max_pair_tokens_used=pair_length,
                  token_limit=256, device='cpu', torch_version=str(torch.__version__),
                  numpy_version=np.__version__)
    (destination / 'semantic_provenance.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--schema-dir', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    build(args.model_dir, args.schema_dir, args.destination)
