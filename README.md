# MASS N06: survey-response prediction with wording embeddings

This repository contains team MASS's N06 submission for the NeurIPS 2026
SimulacraBench competition, the four-page technical report, and the files needed
to inspect and rebuild the released archives. N06 combines a small survey
transformer with frozen embeddings of the published question and answer wording.

The repository contains no respondent microdata or hidden survey answers.

## Repository contents

- `runs/N06_r01/source/` is the exact source tree stored in the frozen N06_r01
  development archive.
- `runs/N06_r01/manifest.json` records the hash of every archive member and the
  complete N06_r01 ZIP.
- `runs/N06_final/source/` is the cleaned public implementation supplied for the
  code release.
- `runs/N06_final/manifest.json` is the manifest supplied with that cleaned
  implementation.
- `artifacts/` preserves both original ZIP archives.
- `tools/rebuild_n06.py` deterministically rebuilds either archive from its run
  directory and verifies all hashes.
- `tools/build_semantic_vectors.py` regenerates the wording-vector asset from
  the pinned MiniLM model and public SimulacraBench schemas.
- `report/` contains the LaTeX source, style file, and PDF of the technical
  report.

```text
simulacrabench-mass-n06/
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── artifacts/
│   ├── N06_r01.zip
│   └── N06_final.zip
├── report/
│   ├── MASS_N06_report.pdf
│   ├── MASS_N06_report.tex
│   └── neurips_2026.sty
├── runs/
│   ├── N06_r01/
│   │   ├── manifest.json
│   │   └── source/
│   │       ├── main.py
│   │       ├── neural_candidate.py
│   │       ├── semantic_candidate.py
│   │       ├── semantic_vectors.npz
│   │       ├── semantic_provenance.json
│   │       ├── requirements.txt
│   │       └── f05_reference/
│   └── N06_final/
│       ├── manifest.json
│       └── source/
│           ├── main.py
│           ├── model.py
│           ├── requirements.txt
│           └── semantic_vectors.npz
└── tools/
    ├── build_semantic_vectors.py
    └── rebuild_n06.py
```

N06_r01 and N06_final are deliberately kept separate. N06_r01 is the frozen
development artifact discussed in the report. N06_final is a smaller cleaned
implementation prepared for public release; it is not represented as a
byte-identical copy of N06_r01.

## Frozen archive hashes

| Archive | SHA-256 |
| --- | --- |
| `N06_r01.zip` | `5e83cb5a5dd6e4cf5a481e1013e0a112960340e7608bf90cc0b7586b0a4a38f2` |
| `N06_final.zip` | `8417f959e53b2407330fefcb068e79ebf55cb9768493e8caae1aefb7ad67d3d8` |
| `semantic_vectors.npz` | `a6a973cc91ae131f5e6fb2a7673cbde8c91f3fd094b9ca019fc60a255ca0dff0` |

## Rebuild the archives

Run these commands from the repository root. Each output path must not already
exist because the rebuild utility refuses to overwrite files.

```bash
python tools/rebuild_n06.py runs/N06_r01 rebuilt-N06_r01.zip
python tools/rebuild_n06.py runs/N06_final rebuilt-N06_final.zip
```

On success, the script prints the verified SHA-256 digest. The rebuilt archive
must match the corresponding digest in the table above.

## Regenerate the wording vectors

The feature generator uses the public schemas from organiser commit
`aa03186f8f083c7785a1a566c802c2077925e265` and MiniLM revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`.

```bash
git clone https://github.com/SituatedEvals/public.git public-schemas
git -C public-schemas checkout aa03186f8f083c7785a1a566c802c2077925e265

hf download sentence-transformers/all-MiniLM-L6-v2 \
  --revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf \
  --local-dir model-miniLM

python tools/build_semantic_vectors.py \
  --model-dir model-miniLM \
  --schema-dir public-schemas/data \
  --destination rebuilt-features
```

The generator writes `semantic_vectors.npz` and
`semantic_provenance.json`. The expected vector-asset digest is listed above.
MiniLM is needed only to regenerate the asset; evaluation uses the bundled
vectors and does not download the model.

## Validate against the public submission contract

The official SimulacraBench repository provides the archive checker and sandbox
grader. For example:

```bash
python public-schemas/tools/check_submission_zip.py artifacts/N06_r01.zip
```

The checker verifies archive layout and execution compatibility. It cannot
reproduce the official score because the held-out respondent data remain with
the organisers.

## Licence and third-party materials

Team-authored source code in this repository is released under the MIT License.
See `THIRD_PARTY_NOTICES.md` for the public schemas, sentence encoder, and other
third-party materials used to create the bundled wording vectors.
