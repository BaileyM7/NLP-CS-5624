# Sample data

Small committed samples for development and tests. Full scenario files go in `data/scenarios/` (gitignored) and are rebuilt from a seed.

| File | Contents | Rebuild with |
|---|---|---|
| `hotpotqa_sample.jsonl` | 5 questions × 6 fault conditions = 30 scenarios | `python -m toolharness.data.hotpotqa --n 5 --seed 0 --out data/sample/hotpotqa_sample.jsonl` |

LongSeal (SealQA) data is never committed: it is a benchmark with a training-contamination canary and this repo is public. Build it locally (downloads ~73 MB once):

```
python -m toolharness.data.longseal --n 30 --seed 0 --out data/scenarios/longseal.jsonl
```

## Source and license
HotpotQA (Yang et al., 2018), distractor setting, dev split, via Hugging Face `hotpotqa/hotpot_qa`. Licensed under CC BY-SA 4.0.
