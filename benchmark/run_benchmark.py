"""Benchmark Silesian<->Polish MT models on two test sets.

Test sets: brjezierski/benchmark-pl-szl (100 pairs) and FLORES+ devtest (1012 pairs, gated).
Writes outputs_<set>.csv and summary.csv next to this file. FLORES+ text is never written
to disk in the repo (its terms forbid re-hosting): its outputs file holds ids and hypotheses only.

Usage: python run_benchmark.py [--sets benchmark-pl-szl flores] [--limit N]
"""
import argparse
import json
import random
from functools import cache
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HERE = Path(__file__).parent
OWN = {"szl-pl": "brjezierski/opus-mt-sla-sla-finetuned-szl-to-pl", "pl-szl": "brjezierski/opus-mt-sla-sla-finetuned-pl-to-szl"}
BASE = "Helsinki-NLP/opus-mt-sla-sla"
NLLB = "facebook/nllb-200-distilled-600M"
NLLB_CODES = {"szl": "szl_Latn", "pl": "pol_Latn"}
BATCH = 16
N_BOOT = 500
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_benchmark(limit):
    path = hf_hub_download("brjezierski/benchmark-pl-szl", "data/test-00000-of-00001.parquet", repo_type="dataset")
    rows = pd.read_parquet(path).translation.tolist()[:limit]
    return [str(i) for i in range(len(rows))], [r["szl"] for r in rows], [r["pl"] for r in rows]


def load_flores(limit):
    texts = {}
    for lang in ("szl_Latn", "pol_Latn"):
        path = hf_hub_download("openlanguagedata/flores_plus", f"devtest/{lang}.jsonl", repo_type="dataset")
        texts[lang] = [json.loads(line) for line in open(path)]
    szl, pl = texts["szl_Latn"][:limit], texts["pol_Latn"][:limit]
    assert [r["id"] for r in szl] == [r["id"] for r in pl], "FLORES+ rows are not aligned"
    return [str(r["id"]) for r in szl], [r["text"] for r in szl], [r["text"] for r in pl]


TESTSETS = {"benchmark-pl-szl": load_benchmark, "flores": load_flores}


@cache
def load_model(model_id, **tok_kwargs):
    return AutoTokenizer.from_pretrained(model_id, **tok_kwargs), AutoModelForSeq2SeqLM.from_pretrained(model_id).eval().to(DEVICE)


def generate(model_id, sources, prefix=""):
    tok, model = load_model(model_id)
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))  # length-sorted batches pad less
    outs = [None] * len(sources)
    for start in range(0, len(order), BATCH):
        idx = order[start:start + BATCH]
        batch = tok([prefix + sources[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**batch)
        for i, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            outs[i] = text
    return outs


def run_nllb(sources, src, tgt):
    tok, model = load_model(NLLB, src_lang=NLLB_CODES[src])
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))
    outs = [None] * len(sources)
    for start in range(0, len(order), BATCH):
        idx = order[start:start + BATCH]
        batch = tok([sources[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**batch, forced_bos_token_id=tok.convert_tokens_to_ids(NLLB_CODES[tgt]), num_beams=4, max_length=256)
        for i, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            outs[i] = text
    return outs


def score(hyps, refs, n_boot, seed=0):
    chrf, bleu = CHRF(word_order=2), BLEU()
    point = (chrf.corpus_score(hyps, [refs]).score, bleu.corpus_score(hyps, [refs]).score)
    rng, n, boots = random.Random(seed), len(refs), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        h, r = [hyps[i] for i in idx], [refs[i] for i in idx]
        boots.append((chrf.corpus_score(h, [r]).score, bleu.corpus_score(h, [r]).score))
    ci = [(sorted(b[k] for b in boots)[int(.025 * n_boot)], sorted(b[k] for b in boots)[int(.975 * n_boot) - 1]) for k in (0, 1)]
    return point, ci


def systems_for(direction, src, all_prefixes=True):
    a, b = direction.split("-")
    target_prefix = ">>pol<< " if b == "pl" else ">>szl<< "
    yield "copy input", list(src)
    yield "NLLB-200-600M", run_nllb(src, a, b)
    yield f"base opus-mt-sla-sla ({target_prefix.strip()})", generate(BASE, src, target_prefix)
    prefixes = [">>pol<< ", ""] if direction == "szl-pl" else [">>pol<< ", ">>szl<< ", ""]
    if not all_prefixes:
        prefixes = [""]
    for p in prefixes:
        yield f"ours (prefix {p.strip() or 'none'})", generate(OWN[direction], src, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=list(TESTSETS), choices=list(TESTSETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--boot", type=int, default=N_BOOT, help="bootstrap resamples")
    ap.add_argument("--no-prefix-only", action="store_true", help="score only 'ours' without a prefix")
    args = ap.parse_args()
    summary_path = HERE / "summary.csv"
    summary = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    summary = [r for r in summary if r["test set"] not in args.sets]
    for name in args.sets:
        ids, szl, pl = TESTSETS[name](args.limit)
        rows = []
        for direction, (src, ref) in {"szl-pl": (szl, pl), "pl-szl": (pl, szl)}.items():
            for system, hyps in systems_for(direction, src, not args.no_prefix_only):
                for i, s, r, h in zip(ids, src, ref, hyps):
                    if name == "flores" and system == "copy input":
                        continue  # the hypothesis is the FLORES+ source text itself
                    row = {"id": i, "direction": direction, "system": system, "hypothesis": h}
                    if name != "flores":
                        row |= {"source": s, "reference": r}
                    rows.append(row)
                (chrf, bleu), (cc, bc) = score(hyps, ref, args.boot)
                summary.append({"test set": name, "direction": direction, "system": system, "chrF++": round(chrf, 1),
                                "chrF++ 95% CI": f"{cc[0]:.1f}-{cc[1]:.1f}", "BLEU": round(bleu, 1),
                                "BLEU 95% CI": f"{bc[0]:.1f}-{bc[1]:.1f}", "n": len(ref)})
                print(summary[-1], flush=True)
        pd.DataFrame(rows).to_csv(HERE / f"outputs_{name}.csv", index=False)
        pd.DataFrame(summary).to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
