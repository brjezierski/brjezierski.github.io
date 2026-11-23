"""Export the Polish-to-Silesian MarianMT model to quantized ONNX for transformers.js.

Needs: torch, optimum-onnx[onnxruntime], transformers<5, sentencepiece, onnx.
Run `python onnx/convert.py out/` and upload `out/` to the Hugging Face model repo the site loads.
"""
import json
import os
import subprocess
import sys

import sentencepiece as spm
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnx.graph_transformations import merge_decoders

MODEL = "brjezierski/opus-mt-sla-sla-finetuned-pl-to-szl"
out = sys.argv[1] if len(sys.argv) > 1 else "out"
raw = os.path.join(out, "_raw")
os.makedirs(os.path.join(out, "onnx"), exist_ok=True)

subprocess.run(["optimum-cli", "export", "onnx", "--model", MODEL, "--task", "text2text-generation-with-past", raw], check=True)

# Per-channel int8 keeps quality close to fp32; per-tensor visibly degrades this model. The decoders are quantized
# separately and merged afterwards because quantizing the already-merged graph leaves its weights in fp32.
kw = dict(weight_type=QuantType.QInt8, per_channel=True, extra_options={"MatMulConstBOnly": True})
quantize_dynamic(f"{raw}/encoder_model.onnx", f"{out}/onnx/encoder_model_quantized.onnx", **kw)
for n in ("decoder_model", "decoder_with_past_model"):
    quantize_dynamic(f"{raw}/{n}.onnx", f"{raw}/q_{n}.onnx", **kw)
merge_decoders(f"{raw}/q_decoder_model.onnx", f"{raw}/q_decoder_with_past_model.onnx",
               save_path=f"{out}/onnx/decoder_model_merged_quantized.onnx", strict=False)

# transformers.js needs a tokenizer.json; Marian only ships two sentencepiece files and a shared vocab.json.
# Segmentation uses the source model's scores, so pieces that exist only in the target model get a score
# low enough never to be chosen (they stay in the vocab for decoding).
vocab = json.load(open(f"{raw}/vocab.json"))
sp = spm.SentencePieceProcessor(model_file=f"{raw}/source.spm")
scores = {sp.id_to_piece(i): sp.get_score(i) for i in range(sp.get_piece_size())}
special = ("</s>", "<unk>", "<pad>")
pieces = [["<missing%d>" % i, -99.0] for i in range(max(vocab.values()) + 1)]
for piece, i in vocab.items():
    pieces[i] = [piece, 0.0 if piece in special else scores.get(piece, -1000.0)]
added = [{"id": vocab[t], "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
         for t in (*special, ">>pol<<")]
tokenizer = {
    "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
    "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFKC"}, {"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}]},
    "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
    "post_processor": {"type": "TemplateProcessing",
                       "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
                       "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                       "special_tokens": {"</s>": {"id": "</s>", "ids": [vocab["</s>"]], "tokens": ["</s>"]}}},
    "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
    "model": {"type": "Unigram", "unk_id": vocab["<unk>"], "vocab": pieces, "byte_fallback": False},
}
json.dump(tokenizer, open(f"{out}/tokenizer.json", "w"), ensure_ascii=False)
for f in ("config.json", "generation_config.json", "special_tokens_map.json", "tokenizer_config.json"):
    os.replace(f"{raw}/{f}", f"{out}/{f}")
