"""Gradio app serving the Silesian-Polish MarianMT models for the portfolio site."""
import gradio as gr
import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL_IDS = {
    "szl-pl": "brjezierski/opus-mt-sla-sla-finetuned-szl-to-pl",
    "pl-szl": "brjezierski/opus-mt-sla-sla-finetuned-pl-to-szl",
}
MAX_CHARS = 500
# Both fine-tunes were checked with the ">>pol<<" prefix: szl-pl needs it, pl-szl is no worse with it.
PREFIX = ">>pol<< "

tokenizers = {d: MarianTokenizer.from_pretrained(m) for d, m in MODEL_IDS.items()}
models = {d: MarianMTModel.from_pretrained(m).eval() for d, m in MODEL_IDS.items()}


def translate(text: str, direction: str = "szl-pl") -> str:
    """Translate text; direction is 'szl-pl' or 'pl-szl'."""
    text = (text or "").strip()
    if not text:
        return ""
    if direction not in MODEL_IDS:
        raise gr.Error(f"Unknown direction: {direction}")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text is too long ({len(text)} characters, maximum {MAX_CHARS}).")
    batch = tokenizers[direction]([PREFIX + text], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = models[direction].generate(**batch)
    return tokenizers[direction].decode(out[0], skip_special_tokens=True)


demo = gr.Interface(
    fn=translate,
    inputs=[gr.Textbox(label="Text", lines=4), gr.Dropdown(["szl-pl", "pl-szl"], value="szl-pl", label="Direction")],
    outputs=gr.Textbox(label="Translation"),
    api_name="translate",
    title="Silesian-Polish translator",
)

if __name__ == "__main__":
    demo.launch()
