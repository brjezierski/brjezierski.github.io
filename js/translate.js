import { AutoModelForSeq2SeqLM, AutoTokenizer, Tensor, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";
import { beamTranslate } from "./beam.js";

// Hugging Face model repo holding the quantized ONNX export (see onnx/convert.py), or a folder under /models/ via ?model=name for local testing.
const localName = new URLSearchParams(location.search).get("model");
const MODEL = localName ?? "brjezierski/opus-mt-pl-szl-onnx";
if (localName) { env.allowLocalModels = true; env.allowRemoteModels = false; env.localModelPath = "/models/"; }
// The fine-tune was evaluated with this language prefix; no space, so the tokenizer sees it as one special token.
const PREFIX = ">>pol<<";

const $ = (id) => document.getElementById(id);
const source = $("source");
const output = $("output");
let loading = null;

function setBusy(busy, message = "") {
  $("translateBtn").disabled = busy;
  $("spinner").classList.toggle("d-none", !busy);
  $("translateText").textContent = busy ? "Translating" : "Translate";
  $("status").textContent = message;
}

function showError(message) {
  $("error").textContent = message;
  $("error").classList.remove("d-none");
}

function loadModel() {
  loading ??= (async () => {
    const sizes = {};
    const progress_callback = (p) => {
      if (p.status !== "progress" || !p.file.endsWith(".onnx")) return;
      sizes[p.file] = [p.loaded, p.total];
      const [l, t] = Object.values(sizes).reduce((a, [x, y]) => [a[0] + x, a[1] + y], [0, 0]);
      $("status").textContent = `Downloading the model (first visit only): ${Math.round(l / 1e6)} / ${Math.round(t / 1e6)} MB`;
    };
    const [tokenizer, model] = await Promise.all([
      AutoTokenizer.from_pretrained(MODEL),
      AutoModelForSeq2SeqLM.from_pretrained(MODEL, { dtype: "q8", progress_callback }),
    ]);
    return { tokenizer, model };
  })().catch((e) => { loading = null; throw e; });
  return loading;
}

async function translate() {
  const text = source.value.trim();
  $("error").classList.add("d-none");
  if (!text) return;
  setBusy(true, "Loading the model, this takes a moment on the first visit.");
  try {
    const { tokenizer, model } = await loadModel();
    $("status").textContent = "";
    output.textContent = await beamTranslate({ model, tokenizer, Tensor, text: PREFIX + text });
    $("copyBtn").disabled = false;
  } catch (e) {
    showError("Translation failed. Please reload the page and try again.");
    console.error(e);
  } finally {
    setBusy(false);
  }
}

source.addEventListener("input", () => { $("count").textContent = source.value.length; });
source.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) translate(); });
$("translateBtn").addEventListener("click", translate);
// Start downloading as soon as the person begins typing, so the model is ready by the time they press Translate.
source.addEventListener("focus", () => loadModel().catch(() => {}), { once: true });

$("copyBtn").addEventListener("click", async () => {
  await navigator.clipboard.writeText(output.textContent);
  $("copyBtn").textContent = "Copied";
  setTimeout(() => { $("copyBtn").textContent = "Copy"; }, 1500);
});
