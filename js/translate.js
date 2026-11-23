import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client@1.15.1/+esm";

// Space id, or a local Gradio URL such as http://127.0.0.1:7860 for testing.
const SPACE = "brjezierski/szl-pl-translator";
const BIDIRECTIONAL = true;

const $ = (id) => document.getElementById(id);
const source = $("source");
const output = $("output");
let direction = "szl-pl";
let clientPromise = null;

const getClient = () => (clientPromise ??= Client.connect(SPACE).catch((e) => { clientPromise = null; throw e; }));

function setBusy(busy) {
  $("translateBtn").disabled = busy;
  $("spinner").classList.toggle("d-none", !busy);
  $("translateText").textContent = busy ? "Translating" : "Translate";
  $("status").textContent = busy ? "Waking up the model, the first request can take up to a minute." : "";
}

function showError(message) {
  $("error").textContent = message;
  $("error").classList.remove("d-none");
}

async function translate() {
  const text = source.value.trim();
  $("error").classList.add("d-none");
  if (!text) return;
  setBusy(true);
  try {
    const client = await getClient();
    const result = await client.predict("/translate", { text, direction });
    output.textContent = result.data[0];
    $("copyBtn").disabled = false;
  } catch (e) {
    showError("Translation failed. The model may be starting up, please try again in a moment.");
    console.error(e);
  } finally {
    setBusy(false);
  }
}

source.addEventListener("input", () => { $("count").textContent = source.value.length; });
source.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) translate(); });
$("translateBtn").addEventListener("click", translate);

$("copyBtn").addEventListener("click", async () => {
  await navigator.clipboard.writeText(output.textContent);
  $("copyBtn").textContent = "Copied";
  setTimeout(() => { $("copyBtn").textContent = "Copy"; }, 1500);
});

if (BIDIRECTIONAL) {
  $("swapBtn").classList.remove("d-none");
  $("swapBtn").addEventListener("click", () => {
    direction = direction === "szl-pl" ? "pl-szl" : "szl-pl";
    const [src, dst] = direction === "szl-pl" ? ["Silesian", "Polish"] : ["Polish", "Silesian"];
    $("srcLabel").textContent = src;
    $("dstLabel").textContent = dst;
    source.value = output.textContent === "Translation" ? "" : output.textContent;
    output.innerHTML = '<span class="text-body-secondary">Translation</span>';
    $("copyBtn").disabled = true;
    $("count").textContent = source.value.length;
  });
}
