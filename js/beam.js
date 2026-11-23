// Beam search for Marian ONNX models running under transformers.js (its built-in beam sampler behaves like greedy).
// Mirrors Hugging Face generate(): num_beams, length_penalty, bad_words_ids=[[pad]], forced EOS at max length.
export async function beamTranslate({ model, tokenizer, Tensor, text, numBeams = 6, maxLength = 256, lengthPenalty = 1.0 }) {
  const cfg = model.generation_config ?? {};
  const eos = cfg.eos_token_id ?? 0;
  const pad = cfg.pad_token_id ?? 38181;
  const start = cfg.decoder_start_token_id ?? pad;
  const enc = tokenizer(text);
  const L = enc.input_ids.dims[1];
  const tile = (t) => {
    const n = t.data.length;
    const d = new t.data.constructor(n * numBeams);
    for (let b = 0; b < numBeams; b++) d.set(t.data, b * n);
    return new Tensor(t.type, d, [numBeams, ...t.dims.slice(1)]);
  };
  const input_ids = tile(enc.input_ids);
  const attention_mask = tile(enc.attention_mask);

  let seqs = [[start]];
  let scores = [0];
  const finished = [];
  const lp = (score, len) => score / Math.pow(len, lengthPenalty);

  for (let step = 0; step < maxLength; step++) {
    const t = seqs[0].length;
    const nb = seqs.length;
    const dec = new BigInt64Array(numBeams * t);
    for (let b = 0; b < numBeams; b++) for (let i = 0; i < t; i++) dec[b * t + i] = BigInt(seqs[Math.min(b, nb - 1)][i]);
    const out = await model({ input_ids, attention_mask, decoder_input_ids: new Tensor("int64", dec, [numBeams, t]) });
    const V = out.logits.dims[2];
    const logits = out.logits.data;
    const cand = [];
    for (let b = 0; b < nb; b++) {
      const off = (b * t + (t - 1)) * V;
      let max = -Infinity;
      for (let v = 0; v < V; v++) if (v !== pad && logits[off + v] > max) max = logits[off + v];
      let sum = 0;
      for (let v = 0; v < V; v++) if (v !== pad) sum += Math.exp(logits[off + v] - max);
      const logZ = max + Math.log(sum);
      const forceEos = step === maxLength - 1;
      // keep top 2*numBeams per beam
      const top = [];
      for (let v = 0; v < V; v++) {
        if (v === pad || (forceEos && v !== eos)) continue;
        const s = scores[b] + logits[off + v] - logZ;
        if (top.length < 2 * numBeams || s > top[top.length - 1][0]) {
          top.push([s, b, v]);
          top.sort((x, y) => y[0] - x[0]);
          if (top.length > 2 * numBeams) top.pop();
        }
      }
      cand.push(...top);
    }
    cand.sort((x, y) => y[0] - x[0]);
    const nextSeqs = [], nextScores = [];
    for (let r = 0; r < cand.length && nextSeqs.length < numBeams; r++) {
      const [s, b, v] = cand[r];
      if (v === eos) {
        if (r < numBeams) finished.push([lp(s, seqs[b].length), seqs[b].slice(1)]);
      } else {
        nextSeqs.push([...seqs[b], v]);
        nextScores.push(s);
      }
    }
    seqs = nextSeqs; scores = nextScores;
    if (finished.length >= numBeams) {
      finished.sort((x, y) => y[0] - x[0]);
      // HF early_stopping=False heuristic: stop once the best running beam cannot beat the worst kept hypothesis.
      const worst = finished[numBeams - 1][0];
      if (lp(Math.max(...scores), seqs[0].length) < worst) break;
    }
    if (!seqs.length) break;
  }
  if (!finished.length) seqs.forEach((q, i) => finished.push([lp(scores[i], q.length), q.slice(1)]));
  finished.sort((x, y) => y[0] - x[0]);
  return tokenizer.decode(finished[0][1], { skip_special_tokens: true });
}
