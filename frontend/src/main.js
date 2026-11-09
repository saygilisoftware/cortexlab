import { validateSession } from './contracts.js';

const $ = id => document.getElementById(id);
const state = { session: null, baseline: null, frame: 0, layer: 1, head: 0, token: 0, playing: false, busy: false, scene: null, lastPlayback: 0, backend: 'tiny', models: [] };
const pad = value => String(value + 1).padStart(2, '0');
const text = (id, value) => { $(id).textContent = value; };
const frame = () => state.session?.frames[state.frame];
const node = (tag, className, content) => { const element = document.createElement(tag); if (className) element.className = className; if (content !== undefined) element.textContent = content; return element; };
const labelToken = token => token === '<bos>' ? '⟨bos⟩' : token === '<eos>' ? '⟨eos⟩' : token.replaceAll('\n', '↵').replaceAll('\t', '⇥').replaceAll(' ', '·');
const isSubset = () => state.session?.schema_version === 'cortexlab.session/v2';
const sourceLayer = (index = state.layer) => frame()?.layers[index]?.index ?? index;
const sourceHead = (index = state.head) => state.session?.capture.head_indices?.[index] ?? index;
function backendStatus() {
  const entry = state.models.find(m => m.id === state.backend);
  const qwen = state.backend === 'qwen3';
  $('backend-status').classList.toggle('unavailable', qwen && !entry?.weights_available);
  const blocked = qwen && !entry?.loaded && (!entry?.weights_available || entry.inference_allowed === false);
  $('run').dataset.blocked = String(blocked); $('run').disabled = state.busy || blocked;
  text('backend-status', qwen ? entry?.loaded ? `Qwen loaded on ${entry.device?.toUpperCase() || 'local device'} · ready for inference` : entry?.weights_available ? entry.inference_allowed === false ? 'Replay available. CPU inference blocked by RAM budget; automatic CUDA is disabled.' : 'Local weights available · CPU mode · loads only on explicit Run' : entry?.reason || 'Checking local Qwen weights…' : 'Original 158.9K-parameter offline fixture · CPU');
  for (const option of $('new-tokens').options) option.disabled = qwen && Number(option.value) > 4;
  if (qwen && Number($('new-tokens').value) > 4) $('new-tokens').value = '4';
  $('ablate').disabled = qwen; if (qwen) $('ablate').checked = false;
  $('ablate').closest('label').title = qwen ? 'Head ablation is available on Cortex Tiny; not implemented for this Qwen adapter.' : '';
}
async function refreshModels() {
  const response = await fetch('/api/models');
  if (response.ok) state.models = (await response.json()).models;
  backendStatus();
}
async function chooseBackend(value) {
  state.backend = value; $('model-select').value = value; setPlaying(false); backendStatus();
  const qwen = value === 'qwen3', entry = state.models.find(m => m.id === value);
  const prompts = qwen ? ['Name the capital of France. Answer with one word.', 'What is attention? Answer briefly.', 'Complete: The sky is'] : ['a comet is cyan . the comet', 'attention mixes', 'a transformer predicts the'];
  document.querySelectorAll('[data-prompt]').forEach((button, i) => { button.dataset.prompt = prompts[i]; button.textContent = qwen ? ['capital', 'attention', 'complete'][i] : ['model', 'attention', 'transformer'][i]; });
  $('prompt').value = prompts[0]; updatePromptCount();
  if (entry?.recording_available || !qwen) {
    try { const response = await fetch(`/api/demo?model=${value}`); if (!response.ok) throw new Error('Recorded demo unavailable'); loadSession(await response.json()); }
    catch (error) { status(error.message, true); }
  } else status(qwen ? 'Qwen backend selected. Run inference to replace the current recording with real pretrained measurements.' : 'Tiny model selected. Run inference.');
}
function status(message, error = false) { text('run-status', message); $('run-status').classList.toggle('error', error); }
function setPlaying(value) {
  state.playing = value; state.lastPlayback = performance.now();
  text('play', value ? 'Ⅱ' : '▶'); $('play').setAttribute('aria-label', value ? 'Pause recording' : 'Play recording');
}
function select(values) {
  if (!state.session) return;
  Object.assign(state, values);
  state.layer = Math.max(0, Math.min(state.layer, frame().layers.length - 1));
  state.head = Math.max(0, Math.min(state.head, frame().layers[state.layer].attention.length - 1));
  state.token = Math.max(0, Math.min(state.token, frame().tokens.length - 1));
  render();
}
function setStep(value) {
  if (!state.session) return;
  const wasLast = state.token === frame().tokens.length - 1;
  state.frame = Math.max(0, Math.min(Number(value), state.session.frames.length - 1));
  if (wasLast) state.token = frame().tokens.length - 1;
  select({});
}
function renderModel() {
  const m = state.session.model;
  text('model-name', m.name.split(' · ')[0]);
  text('trained-badge', isSubset() ? 'PRETRAINED' : m.trained ? 'TRAINED FIXTURE' : 'UNTRAINED');
  text('model-description', `${state.session.runtime.device.toUpperCase()} · causal decoder · ${state.session.runtime.dtype || state.session.capture.dtype || 'float32'}`);
  text('param-count', Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(m.parameter_count));
  text('context-limit', m.context_limit); text('model-layers', m.layers); text('model-heads', m.heads);
  const t = m.training || {};
  text('training-info', isSubset() ? `${t.scope} License: ${t.license}. Displaying ${frame().layers.length} of ${m.layers} actual blocks, ${state.session.capture.head_indices.length} of ${m.heads} attention heads, and ${state.session.capture.channel_indices.length} of ${m.d_model} raw residual channels. Node norms use the full model width.` : `${t.scope || 'See model metadata for training scope.'} ${t.steps ?? 'Unknown'} training steps; ${t.train_sequences ?? '?'} training / ${t.heldout_sequences ?? '?'} held-out sequences. Held-out accuracy ${t.final_heldout ? (t.final_heldout.accuracy * 100).toFixed(2) + '%' : 'unreported'}; majority baseline ${t.baseline ? (t.baseline.heldout_accuracy * 100).toFixed(2) + '%' : 'unreported'}. Seed ${m.seed}.`);
  const fingerprint = m.checkpoint_sha256 || m.artifacts?.files?.find(file => file.file === 'config.json')?.sha256;
  text('model-hash', `${isSubset() ? 'Config SHA-256' : 'Checkpoint SHA-256'} ${fingerprint || 'See exported artifact manifest'}`);
  text('model-scope', isSubset() ? 'Pretrained Qwen · selected real tensor slices' : 'Educational model · domain-limited vocabulary');
  $('layer-list').replaceChildren(...frame().layers.map((captured, layer) => {
    const button = node('button', 'layer-button'); button.type = 'button'; button.dataset.layer = layer;
    button.append(node('span', 'layer-number', pad(captured.index)), node('span', '', 'Transformer block'), node('span', '', `${m.heads}H / ${m.d_model}D`));
    button.onclick = () => select({ layer }); return button;
  }));
  $('head-list').replaceChildren(...Array.from({ length: frame().layers[0].attention.length }, (_, head) => {
    const button = node('button', '', pad(sourceHead(head))); button.type = 'button'; button.dataset.head = head;
    button.setAttribute('aria-label', `Attention head ${sourceHead(head) + 1}`); button.onclick = () => select({ head }); return button;
  }));
}
function loadSession(data, source = 'recorded') {
  const validated = validateSession(data);
  setPlaying(false);
  state.session = validated; state.frame = 0;
  state.layer = Math.min(state.layer, validated.frames[0].layers.length - 1); state.head = Math.min(state.head, validated.frames[0].layers[0].attention.length - 1);
  state.token = validated.frames[0].tokens.length - 1;
  if (!validated.request.ablate) state.baseline = validated;
  $('prompt').value = validated.request.prompt; updatePromptCount();
  state.backend = validated.schema_version === 'cortexlab.session/v2' ? 'qwen3' : 'tiny'; $('model-select').value = state.backend; backendStatus();
  $('new-tokens').value = String(validated.request.max_new_tokens);
  $('ablate').checked = !!validated.request.ablate;
  if (validated.request.ablate) { state.layer = validated.request.ablate.layer; state.head = validated.request.ablate.head; }
  $('timeline').max = validated.frames.length - 1;
  $('timeline-ticks').replaceChildren(...validated.frames.map((f, i) => node('span', '', pad(i))));
  renderModel(); render();
  state.scene?.setCamera('overview');
  text('view-mode', source === 'inference' ? 'CAPTURED' : source === 'import' ? 'IMPORTED' : 'RECORDED');
  status(`${source === 'inference' ? 'Inference captured' : source === 'import' ? 'Recording imported' : 'Real inference demo loaded'} · ${validated.frames.length} steps · ${validated.frames[0].tokens.length} input tokens`);
}
function render() {
  const f = frame(); if (!f) return;
  const layer = f.layers[state.layer], row = layer.attention[state.head][state.token];
  const token = f.tokens[state.token], norm = layer.norms[state.token];
  text('selected-layer', `BLOCK ${pad(sourceLayer())}`); text('query-token', labelToken(token)); text('token-index', `POSITION ${String(state.token).padStart(2, '0')}`);
  text('ablate-layer', pad(sourceLayer())); text('ablate-head', pad(sourceHead())); text('norm-value', `L2 ${norm.toFixed(4)}`);
  document.querySelectorAll('[data-layer]').forEach(button => { const active = Number(button.dataset.layer) === state.layer; button.classList.toggle('selected', active); button.setAttribute('aria-pressed', active); });
  document.querySelectorAll('[data-head]').forEach(button => { const active = Number(button.dataset.head) === state.head; button.classList.toggle('selected', active); button.setAttribute('aria-pressed', active); });
  $('token-list').replaceChildren(...f.tokens.map((value, tokenIndex) => {
    const button = node('button', tokenIndex === state.token ? 'selected' : '', labelToken(value)); button.type = 'button'; button.dataset.token = tokenIndex;
    button.setAttribute('aria-label', `Query position ${tokenIndex}: ${value}`); button.setAttribute('aria-pressed', tokenIndex === state.token);
    button.append(node('small', '', tokenIndex)); button.onclick = () => select({ token: tokenIndex }); return button;
  }));
  $('attention-list').replaceChildren(...row.map((weight, key) => {
    const button = node('button', `attention-row${key === state.token ? ' selected' : ''}${key > state.token ? ' masked' : ''}`); button.type = 'button'; button.dataset.key = key;
    button.setAttribute('aria-label', `${f.tokens[key]}, key ${key}, attention ${weight.toFixed(6)}`);
    const label = node('span', 'key-token'); label.append(node('small', '', String(key).padStart(2, '0')), document.createTextNode(labelToken(f.tokens[key])));
    const track = node('span', 'bar-track'), fill = node('span', 'bar-fill'); fill.style.width = `${weight * 100}%`; track.append(fill);
    const number = node('span', 'weight', weight.toFixed(3)); number.dataset.weight = weight;
    button.append(label, track, number); button.title = `P(key=${key} | query=${state.token}, head=${sourceHead()}, layer=${sourceLayer()}) = ${weight}`;
    button.onclick = () => select({ token: key }); return button;
  }));
  const channels = layer.residual[state.token], scale = Math.max(...channels.map(Math.abs), 1e-6);
  $('activation-bars').replaceChildren(...channels.map((value, index) => {
    const bar = node('span', 'activation-channel'), fill = node('i'); const height = Math.abs(value) / scale * 48;
    fill.style.height = `${height}%`; fill.style.background = value >= 0 ? 'var(--cyan)' : 'var(--violet)';
    if (value >= 0) fill.style.bottom = '50%'; else fill.style.top = '50%';
    const actualChannel = state.session.capture.channel_indices?.[index] ?? index;
    bar.append(fill); bar.title = `Post-block residual channel ${actualChannel}: ${value}`; return bar;
  }));
  text('channel-count', isSubset() ? `${channels.length}/${state.session.model.d_model} raw channels` : `${channels.length} channels · per-vector scale`);
  text('activation-note', isSubset() ? `Showing ${channels.length} measured channels from the ${state.session.model.d_model}-channel post-block vector. L2 and node size use all ${state.session.model.d_model} channels. Blocks and heads are selected slices.` : 'Post-block residual vector. Nodes represent token vectors; positions are schematic.');
  $('prediction-list').replaceChildren(...f.top_predictions.slice(0, 5).map(p => { const row = node('div', 'prediction-row'); row.append(node('b', '', labelToken(p.token)), node('span', '', `${(p.probability * 100).toFixed(2)}%`)); return row; }));
  $('timeline').value = state.frame; text('frame-label', `STEP ${pad(state.frame)} / ${String(state.session.frames.length).padStart(2, '0')}`);
  text('capture-time', `Captured: +${f.elapsed_ms.toFixed(1)} ms`);
  text('inference-time', `${state.session.metrics.model_ms.toFixed(1)} ms forward · ${state.session.metrics.capture_ms.toFixed(1)} ms capture`);
  const input = node('span', 'prompt-output', state.session.request.prompt + ' ');
  const continuation = node('span', 'generated-output');
  state.session.frames.forEach((step, i) => { if (step.predicted_token === '<eos>' || isSubset() && step.predicted_token === '<|im_end|>') return; const word = node('span', i === state.frame ? 'predicted-current' : '', `${step.predicted_token}${!isSubset() && i < state.session.frames.length - 1 ? ' ' : ''}`); continuation.append(word); });
  $('output').replaceChildren(input, continuation);
  const unknowns = f.tokens.filter(t => t === '<unk>').length;
  $('unknown-warning').hidden = !unknowns;
  if (unknowns) text('unknown-warning', `${unknowns} token(s) mapped to <unk>. This model has a fixed ${state.session.model.vocab_size}-token educational vocabulary.`);
  const baseline = state.baseline;
  const comparable = state.session.request.ablate && baseline && baseline.model.checkpoint_sha256 === state.session.model.checkpoint_sha256 && baseline.frames[0].input_ids.join(',') === state.session.frames[0].input_ids.join(',');
  $('comparison').hidden = !comparable;
  if (comparable) {
    const first = state.session.frames[0], b = baseline.frames[0], q = Math.min(state.token, first.tokens.length - 1);
    const delta = first.layers[state.layer].norms[q] - b.layers[state.layer].norms[q];
    text('comparison', `Intervention vs baseline · first-frame prediction: ${b.predicted_token} → ${first.predicted_token} · residual L2 Δ ${delta >= 0 ? '+' : ''}${delta.toFixed(4)} at L${state.layer + 1}, token ${q}.`);
  }
  const threshold = Number($('threshold').value);
  text('threshold-value', threshold.toFixed(2));
  text('scene-label', `POST-BLOCK RESIDUALS · L${sourceLayer() + 1} / H${sourceHead() + 1}${isSubset() ? ' · SELECTED MODEL SLICES' : ''}\n${row.filter(w => w > 0 && w >= threshold).length} of ${row.length} attention links shown`);
  state.scene?.setFrame(f, { layer: state.layer, head: state.head, displayHead: sourceHead(), token: state.token, threshold, showAttention: $('show-attention').checked });
  if (!$('matrix-panel').hidden) drawMatrix();
}
function drawMatrix() {
  const f = frame(); if (!f) return;
  const canvas = $('matrix'), n = f.tokens.length, cell = Math.min(28, 360 / n), left = 70, top = 28;
  const size = Math.ceil(left + n * cell + 10), height = Math.ceil(top + n * cell + 12);
  canvas.width = size * 2; canvas.height = height * 2;
  const ctx = canvas.getContext('2d'); ctx.scale(2, 2); ctx.fillStyle = '#101d2d'; ctx.fillRect(0, 0, size, height);
  ctx.font = '8px monospace';
  const matrix = f.layers[state.layer].attention[state.head];
  matrix.forEach((row, q) => {
    ctx.fillStyle = q === state.token ? '#f2ce8e' : '#8ca8c3'; ctx.textAlign = 'right'; ctx.fillText(`${q} ${labelToken(f.tokens[q])}`, left - 5, top + q * cell + cell * .65, left - 8);
    row.forEach((weight, k) => {
      ctx.fillStyle = k > q ? '#0c1522' : `rgb(${Math.round(20 + weight * 100)},${Math.round(40 + weight * 180)},${Math.round(60 + weight * 170)})`;
      ctx.fillRect(left + k * cell + 1, top + q * cell + 1, cell - 2, cell - 2);
    });
    if (q === state.token) { ctx.strokeStyle = '#f2ce8e'; ctx.lineWidth = .7; ctx.strokeRect(left, top + q * cell, n * cell, cell); }
  });
  ctx.fillStyle = '#809ab6'; ctx.textAlign = 'center'; f.tokens.forEach((_, k) => ctx.fillText(k, left + k * cell + cell / 2, 18));
  canvas.onclick = event => { const r = canvas.getBoundingClientRect(); const y = (event.clientY - r.top) * height / r.height; const token = Math.floor((y - top) / cell); if (token >= 0 && token < n) select({ token }); };
}

async function infer(event) {
  event.preventDefault(); if (state.busy) return;
  state.busy = true; setPlaying(false); $('run').disabled = true; $('model-select').disabled = true; $('run').firstElementChild.textContent = 'Capturing tensors…'; status(state.backend === 'qwen3' ? 'Loading local Qwen weights if needed; capturing real CUDA attention and residuals…' : 'Running a real PyTorch forward pass for each generation step…');
  const body = { prompt: $('prompt').value.trim(), max_new_tokens: Number($('new-tokens').value), model: state.backend, ablate: $('ablate').checked ? { layer: sourceLayer(), head: sourceHead() } : null };
  const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), state.backend === 'qwen3' ? 180000 : 30000);
  try {
    const response = await fetch('/api/infer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: controller.signal });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Invalid request. Check prompt length and generation limits.');
    loadSession(data, 'inference');
    refreshModels().catch(() => {});
  } catch (error) { status(error.name === 'AbortError' ? 'Inference timed out. The server may still be finishing this recording.' : error.message, true); }
  finally { clearTimeout(timer); state.busy = false; backendStatus(); $('model-select').disabled = false; $('run').firstElementChild.textContent = 'Run inference'; }
}
function updatePromptCount() { text('prompt-count', `${$('prompt').value.length} / 400`); }
$('inference-form').addEventListener('submit', infer); $('prompt').addEventListener('input', updatePromptCount);
$('model-select').onchange = event => chooseBackend(event.target.value);
document.querySelectorAll('[data-prompt]').forEach(button => button.onclick = () => { $('prompt').value = button.dataset.prompt; updatePromptCount(); $('prompt').focus(); });
document.querySelectorAll('[data-camera]').forEach(button => button.onclick = () => { state.scene?.setCamera(button.dataset.camera); document.querySelectorAll('[data-camera]').forEach(b => b.classList.toggle('selected', b === button)); });
$('focus').onclick = () => state.scene?.focusSelection();
$('threshold').oninput = render; $('show-attention').onchange = render;
$('timeline').oninput = event => { setPlaying(false); setStep(event.target.value); };
$('play').onclick = () => { if (!state.session) return; if (!state.playing && state.frame === state.session.frames.length - 1) setStep(0); setPlaying(!state.playing); };
function animate(time) {
  if (state.playing && state.session && time - state.lastPlayback >= 1300 / Number($('speed').value)) {
    state.lastPlayback = time;
    if (state.frame >= state.session.frames.length - 1) setPlaying(false); else setStep(state.frame + 1);
  }
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
document.addEventListener('keydown', event => {
  if (['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(event.target.tagName) || $('help').open) return;
  if (event.code === 'Space') { event.preventDefault(); $('play').click(); }
  if (event.code === 'ArrowRight') { setPlaying(false); setStep(state.frame + 1); }
  if (event.code === 'ArrowLeft') { setPlaying(false); setStep(state.frame - 1); }
});
function toggleMatrix(visible) { $('matrix-panel').hidden = !visible; $('matrix-toggle').setAttribute('aria-expanded', visible); if (visible) drawMatrix(); }
$('matrix-toggle').onclick = () => toggleMatrix($('matrix-panel').hidden); $('matrix-close').onclick = () => toggleMatrix(false);
$('export-session').onclick = () => {
  if (!state.session) return;
  // Compact output keeps maximum supported captures inside the same 5 MiB import budget.
  const blob = new Blob([JSON.stringify(state.session)], { type: 'application/json' }), url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = `cortexlab-${state.session.id.slice(0, 12)}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$('session-file').onchange = async event => {
  const file = event.target.files[0]; if (!file) return;
  try { if (file.size > 5 * 1024 * 1024) throw new Error('Recording exceeds the 5 MiB import limit.'); loadSession(JSON.parse(await file.text()), 'import'); }
  catch (error) { status(error.message, true); }
  finally { event.target.value = ''; }
};
document.querySelector('.import-button').onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('session-file').click(); } };
$('help-toggle').onclick = () => { $('help').showModal(); $('help-toggle').setAttribute('aria-expanded', 'true'); };
$('help-close').onclick = () => $('help').close(); $('help').onclose = () => $('help-toggle').setAttribute('aria-expanded', 'false');
$('help').onclick = event => { if (event.target === $('help')) { const r = $('help').getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) $('help').close(); } };

// Read-only diagnostics make real browser verification reproducible; no fabricated FPS.
window.cortexlab = Object.freeze({ getSession: () => state.session, getSelection: () => ({ frame: state.frame, layer: state.layer, head: state.head, token: state.token }), getSourceSelection: () => ({ layer: sourceLayer(), head: sourceHead(), token: state.token }), getStats: () => state.scene?.getStats() || null, getScene: () => state.scene });
setInterval(() => { const stats = state.scene?.getStats(); if (stats?.fps) text('render-stats', `${Math.round(stats.fps)} FPS · ${stats.drawCalls} calls · ${stats.nodes} nodes`); }, 1000);

async function start() {
  // Import the renderer independently so a WebGL failure does not erase the tensor inspector.
  try {
    const { ObservatoryScene } = await import('./scene.js');
    state.scene = new ObservatoryScene($('viewport'), {
      onSelect: pick => select(pick),
      onHover: pick => {
        const card = $('hover-card'); card.hidden = !pick || !frame(); if (card.hidden) return;
        const f = frame(); card.textContent = `${labelToken(f.tokens[pick.token])} · position ${pick.token} · block ${f.layers[pick.layer].index + 1} · L2 ${f.layers[pick.layer].norms[pick.token].toFixed(4)}`;
        card.style.left = `${Math.min(pick.x + 12, window.innerWidth - 280)}px`; card.style.top = `${Math.max(8, pick.y - 44)}px`;
      }
    });
  } catch (error) { $('scene-fallback').hidden = false; text('scene-error', error.message); text('render-stats', 'Tensor inspection available · 3D offline'); console.error('3D initialization failed:', error); }
  try {
    await refreshModels();
    const desired = new URLSearchParams(location.search).get('model') === 'qwen3' ? 'qwen3' : 'tiny';
    await chooseBackend(desired);
  } catch (error) { status(error.message, true); }
  updatePromptCount();
}
start();
