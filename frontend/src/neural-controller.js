import { parseWeightEnvelope } from './neural-protocol.js';
import { applyNeuralEvent, createRunState } from './neural-state.js';

const $ = id => document.getElementById(id);
const format = n => Intl.NumberFormat('en').format(n);
const architecture = { layers: 28, intermediate_size: 6144, d_model: 2048 };
const budgets = { eco: 50000, balanced: 250000, full: 2000000 };
const requestedQuality = new URLSearchParams(location.search).get('quality');
const initialQuality = Object.hasOwn(budgets, requestedQuality) ? requestedQuality : 'eco';
$('quality').value = initialQuality;
textQuality(initialQuality);
function textQuality(quality) { document.getElementById('quality-badge').textContent = `${quality.toUpperCase()} / ON DEMAND`; }
let scene = null, run = createRunState(), source = null, layer = 13, playing = false;
let weightEpoch = 0, weightAbort = null, runEpoch = 0, playTimer = null, alive = true;
const text = (id, value) => { $(id).textContent = value; };
function error(message) { $('error').hidden = !message; text('error', message || ''); }
function active() { return ['loading', 'running', 'cancelling'].includes(run.phase); }
function setBusy(value) { $('run').disabled = value || $('run').dataset.blocked === 'true'; $('cancel').disabled = !value; }
function showStats(stats) {
  text('unit-count', format(stats.totalUnits)); text('edge-count', format(stats.drawnEdges)); text('measured-count', format(stats.measuredUnits));
  text('sample-count', format(stats.sampledEdges)); text('passing-count', format(stats.thresholdPassingEdges));
  text('full-connections', format(stats.fullMatrixConnections || 12582912));
  text('render-stats', `${stats.drawCalls} calls · ${format(stats.drawnUnits)}/${format(stats.totalUnits)} sites · ≤${stats.frameCap} Hz · on demand`);
  if (stats.contextLost) error('The graphics context was lost. Reload this page in Eco mode; no model execution was restarted.');
  if (stats.renderError) error(`Rendering paused: ${stats.renderError}. Choose Eco or reload before retrying.`);
}
function showFrame(index) {
  if (!run.frames.length) return;
  run.index = Math.max(0, Math.min(index, run.frames.length - 1));
  const frame = run.frames[run.index]; scene?.setFrame(frame.activations, frame);
  $('timeline').value = run.index; text('step', `${run.index + 1} / ${run.frames.length}`);
  text('capture-time', `Input ${frame.input_token_count} tokens · +${frame.elapsed_ms.toFixed(1)} ms · actual last-token capture`);
  text('input-count', frame.input_token_count); text('forward-time', frame.inference_ms.toFixed(1));
  document.querySelectorAll('#token-stream span').forEach((span, i) => span.classList.toggle('active', i === run.index));
  text('mode', active() && $('follow').checked ? 'LIVE CAPTURE' : 'RECORDED FRAME');
  text('capture-notice', 'Measured last-input-token activations at all blocks. Eco displays a subset of sites; full raw values remain in the recording. Weight links are a separate learned matrix, not attention or a causal explanation.');
}
function renderRun() {
  text('connection-status', run.phase.toUpperCase()); setBusy(active());
  text('output', run.output || (active() ? 'Waiting for the next measured token…' : 'No generated output.'));
  $('timeline').max = Math.max(0, run.frames.length - 1);
  text('memory', `${(run.retainedBytes / 1024 ** 2).toFixed(1)} MiB browser recording`);
  $('token-stream').replaceChildren(...run.frames.map((frame, index) => {
    const item = document.createElement('span'); item.textContent = frame.token_text.replaceAll('\n', '↵').replaceAll(' ', '·');
    item.title = `Token ${frame.token_id}, frame ${index + 1}, forward ${frame.inference_ms.toFixed(1)} ms`;
    item.classList.toggle('active', index === run.index); return item;
  }));
  if (run.phase === 'error') error(run.error);
}
function stopPlayback() { playing = false; clearInterval(playTimer); playTimer = null; text('play', '▶'); }
function closeStream() { source?.close(); source = null; }
async function consumeEvent(kind, data, id, epoch) {
  if (!alive || epoch !== runEpoch) return;
  try {
    if (!applyNeuralEvent(run, kind, data, Number(id))) return;
    if (kind === 'start') scene?.setArchitecture(run.architecture);
    if (kind === 'frame' && $('follow').checked) showFrame(run.frames.length - 1);
    renderRun();
    if (['done', 'error', 'cancelled'].includes(kind)) { closeStream(); if (run.frames.length && run.index < 0) showFrame(0); }
  } catch (failure) {
    closeStream(); error(failure.message); run.phase = 'error'; run.error = failure.message; renderRun();
    // Stop future computation when an invalid stream cannot safely be consumed.
    if (run.id) fetch(`/api/neural/runs/${encodeURIComponent(run.id)}/cancel`, { method: 'POST' }).catch(() => {});
  }
}
async function startRun(event) {
  event?.preventDefault(); if (active() || $('run').dataset.blocked === 'true') return;
  stopPlayback(); closeStream(); error(null); const epoch = ++runEpoch;
  run = createRunState(); run.phase = 'loading'; renderRun();
  const body = { prompt: $('prompt').value.trim(), max_new_tokens: Number($('max-tokens').value) };
  try {
    const response = await fetch('/api/neural/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Run request was rejected');
    if (epoch !== runEpoch || !alive) return;
    run.id = data.id;
    source = new EventSource(data.events_url || `/api/neural/runs/${encodeURIComponent(data.id)}/events`);
    for (const kind of ['start', 'frame', 'done', 'error', 'cancelled']) {
      source.addEventListener(kind, event => {
        if (!event.data) return; // Native EventSource network errors auto-reconnect.
        try { consumeEvent(kind, JSON.parse(event.data), event.lastEventId, epoch); }
        catch (failure) { error(failure.message); closeStream(); }
      });
    }
    source.onopen = () => { if (active()) text('connection-status', 'CONNECTED'); };
    source.onerror = event => { if (!event.data && active()) text('connection-status', 'RECONNECTING · NO NEW JOB'); };
  } catch (failure) { run.phase = 'error'; run.error = failure.message; renderRun(); }
}
async function cancelRun() {
  if (!run.id || !active()) return;
  $('cancel').disabled = true; run.phase = 'cancelling'; renderRun();
  try {
    const response = await fetch(`/api/neural/runs/${encodeURIComponent(run.id)}/cancel`, { method: 'POST' });
    if (!response.ok) throw new Error('Cancellation could not be confirmed; reconnect to inspect the retained run.');
  } catch (failure) { error(failure.message); }
}
async function loadWeights() {
  const epoch = ++weightEpoch; weightAbort?.abort(); weightAbort = new AbortController();
  const requestedLayer = layer, budget = budgets[$('quality').value];
  text('block-label', `BLOCK ${String(layer + 1).padStart(2, '0')} · LOADING CPU SAMPLE`);
  try {
    const response = await fetch(`/api/neural/weights?layer=${requestedLayer}&budget=${budget}`, { signal: weightAbort.signal });
    if (!response.ok) { const data = await response.json(); throw new Error(data.detail || 'Weight sample unavailable'); }
    const parsed = parseWeightEnvelope(await response.arrayBuffer());
    if (epoch !== weightEpoch || !alive) return;
    if (parsed.header.layer !== requestedLayer) throw new Error('Weight sample belongs to a different block');
    scene?.setWeights(parsed); scene?.setLayer(requestedLayer); scene?.setThreshold(Number($('threshold').value));
    text('sample-count', format(parsed.header.count)); text('full-connections', format(parsed.header.total_connections));
    text('block-label', `BLOCK ${String(requestedLayer + 1).padStart(2, '0')} · MEASURED WEIGHTS`);
  } catch (failure) { if (failure.name !== 'AbortError' && epoch === weightEpoch) error(failure.message); }
}
function setLayer(value) {
  layer = value; scene?.setLayer(layer);
  document.querySelectorAll('[data-layer]').forEach(button => { const selected = Number(button.dataset.layer) === layer; button.classList.toggle('active', selected); button.setAttribute('aria-pressed', selected); });
  loadWeights();
}
function selectedUnit(pick) {
  text('selected-name', `Block ${pick.layer + 1} · ${pick.kind === 'mlp' ? 'MLP unit' : 'residual channel'} ${pick.unit}`);
  text('selected-value', pick.measured && Number.isFinite(pick.value) ? pick.value.toPrecision(7) : 'NOT CAPTURED');
  text('selected-detail', pick.kind === 'mlp' ? 'Gated SwiGLU product entering the down projection, for the last input token.' : 'Post-block residual channel before the final model RMSNorm, for the last input token.');
  if (pick.layer !== layer) setLayer(pick.layer);
}
async function resourceStatus() {
  const response = await fetch('/api/models'); if (!response.ok) throw new Error('Local API is unavailable');
  const qwen = (await response.json()).models.find(model => model.id === 'qwen3');
  const blocked = !qwen?.weights_available || !qwen.loaded && qwen.inference_allowed === false;
  $('run').dataset.blocked = String(blocked); setBusy(active()); $('runtime-status').classList.toggle('blocked', blocked);
  text('runtime-status', !qwen?.weights_available ? qwen?.reason || 'Local pretrained weights are missing.' : qwen.inference_allowed === false ? `CPU inference paused: ${(qwen.available_ram_bytes / 1024 ** 3).toFixed(1)} GiB RAM available; ${(qwen.minimum_cpu_available_bytes / 1024 ** 3).toFixed(0)} GiB required. Weights and recorded replay remain available. CUDA will not be used automatically.` : 'CPU inference available on explicit Run. Page startup does not load the model. Renderer defaults to low-power Eco mode.');
}
async function loadDemo() {
  const statusResponse = await fetch('/api/neural/status');
  if (!statusResponse.ok || !(await statusResponse.json()).demo_available) return;
  const response = await fetch('/api/neural/demo'); if (response.status === 404) return;
  if (!response.ok) throw new Error('Neural recording could not be read');
  const data = await response.json(); if (data.schema_version !== 'cortexlab.neural.demo/v1' || !Array.isArray(data.events) || data.events.length > 18) throw new Error('Unsupported neural demo recording');
  run = createRunState();
  for (const item of data.events) applyNeuralEvent(run, item.type, item.data, item.id);
  if (run.architecture) scene?.setArchitecture(run.architecture);
  renderRun(); if (run.frames.length) showFrame(0);
}

$('run-form').addEventListener('submit', startRun); $('cancel').onclick = cancelRun;
$('timeline').oninput = event => { stopPlayback(); $('follow').checked = false; showFrame(Number(event.target.value)); };
$('play').onclick = () => {
  if (playing) { stopPlayback(); return; }
  if (!run.frames.length) return;
  $('follow').checked = false; playing = true; text('play', 'Ⅱ');
  if (run.index === run.frames.length - 1) showFrame(0);
  playTimer = setInterval(() => { if (run.index >= run.frames.length - 1) stopPlayback(); else showFrame(run.index + 1); }, 850);
};
$('follow').onchange = () => { if ($('follow').checked && run.frames.length) { stopPlayback(); showFrame(run.frames.length - 1); } };
$('quality').onchange = () => { textQuality($('quality').value); scene?.setQuality($('quality').value); loadWeights(); };
$('edges').onchange = () => scene?.setEdgeVisibility($('edges').checked);
$('emphasis').onchange = () => scene?.setActivationEmphasis($('emphasis').checked);
$('threshold').oninput = () => { text('threshold-value', Number($('threshold').value).toFixed(3)); scene?.setThreshold(Number($('threshold').value)); };
$('focus-unit').onclick = () => scene?.focusSelection();
document.querySelectorAll('[data-camera]').forEach(button => button.onclick = () => { scene?.setCamera(button.dataset.camera); document.querySelectorAll('[data-camera]').forEach(item => item.classList.toggle('active', item === button)); });
$('blocks').replaceChildren(...Array.from({ length: 28 }, (_, index) => { const button = document.createElement('button'); button.textContent = String(index + 1).padStart(2, '0'); button.dataset.layer = index; button.type = 'button'; button.className = index === layer ? 'active' : ''; button.setAttribute('aria-label', `Select actual block ${index + 1}`); button.onclick = () => setLayer(index); return button; }));
window.addEventListener('pagehide', () => {
  alive = false;
  if (run.id && active()) fetch(`/api/neural/runs/${encodeURIComponent(run.id)}/cancel`, { method: 'POST', keepalive: true }).catch(() => {});
  closeStream(); weightAbort?.abort(); stopPlayback(); scene?.dispose();
});
window.cortexNeural = Object.freeze({ getStats: () => scene?.getStats(), getRun: () => run, getSelection: () => ({ layer }), dispose: () => scene?.dispose() });

async function start() {
  try { const { NeuralScene } = await import('./neural-scene.js'); scene = new NeuralScene($('viewport'), { onSelect: selectedUnit, onStats: showStats }); scene.setArchitecture(architecture); scene.setLayer(layer); scene.setQuality(initialQuality); scene.setCamera('focus'); }
  catch (failure) { $('renderer-error').hidden = false; error(failure.message); }
  try { await resourceStatus(); await loadWeights(); await loadDemo(); }
  catch (failure) { error(failure.message); }
}
start();
