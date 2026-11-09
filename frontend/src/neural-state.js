import { decodeActivationFrame, normalizeArchitecture } from './neural-protocol.js';

const terminal = new Set(['done', 'error', 'cancelled']);
export function createRunState(id = null) {
  return { id, phase: 'idle', lastEventId: 0, architecture: null, start: null, frames: [], index: -1, output: '', retainedBytes: 0, error: null };
}
/** Validate ordered events before changing an existing recording. Never retain base64 twice. */
export function applyNeuralEvent(state, kind, payload, eventId) {
  if (!Number.isInteger(eventId) || eventId < 1) throw new Error('Invalid stream event ID');
  if (eventId <= state.lastEventId) return false; // SSE reconnect duplicates are harmless.
  if (eventId !== state.lastEventId + 1) throw new Error('Missing stream event; reconnect from the last confirmed event');
  if (terminal.has(state.phase)) throw new Error('Event after terminal run state');
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error('Invalid stream payload');
  if (kind === 'start') {
    if (state.start) throw new Error('Duplicate stream start');
    const architecture = normalizeArchitecture(payload.architecture);
    if (payload.version !== 1 || payload.capture?.dtype !== 'float32' || payload.capture?.byte_order !== 'little' || payload.capture?.order !== 'layer-major: mlp then residual') throw new Error('Unsupported activation contract');
    state.architecture = architecture; state.start = payload; state.phase = 'running';
  } else if (kind === 'frame') {
    if (!state.start || state.frames.length >= 16 || payload.index !== state.frames.length) throw new Error('Invalid frame sequence');
    if (!Number.isInteger(payload.input_token_count) || payload.input_token_count < 1 || payload.input_token_count > 128 || !Number.isInteger(payload.token_id) || payload.token_id < 0 || typeof payload.token_text !== 'string' || payload.token_text.length > 400 || typeof payload.output !== 'string' || payload.output.length > 8000) throw new Error('Invalid token capture');
    for (const key of ['elapsed_ms', 'inference_ms', 'capture_ms']) if (!Number.isFinite(payload[key]) || payload[key] < 0) throw new Error('Invalid capture timing');
    const previous = state.frames.at(-1);
    if (previous && (payload.input_token_count !== previous.input_token_count + 1 || payload.elapsed_ms < previous.elapsed_ms || payload.input_token_id !== undefined && payload.input_token_id !== previous.token_id)) throw new Error('Invalid autoregressive context continuity');
    const activations = decodeActivationFrame(payload.activation_b64, state.architecture);
    if (state.retainedBytes + activations.byteLength > 16 * 1024 * 1024) throw new Error('Browser recording exceeds memory budget');
    const { activation_b64, ...metadata } = payload;
    state.frames.push({ ...metadata, activations }); state.retainedBytes += activations.byteLength; state.output = payload.output;
  } else if (terminal.has(kind)) {
    if (kind === 'done' && !state.start) throw new Error('Completed run has no start event');
    state.phase = kind; state.output = typeof payload.output === 'string' ? payload.output : state.output;
    if (kind === 'error') state.error = payload.message || payload.error || 'Inference failed';
  } else throw new Error('Unknown stream event type');
  state.lastEventId = eventId;
  return true;
}
