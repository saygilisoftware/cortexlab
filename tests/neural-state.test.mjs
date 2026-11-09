import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRunState, applyNeuralEvent } from '../frontend/src/neural-state.js';

// Explicit tiny transport fixture, not a pretrained inference recording.
const start = { version: 1, architecture: { layers: 1, d_model: 2, intermediate_size: 2 }, capture: { dtype: 'float32', byte_order: 'little', order: 'layer-major: mlp then residual' } };
const buffer = Buffer.alloc(16); [1, -2, 3, -4].forEach((v, i) => buffer.writeFloatLE(v, i * 4));
const first = { index: 0, token_id: 9, token_text: 'test', output: 'test', input_token_count: 3, input_token_id: 2, elapsed_ms: 1, inference_ms: .5, capture_ms: .2, activation_b64: buffer.toString('base64') };
test('stores raw signed activation data once and deduplicates reconnect events', () => {
  const state = createRunState('test'); applyNeuralEvent(state, 'start', start, 1); applyNeuralEvent(state, 'frame', first, 2);
  assert.deepEqual([...state.frames[0].activations], [1, -2, 3, -4]); assert.equal(state.retainedBytes, 16);
  assert.equal('activation_b64' in state.frames[0], false);
  assert.equal(applyNeuralEvent(state, 'frame', first, 2), false); assert.equal(state.frames.length, 1);
  applyNeuralEvent(state, 'done', { output: 'test' }, 3); assert.equal(state.phase, 'done');
});
test('invalid event does not advance cursor or overwrite a valid frame', () => {
  const state = createRunState(); applyNeuralEvent(state, 'start', start, 1);
  assert.throws(() => applyNeuralEvent(state, 'frame', { ...first, activation_b64: 'bad' }, 2));
  assert.equal(state.lastEventId, 1); assert.equal(state.frames.length, 0);
});
test('rejects missing events and changed autoregressive token continuity', () => {
  const state = createRunState(); assert.throws(() => applyNeuralEvent(state, 'start', start, 2));
  applyNeuralEvent(state, 'start', start, 1); applyNeuralEvent(state, 'frame', first, 2);
  assert.throws(() => applyNeuralEvent(state, 'frame', { ...first, index: 1, input_token_count: 4, input_token_id: 88 }, 3));
  assert.equal(state.frames.length, 1);
});
test('memory/initialization failure can be terminal without an invented start or activation', () => {
  const state = createRunState(); applyNeuralEvent(state, 'error', { message: 'RAM budget blocked loading' }, 1);
  assert.equal(state.phase, 'error'); assert.equal(state.frames.length, 0); assert.match(state.error, /RAM/);
});
test('rejects events after cancellation', () => {
  const state = createRunState(); applyNeuralEvent(state, 'cancelled', {}, 1);
  assert.throws(() => applyNeuralEvent(state, 'frame', first, 2));
});
