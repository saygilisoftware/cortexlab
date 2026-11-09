import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { validateSession } from '../frontend/src/contracts.js';

// Hand-authored numerical CONTRACT fixture. These values are intentionally not
// represented as captured Qwen weights, activations, inference or a demo.
function fixture() {
  const layerIndices = [0, 13, 27];
  const frames = [0, 1].map(index => {
    const inputIds = index === 0 ? [11, 12] : [11, 12, 13];
    const tokens = index === 0 ? ['a', 'b'] : ['a', 'b', 'c'];
    const predictedId = index === 0 ? 13 : 14;
    const predictedToken = index === 0 ? 'c' : 'd';
    const length = inputIds.length;
    return {
      index, elapsed_ms: index + 1, inference_ms: 0.5,
      input_ids: inputIds, tokens, predicted_id: predictedId,
      predicted_token: predictedToken,
      top_predictions: [{ id: predictedId, token: predictedToken, probability: 0.2 }, { id: 15, token: 'e', probability: 0.1 }],
      layers: layerIndices.map(layerIndex => ({
        index: layerIndex,
        attention: Array.from({ length: 4 }, () => Array.from({ length }, (_, query) => Array.from({ length }, (_, key) => key <= query ? 1 / (query + 1) : 0))),
        residual: Array.from({ length }, () => Array(64).fill(0.125)),
        // Sample L2 is 1; full-width residual L2 can be larger, and is supplied
        // independently. Comparing these norms for equality would be wrong.
        norms: Array(length).fill(4),
      })),
    };
  });
  return {
    schema_version: 'cortexlab.session/v2', id: 'numerical-contract-test-only',
    created_at: '2026-11-06T12:00:00Z',
    model: { name: 'Contract fixture only; no model execution', layers: 28, heads: 16, d_model: 2048, context_limit: 64, vocab_size: 151936, trained: true, tokenizer: { kind: 'sparse token labels' } },
    request: { prompt: 'a b', max_new_tokens: 2, ablate: null },
    runtime: { device: 'contract-test' },
    capture: { layer_indices: [0, 13, 27], head_indices: [0, 5, 10, 15], channel_indices: Array.from({ length: 64 }, (_, i) => i) },
    frames, metrics: { total_ms: 2, model_ms: 1, capture_ms: 1 }, output: 'c d',
  };
}

test('v2 validates sparse capture with actual model axes and independent full-width norm', () => {
  const value = fixture();
  assert.equal(validateSession(value), value);
  assert.deepEqual(validateSession(JSON.parse(JSON.stringify(value))), value);
});

const actualRecording = new URL('../examples/qwen-demo-session.json', import.meta.url);
test('genuine pretrained Qwen recording validates without changing measured tensors', { skip: !existsSync(actualRecording) }, () => {
  const value = JSON.parse(readFileSync(actualRecording, 'utf8'));
  const before = structuredClone(value);
  assert.equal(value.schema_version, 'cortexlab.session/v2');
  assert.equal(value.model.parameter_count, 1720574976);
  assert.deepEqual(value.capture.layer_indices, [0, 13, 27]);
  assert.deepEqual(value.capture.head_indices, [0, 5, 10, 15]);
  assert.equal(validateSession(value), value);
  assert.deepEqual(value, before);
});

test('v2 preserves declared BF16 source row rounding without renormalizing', () => {
  const value = fixture();
  value.runtime.dtype = 'bfloat16';
  value.capture.source_dtype = 'bfloat16';
  value.capture.dtype = 'float32';
  const rounded = [0.50390625, 0.5];
  value.frames[0].layers[0].attention[0][1] = rounded;
  assert.equal(validateSession(value), value);
  assert.deepEqual(value.frames[0].layers[0].attention[0][1], rounded);
});

const bad = (name, mutate) => test(name, () => {
  const value = fixture(); mutate(value);
  assert.throws(() => validateSession(value), /Invalid session/);
});

bad('v2 rejects duplicate captured layers', s => s.capture.layer_indices = [0, 13, 13]);
bad('v2 rejects unordered captured heads', s => s.capture.head_indices = [0, 10, 5, 15]);
bad('v2 rejects captured layer outside full model', s => s.capture.layer_indices[2] = 28);
bad('v2 rejects captured head outside full model', s => s.capture.head_indices[3] = 16);
bad('v2 rejects captured channel outside full model', s => s.capture.channel_indices[63] = 2048);
bad('v2 rejects layer identity inconsistent with capture metadata', s => s.frames[0].layers[1].index = 1);
bad('v2 rejects sampled channel shape mismatch', s => s.frames[0].layers[0].residual[0].pop());
bad('v2 rejects sample norm exceeding full residual norm', s => s.frames[0].layers[0].norms[0] = 0.1);
bad('v2 rejects non-finite residual sample', s => s.frames[0].layers[0].residual[0][0] = Infinity);
bad('v2 rejects missing captured attention head', s => s.frames[0].layers[0].attention.pop());
bad('v2 rejects causal attention violation', s => s.frames[0].layers[0].attention[0][0] = [0.5, 0.5]);
bad('v2 rejects attention row without normalized mass', s => s.frames[0].layers[0].attention[0][1] = [0.2, 0.3]);
bad('v2 rejects greedy prediction mismatch', s => s.frames[0].top_predictions[0] = { id: 15, token: 'e', probability: 0.3 });
bad('v2 rejects unordered prediction probabilities', s => s.frames[0].top_predictions.reverse());
bad('v2 rejects predicted ID outside full vocabulary', s => s.frames[0].predicted_id = 151936);
bad('v2 rejects inconsistent decoded label for an observed ID', s => s.frames[1].tokens[0] = 'changed');
bad('v2 rejects frame context not extended by prior prediction', s => { s.frames[1].input_ids[2] = 15; s.frames[1].tokens[2] = 'e'; });
bad('v2 rejects backward elapsed time', s => s.frames[1].elapsed_ms = 0);
bad('v2 rejects generation beyond bounded adapter limit', s => s.request.max_new_tokens = 5);
bad('v2 rejects rounded row mass when source is float32', s => { s.runtime.dtype = 'float32'; s.frames[0].layers[0].attention[0][1] = [0.50390625, 0.5]; });
bad('v2 rejects excessive BF16 row mass error', s => { s.runtime.dtype = 'bfloat16'; s.frames[0].layers[0].attention[0][1] = [0.51, 0.5]; });
