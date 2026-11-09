import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { validateSession } from '../frontend/src/contracts.js';

const fixture = JSON.parse(readFileSync(new URL('../examples/demo-session.json', import.meta.url), 'utf8'));
test('real PyTorch fixture validates with exact values after JSON round trip', () => {
  assert.equal(validateSession(fixture), fixture);
  assert.deepEqual(validateSession(JSON.parse(JSON.stringify(fixture))), fixture);
});
const bad = (name, mutate) => test(name, () => {
  const value = structuredClone(fixture); mutate(value); assert.throws(() => validateSession(value), /Invalid session/);
});
bad('reject incompatible schema', s => s.schema_version = 'cortexlab.session/v2');
bad('reject causal-mask violation', s => s.frames[0].layers[0].attention[0][0][1] = .2);
bad('reject non-normalized attention', s => s.frames[0].layers[0].attention[0][0][0] = .8);
bad('reject wrong tensor shape', s => s.frames[0].layers[0].residual[0].pop());
bad('reject non-finite residual', s => s.frames[0].layers[0].residual[0][0] = NaN);
bad('reject false activation norm', s => s.frames[0].layers[0].norms[0] += 1);
bad('reject incorrect token label', s => s.frames[0].tokens[0] = 'not the actual token');
bad('reject reordered generation context', s => s.frames[1].input_ids[0] = s.frames[0].predicted_id);
bad('reject non-finite metadata', s => s.metrics.total_ms = Infinity);
bad('reject non-integer head ablation', s => s.request.ablate = {layer: 0, head: true});
