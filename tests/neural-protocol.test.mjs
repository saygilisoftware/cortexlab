import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseWeightEnvelope, decodeActivationFrame, normalizeArchitecture, samplingStride, sampleFlatIndex, NEURAL_LIMITS } from '../frontend/src/neural-protocol.js';

// Explicitly synthetic numerical protocol fixtures. Nothing here is an
// inference recording, pretrained parameter sample, or GPU benchmark.
function envelope(change = {}, values) {
  const total = 35, count = 7, stride = samplingStride(total, count);
  const sample = values || Float32Array.from({ length: count }, (_, i) => (((i * stride) % total) - 17) / 32);
  const header = { version: 1, layer: 13, source_count: 7, target_count: 5, total_connections: total, count, budget: count, stride, dtype: 'float32', sample_policy: 'coprime-modular-v1', shape: [5, 7], weight_abs_max: Math.max(...sample.map(Math.abs)), ...change };
  const json = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(12 + json.length + sample.length * 4);
  bytes.set(new TextEncoder().encode('CTXWGT01'));
  const view = new DataView(bytes.buffer); view.setUint32(8, json.length, true);
  bytes.set(json, 12);
  sample.forEach((value, i) => view.setFloat32(12 + json.length + i * 4, value, true));
  return { bytes, header, sample };
}

test('binary sample preserves signed float32 and exact coprime matrix indices', () => {
  const { bytes, sample } = envelope();
  const result = parseWeightEnvelope(bytes);
  assert.deepEqual(result.weights, sample);
  assert.deepEqual(Array.from({ length: 7 }, (_, i) => sampleFlatIndex(result.header, i)), [0, 6, 12, 18, 24, 30, 1]);
});
test('parser accepts an offset byte view without reading surrounding bytes', () => {
  const { bytes } = envelope(); const wrapped = new Uint8Array(bytes.length + 13);
  wrapped.set(bytes, 5); assert.equal(parseWeightEnvelope(wrapped.subarray(5, 5 + bytes.length)).weights.length, 7);
});
test('unaligned UTF-8 header still decodes little-endian float32 correctly', () => {
  let value = envelope({ note: 'λ' });
  while ((12 + new TextEncoder().encode(JSON.stringify(value.header)).length) % 4 === 0) value = envelope({ note: value.header.note + 'x' });
  assert.deepEqual(parseWeightEnvelope(value.bytes).weights, value.sample);
});
test('default 50K and opt-in 2M/4M mappings use exact uint-compatible arithmetic', () => {
  const total = 6144 * 2048;
  assert.equal(samplingStride(total, 50000), 253);
  assert.equal(samplingStride(total, 2000000), 7);
  assert.equal(samplingStride(total, 4000000), 5);
  const header = { total_connections: total, count: 4000000, stride: 5 };
  assert.equal(sampleFlatIndex(header, 3999999), 7417083);
  assert.notEqual(sampleFlatIndex(header, 3999999), Math.fround(Math.fround(3999999) * 5) % total);
});
test('small coprime sample has no repeated matrix indices', () => {
  const { header } = envelope(); const seen = new Set();
  for (let i = 0; i < header.count; i++) seen.add(sampleFlatIndex(header, i));
  assert.equal(seen.size, header.count);
});
test('single-edge budget keeps canonical stride and samples actual flat index zero', () => {
  const total = 6144 * 2048;
  const header = { version: 1, layer: 0, source_count: 6144, target_count: 2048, total_connections: total, shape: [2048, 6144], count: 1, budget: 1, stride: samplingStride(total, 1), dtype: 'float32', sample_policy: 'coprime-modular-v1', weight_abs_max: 0.25 };
  assert.equal(header.stride, total + 1);
  const value = envelope(header, Float32Array.of(-0.25));
  const parsed = parseWeightEnvelope(value.bytes);
  assert.equal(sampleFlatIndex(parsed.header, 0), 0);
  assert.equal(parsed.weights[0], -0.25);
});

const badHeader = (name, change, values) => test(name, () => assert.throws(() => parseWeightEnvelope(envelope(change, values).bytes), /Invalid neural data/));
badHeader('reject unsupported version', { version: 2 });
badHeader('reject float64 claim', { dtype: 'float64' });
badHeader('reject obsolete aliased striding policy', { sample_policy: 'index=i*stride' });
badHeader('reject noncanonical modular stride', { stride: 5 });
badHeader('reject inconsistent matrix shape', { shape: [7, 5] });
badHeader('reject inconsistent matrix element count', { total_connections: 36 });
badHeader('reject count/budget mismatch', { budget: 8 });
badHeader('reject invalid layer', { layer: 28 });
badHeader('reject oversized connection allocation', { budget: 4000001 });
badHeader('reject zero sample count', { count: 0 });
badHeader('reject false measured maximum', { weight_abs_max: 99 });
badHeader('reject non-finite raw weights', { weight_abs_max: 1 }, Float32Array.of(0, 0, NaN, 0, 0, 0, 0));
test('reject wrong magic', () => { const { bytes } = envelope(); bytes[0] = 0; assert.throws(() => parseWeightEnvelope(bytes), /weight magic/); });
test('reject truncated and trailing payloads', () => {
  const { bytes } = envelope(); assert.throws(() => parseWeightEnvelope(bytes.slice(0, -1)), /payload/);
  const extended = new Uint8Array(bytes.length + 1); extended.set(bytes); assert.throws(() => parseWeightEnvelope(extended), /payload/);
});
test('reject invalid or excessively long header length before decoding', () => {
  const { bytes } = envelope(); new DataView(bytes.buffer).setUint32(8, NEURAL_LIMITS.headerBytes + 1, true);
  assert.throws(() => parseWeightEnvelope(bytes), /header length/);
});
test('reject invalid JSON header', () => {
  const { bytes } = envelope(); bytes[12] = '!'.charCodeAt(0); assert.throws(() => parseWeightEnvelope(bytes), /JSON/);
});
test('sample lookup rejects out-of-range indices', () => {
  const { header } = envelope(); assert.throws(() => sampleFlatIndex(header, -1)); assert.throws(() => sampleFlatIndex(header, header.count));
});

const smallArchitecture = { layers: 2, mlp_width: 3, residual_width: 2, units_per_layer: 5, activation_count: 10 };
function encodedFrame(values) {
  const bytes = new Uint8Array(values.length * 4), view = new DataView(bytes.buffer);
  values.forEach((value, i) => view.setFloat32(i * 4, value, true));
  return Buffer.from(bytes).toString('base64');
}
test('frame decoding preserves exact layer-major values, signs, and negative zero', () => {
  const values = Float32Array.of(-3, 1.25, -0, 9, 0, 2, 4, 5, -6, 7);
  assert.deepEqual(decodeActivationFrame(encodedFrame(values), smallArchitecture), values);
});
test('architecture aliases resolve to the same bounded full capture shape', () => {
  assert.deepEqual(normalizeArchitecture(smallArchitecture), normalizeArchitecture({ layers: 2, intermediate_size: 3, d_model: 2 }));
  assert.equal(normalizeArchitecture().activation_count, 229376);
  assert.equal(Math.ceil(normalizeArchitecture().activation_count / 8), 28672);
});
test('reject sparse legacy channels presented as a complete dense frame', () => assert.throws(() => decodeActivationFrame(encodedFrame(new Float32Array(64))), /base64 size/));
test('reject non-finite activation', () => assert.throws(() => decodeActivationFrame(encodedFrame([0, 0, 0, 0, 0, Infinity, 0, 0, 0, 0]), smallArchitecture), /non-finite activation/));
test('reject malformed frame characters and byte length', () => {
  const encoded = encodedFrame(new Float32Array(10));
  assert.throws(() => decodeActivationFrame(`!${encoded.slice(1)}`, smallArchitecture), /characters/);
  assert.throws(() => decodeActivationFrame(encoded.slice(4), smallArchitecture), /size/);
});
test('reject contradictory or unsupported architecture', () => {
  assert.throws(() => normalizeArchitecture({ ...smallArchitecture, activation_count: 11 }), /activation count/);
  assert.throws(() => normalizeArchitecture({ layers: 29 }), /layers/);
  assert.throws(() => normalizeArchitecture({ d_model: 2049 }), /d_model/);
  assert.throws(() => normalizeArchitecture(null), /architecture object/);
});
