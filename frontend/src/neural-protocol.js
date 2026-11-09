/** Bounded, data-only transport helpers. No model execution or normalization. */
export const NEURAL_LIMITS = Object.freeze({ layers: 28, mlp: 6144, residual: 2048, activations: 229376, edges: 4000000, headerBytes: 65536 });

function invalid(message) { throw new Error(`Invalid neural data: ${message}`); }
function integer(value, minimum, maximum, name) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) invalid(name);
  return value;
}
export function greatestCommonDivisor(a, b) {
  while (b) { const remainder = a % b; a = b; b = remainder; }
  return a;
}
export function samplingStride(total, count) {
  integer(total, 1, NEURAL_LIMITS.mlp * NEURAL_LIMITS.residual, 'total connections');
  integer(count, 1, Math.min(total, NEURAL_LIMITS.edges), 'sample count');
  let stride = Math.ceil(total / count);
  while (greatestCommonDivisor(stride, total) !== 1) stride++;
  return stride;
}

/** Flat row-major matrix index, identical to the unsigned integer GPU formula. */
export function sampleFlatIndex(header, sample) {
  integer(sample, 0, header.count - 1, 'sample index');
  return (sample * header.stride) % header.total_connections;
}

/** Accept explicit model names and the SSE start architecture field names. */
export function normalizeArchitecture(value = {}) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid('architecture object');
  const architecture = {
    layers: value.layers ?? NEURAL_LIMITS.layers,
    d_model: value.d_model ?? value.residual_width ?? NEURAL_LIMITS.residual,
    intermediate_size: value.intermediate_size ?? value.mlp_width ?? NEURAL_LIMITS.mlp,
  };
  integer(architecture.layers, 1, NEURAL_LIMITS.layers, 'architecture.layers');
  integer(architecture.d_model, 1, NEURAL_LIMITS.residual, 'architecture.d_model');
  integer(architecture.intermediate_size, 1, NEURAL_LIMITS.mlp, 'architecture.intermediate_size');
  architecture.units_per_layer = architecture.d_model + architecture.intermediate_size;
  architecture.activation_count = architecture.layers * architecture.units_per_layer;
  if (value.activation_count !== undefined && value.activation_count !== architecture.activation_count) invalid('activation count does not match architecture');
  if (value.units_per_layer !== undefined && value.units_per_layer !== architecture.units_per_layer) invalid('units per layer do not match architecture');
  return architecture;
}

/** Parse CTXWGT01 + uint32LE header length + UTF-8 JSON + float32LE values. */
export function parseWeightEnvelope(input) {
  if (!(input instanceof ArrayBuffer) && !ArrayBuffer.isView(input)) invalid('binary envelope required');
  const bytes = input instanceof ArrayBuffer ? new Uint8Array(input) : new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  if (bytes.byteLength < 12 || bytes.byteLength > 12 + NEURAL_LIMITS.headerBytes + NEURAL_LIMITS.edges * 4) invalid('envelope size');
  const magic = String.fromCharCode(...bytes.subarray(0, 8));
  if (magic !== 'CTXWGT01') invalid('weight magic');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = view.getUint32(8, true);
  if (!headerLength || headerLength > NEURAL_LIMITS.headerBytes || 12 + headerLength > bytes.byteLength) invalid('header length');
  let header;
  try { header = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(12, 12 + headerLength))); }
  catch { invalid('header JSON/UTF-8'); }
  if (!header || typeof header !== 'object' || Array.isArray(header)) invalid('header object');
  if (header.version !== 1 || header.dtype !== 'float32' || header.sample_policy !== 'coprime-modular-v1') invalid('unsupported weight contract');
  integer(header.layer, 0, NEURAL_LIMITS.layers - 1, 'weight layer');
  integer(header.source_count, 1, NEURAL_LIMITS.mlp, 'source count');
  integer(header.target_count, 1, NEURAL_LIMITS.residual, 'target count');
  const total = header.source_count * header.target_count;
  if (header.total_connections !== total || !Array.isArray(header.shape) || header.shape.length !== 2 || header.shape[0] !== header.target_count || header.shape[1] !== header.source_count) invalid('weight matrix shape');
  integer(header.budget, 1, NEURAL_LIMITS.edges, 'weight budget');
  integer(header.count, 1, Math.min(total, header.budget), 'weight count');
  if (header.count !== Math.min(total, header.budget)) invalid('count/budget mismatch');
  const expectedStride = samplingStride(total, header.count);
  if (header.stride !== expectedStride) invalid('non-canonical coprime stride');
  if ((header.count - 1) * header.stride > 0xffffffff) invalid('unsigned GPU index overflow');
  if (typeof header.weight_abs_max !== 'number' || !Number.isFinite(header.weight_abs_max) || header.weight_abs_max < 0) invalid('weight_abs_max');
  const offset = 12 + headerLength;
  if (bytes.byteLength !== offset + header.count * 4) invalid('truncated or trailing weight payload');
  // Header length need not be aligned to four bytes. DataView also makes the
  // little-endian wire format independent of the host's byte order.
  const weights = new Float32Array(header.count);
  let maximum = 0;
  for (let i = 0; i < weights.length; i++) {
    const value = view.getFloat32(offset + i * 4, true);
    if (!Number.isFinite(value)) invalid('non-finite weight');
    weights[i] = value;
    maximum = Math.max(maximum, Math.abs(value));
  }
  if (Math.abs(maximum - header.weight_abs_max) > 1e-6 * Math.max(1e-12, maximum)) invalid('weight_abs_max differs from measured sample');
  return { header, weights };
}

/** Decode one last-token, layer-major MLP + residual activation capture. */
export function decodeActivationFrame(encoded, architecture) {
  const count = normalizeArchitecture(architecture).activation_count;
  const expectedBytes = count * 4;
  if (typeof encoded !== 'string' || encoded.length !== Math.ceil(expectedBytes / 3) * 4 || !/^[A-Za-z0-9+/]*={0,2}$/.test(encoded)) invalid('activation base64 size/characters');
  let binary;
  try { binary = atob(encoded); } catch { invalid('activation base64'); }
  if (binary.length !== expectedBytes) invalid('activation byte length');
  const bytes = new Uint8Array(expectedBytes);
  for (let i = 0; i < expectedBytes; i++) bytes[i] = binary.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const values = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const value = view.getFloat32(i * 4, true);
    if (!Number.isFinite(value)) invalid('non-finite activation');
    values[i] = value;
  }
  return values;
}

// Explicit aliases keep controller call sites concise without a second format.
export const parseWeights = parseWeightEnvelope;
export const decodeFrame = decodeActivationFrame;
