/** Validate portable recordings before they can alter scene state. */
export function validateSession(data) {
  const fail = (message) => { throw new Error(`Invalid session: ${message}`); };
  const object = (value, name) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) fail(name);
  };
  const integer = (value, min, max, name) => {
    if (!Number.isInteger(value) || value < min || value > max) fail(name);
  };
  const finite = (value, name, min = -Infinity, max = Infinity) => {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) fail(name);
  };
  const string = (value, name, max = 1000) => {
    if (typeof value !== 'string' || value.length > max) fail(name);
  };
  const array = (value, length, name) => {
    if (!Array.isArray(value) || value.length !== length) fail(`${name} shape`);
  };
  object(data, 'recording object required');
  const subset = data.schema_version === 'cortexlab.session/v2';
  if (!subset && data.schema_version !== 'cortexlab.session/v1') fail('unsupported schema_version');
  string(data.id, 'id', 200);
  string(data.created_at, 'created_at', 100);
  if (!Number.isFinite(Date.parse(data.created_at))) fail('created_at date');
  object(data.model, 'model');
  const m = data.model;
  integer(m.layers, 1, subset ? 128 : 8, 'model.layers');
  integer(m.heads, 1, subset ? 128 : 16, 'model.heads');
  integer(m.d_model, 1, subset ? 32768 : 256, 'model.d_model');
  integer(m.context_limit, 1, subset ? 64 : 48, 'model.context_limit');
  integer(m.vocab_size, 1, subset ? 1000000 : 100000, 'model.vocab_size');
  if (!subset && m.d_model % m.heads !== 0) fail('head dimension');
  if (typeof m.trained !== 'boolean') fail('model.trained');
  string(m.name, 'model.name', 300);
  object(m.tokenizer, 'model.tokenizer');
  const vocabulary = subset ? null : m.tokenizer.vocabulary;
  if (!subset) {
    array(vocabulary, m.vocab_size, 'vocabulary');
    vocabulary.forEach(token => string(token, 'vocabulary token', 400));
    if (new Set(vocabulary).size !== vocabulary.length) fail('duplicate vocabulary token');
  }
  object(data.capture, 'capture');
  const indices = (values, total, limit, name) => {
    if (!Array.isArray(values) || !values.length || values.length > limit) fail(name);
    values.forEach((value, i) => {
      integer(value, 0, total - 1, name);
      if (i && value <= values[i - 1]) fail(`${name} must be unique and increasing`);
    });
    return values;
  };
  const layers = subset ? indices(data.capture.layer_indices, m.layers, 3, 'capture.layer_indices') : Array.from({ length: m.layers }, (_, i) => i);
  const heads = subset ? indices(data.capture.head_indices, m.heads, 4, 'capture.head_indices') : Array.from({ length: m.heads }, (_, i) => i);
  const channels = subset ? indices(data.capture.channel_indices, m.d_model, 64, 'capture.channel_indices') : Array.from({ length: m.d_model }, (_, i) => i);
  // Serialization converts measured weights to float32 numbers, but cannot
  // undo source BF16 rounding. Keep the raw probabilities; do not renormalize.
  const sourceDtype = data.capture.source_dtype || data.runtime?.dtype || data.capture.dtype;
  const attentionMassTolerance = subset && sourceDtype === 'bfloat16' ? 0.006 : 0.00002;
  object(data.request, 'request');
  string(data.request.prompt, 'prompt', 400);
  if (!data.request.prompt.trim()) fail('empty prompt');
  integer(data.request.max_new_tokens, 1, subset ? 4 : 8, 'max_new_tokens');
  if (data.request.ablate != null) {
    object(data.request.ablate, 'ablate');
    integer(data.request.ablate.layer, 0, m.layers - 1, 'ablate.layer');
    integer(data.request.ablate.head, 0, m.heads - 1, 'ablate.head');
  }
  if (!Array.isArray(data.frames) || !data.frames.length || data.frames.length > data.request.max_new_tokens) fail('frame count');
  const labels = new Map();
  const observeLabel = (id, token) => {
    if (labels.has(id) && labels.get(id) !== token) fail('inconsistent decoded token label');
    labels.set(id, token);
  };
  let previous = null;
  for (let fi = 0; fi < data.frames.length; fi++) {
    const f = data.frames[fi];
    object(f, `frame ${fi}`);
    if (f.index !== fi) fail('frame ordering');
    finite(f.elapsed_ms, 'elapsed_ms', 0);
    finite(f.inference_ms, 'inference_ms', 0);
    if (previous && f.elapsed_ms < previous.elapsed_ms) fail('elapsed_ms ordering');
    if (!Array.isArray(f.input_ids) || !f.input_ids.length || f.input_ids.length > m.context_limit) fail('context bounds');
    const n = f.input_ids.length;
    array(f.tokens, n, 'tokens');
    f.input_ids.forEach(id => integer(id, 0, m.vocab_size - 1, 'token id'));
    f.tokens.forEach(t => string(t, 'token text', 400));
    integer(f.predicted_id, 0, m.vocab_size - 1, 'predicted_id');
    string(f.predicted_token, 'predicted_token', 400);
    f.input_ids.forEach((id, i) => {
      if (subset) observeLabel(id, f.tokens[i]);
      else if (f.tokens[i] !== vocabulary[id]) fail('token label/id mismatch');
    });
    if (subset) observeLabel(f.predicted_id, f.predicted_token);
    else if (f.predicted_token !== vocabulary[f.predicted_id]) fail('predicted label/id mismatch');
    if (fi === 0 && n + data.request.max_new_tokens > m.context_limit) fail('requested generation exceeds context');
    if (previous) {
      if (n !== previous.input_ids.length + 1) fail('context must grow by one token');
      previous.input_ids.forEach((id, i) => {
        if (f.input_ids[i] !== id || f.tokens[i] !== previous.tokens[i]) fail('context prefix changed');
      });
      if (f.input_ids[n - 1] !== previous.predicted_id || f.tokens[n - 1] !== previous.predicted_token) fail('prediction/context mismatch');
    }
    if (!Array.isArray(f.top_predictions) || !f.top_predictions.length || f.top_predictions.length > m.vocab_size) fail('top_predictions');
    let topSum = 0;
    f.top_predictions.forEach((p, pi) => {
      object(p, 'top prediction'); string(p.token, 'prediction token', 400);
      if (!subset && !vocabulary.includes(p.token)) fail('prediction token outside vocabulary');
      if (p.id !== undefined) {
        integer(p.id, 0, m.vocab_size - 1, 'top prediction id');
        if (subset) observeLabel(p.id, p.token);
        else if (p.token !== vocabulary[p.id]) fail('top prediction label/id mismatch');
      }
      finite(p.probability, 'prediction probability', 0, 1); topSum += p.probability;
      if (subset && pi && p.probability > f.top_predictions[pi - 1].probability + 1e-7) fail('top predictions not ordered');
    });
    if (subset && (f.top_predictions[0].token !== f.predicted_token || (f.top_predictions[0].id !== undefined && f.top_predictions[0].id !== f.predicted_id))) fail('greedy prediction mismatch');
    if (topSum > 1.002) fail('top prediction probability sum');
    array(f.layers, layers.length, 'captured layers');
    f.layers.forEach((layer, li) => {
      object(layer, 'layer');
      if (layer.index !== layers[li]) fail('captured layer ordering');
      array(layer.norms, n, 'norms');
      array(layer.residual, n, 'residual');
      layer.residual.forEach((sample, ti) => {
        array(sample, channels.length, 'residual channels');
        let squared = 0;
        sample.forEach(v => { finite(v, 'residual value', -1e6, 1e6); squared += v * v; });
        finite(layer.norms[ti], 'residual norm', 0, 1e9);
        // v2 norms are measured across the FULL model width; samples do not
        // determine that norm. Their own norm can only supply a lower bound.
        if (subset) {
          if (Math.sqrt(squared) > layer.norms[ti] + 0.0001 * Math.max(1, layer.norms[ti])) fail('sample norm exceeds full residual norm');
        } else if (Math.abs(Math.sqrt(squared) - layer.norms[ti]) > 0.0001 * Math.max(1, layer.norms[ti])) fail('norm does not match residual');
      });
      array(layer.attention, heads.length, 'captured attention heads');
      layer.attention.forEach(head => {
        array(head, n, 'attention queries');
        head.forEach((row, qi) => {
          array(row, n, 'attention keys');
          let sum = 0;
          row.forEach((weight, ki) => {
            finite(weight, 'attention weight', 0, 1);
            if (ki > qi && weight > 1e-6) fail('causal attention mask');
            sum += weight;
          });
          if (Math.abs(sum - 1) > attentionMassTolerance) fail('attention row sum');
        });
      });
    });
    previous = f;
  }
  object(data.metrics, 'metrics');
  ['total_ms', 'model_ms', 'capture_ms'].forEach(k => finite(data.metrics[k], `metrics.${k}`, 0));
  object(data.runtime, 'runtime');
  object(data.capture, 'capture');
  string(data.output, 'output', 4000);
  const checkFiniteTree = (value, depth = 0) => {
    if (depth > 20) fail('metadata nesting limit');
    if (typeof value === 'number' && !Number.isFinite(value)) fail('non-finite metadata');
    if (value && typeof value === 'object') Object.values(value).forEach(child => checkFiniteTree(child, depth + 1));
  };
  checkFiniteTree(data);
  return data;
}
