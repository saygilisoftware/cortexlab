import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { NEURAL_LIMITS, normalizeArchitecture, samplingStride } from './neural-protocol.js';

const QUALITY = Object.freeze({
  eco: { pointStride: 8, edgeLimit: 50000, hz: 15 },
  balanced: { pointStride: 4, edgeLimit: 250000, hz: 15 },
  full: { pointStride: 1, edgeLimit: 4000000, hz: 15 },
});

const LAYOUT = `
uniform int uLayers;
uniform int uMLP;
uniform int uResidual;
uniform int uUnits;
vec3 neuronPosition(int layer, int kind, int unit) {
  float angle = float(layer) / float(uLayers) * 6.28318530718 - 1.57079632679;
  vec3 radial = vec3(cos(angle), 0.0, sin(angle));
  vec3 tangent = vec3(-sin(angle), 0.0, cos(angle));
  float radius = uLayers > 1 ? max(12.0, float(uLayers) * 0.82) : 0.0;
  int count = kind == 0 ? uMLP : uResidual;
  int columns = min(count, kind == 0 ? 96 : 64);
  int rows = (count + columns - 1) / columns;
  float x = (float(unit % columns) + 0.5) / float(columns) - 0.5;
  float y = (float(unit / columns) + 0.5) / float(rows) - 0.5;
  return radial * (radius + (kind == 0 ? -2.6 : 2.6))
    + tangent * x * (kind == 0 ? 6.0 : 5.4)
    + vec3(0.0, y * (kind == 0 ? 4.8 : 4.2), 0.0);
}
uniform sampler2D uPrevious;
uniform sampler2D uCurrent;
uniform int uActivationWidth;
uniform float uBlend;
uniform bool uHasFrame;
float activation(int index) {
  ivec2 texel = ivec2(index % uActivationWidth, index / uActivationWidth);
  return mix(texelFetch(uPrevious, texel, 0).r, texelFetch(uCurrent, texel, 0).r, uBlend);
}
`;

const POINT_VERTEX = `${LAYOUT}
uniform int uPointStride;
uniform int uSelectedLayer;
uniform int uSelectedUnit;
uniform float uActivationMax;
uniform float uPointSize;
uniform float uFocus;
out vec3 vColor;
out float vAlpha;
void main() {
  int flatIndex = gl_InstanceID * uPointStride;
  int layer = flatIndex / uUnits;
  int local = flatIndex % uUnits;
  int kind = local < uMLP ? 0 : 1;
  int unit = kind == 0 ? local : local - uMLP;
  vec3 p = neuronPosition(layer, kind, unit);
  float value = uHasFrame ? activation(flatIndex) : 0.0;
  float strength = log(1.0 + abs(value)) / max(0.000001, log(1.0 + uActivationMax));
  vec3 signColor = value >= 0.0 ? vec3(0.14, 0.78, 0.98) : vec3(0.65, 0.28, 1.0);
  vColor = uHasFrame ? signColor * (0.18 + 0.82 * strength) : vec3(0.18, 0.29, 0.42);
  vAlpha = layer == uSelectedLayer ? 0.9 : mix(0.48, 0.16, uFocus);
  vec4 view = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * view;
  gl_PointSize = clamp(uPointSize * 35.0 / max(10.0, -view.z), 1.0, 4.0);
  if (flatIndex == uSelectedUnit) { vColor = vec3(1.0, 0.72, 0.36); vAlpha = 1.0; gl_PointSize = 8.0; }
}`;

const POINT_FRAGMENT = `precision highp float;
in vec3 vColor;
in float vAlpha;
out vec4 outColor;
void main() {
  float distance = length(gl_PointCoord - vec2(0.5));
  if (distance > 0.5) discard;
  outColor = vec4(vColor, vAlpha * (1.0 - smoothstep(0.12, 0.5, distance)));
}`;

const EDGE_VERTEX = `${LAYOUT}
uniform sampler2D uWeights;
uniform int uWeightWidth;
uniform uint uStride;
uniform uint uTotal;
uniform int uWeightLayer;
uniform float uWeightMax;
uniform float uThreshold;
uniform float uEdgeOpacity;
uniform float uActivationMax;
uniform bool uActivationEmphasis;
out vec3 vColor;
out float vAlpha;
void main() {
  // Integer multiplication is essential: 4M-edge products exceed the exact
  // integer range of float32. This matches the coprime-modular wire contract.
  uint edge = uint(gl_InstanceID);
  uint flatIndex = (edge * uStride) % uTotal;
  int source = int(flatIndex % uint(uMLP));
  int target = int(flatIndex / uint(uMLP));
  int edgeIndex = gl_InstanceID;
  float weight = texelFetch(uWeights, ivec2(edgeIndex % uWeightWidth, edgeIndex / uWeightWidth), 0).r;
  int endpoint = gl_VertexID % 2;
  vec3 p = neuronPosition(uWeightLayer, endpoint, endpoint == 0 ? source : target);
  float strength = sqrt(clamp(abs(weight) / max(0.00000001, uWeightMax), 0.0, 1.0));
  float emphasis = 1.0;
  if (uHasFrame && uActivationEmphasis) {
    float sourceValue = abs(activation(uWeightLayer * uUnits + source));
    // Actual weighted source contribution; its normalization is presentation.
    emphasis = 0.18 + 0.82 * clamp(log(1.0 + sourceValue * abs(weight) * 20.0)
      / max(0.000001, log(1.0 + uActivationMax * uWeightMax * 20.0)), 0.0, 1.0);
  }
  vColor = weight >= 0.0 ? vec3(0.08, 0.62, 0.96) : vec3(0.57, 0.20, 0.88);
  vAlpha = uEdgeOpacity * (0.0005 + 0.018 * strength) * emphasis;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  if (abs(weight) < uThreshold) { vAlpha = 0.0; gl_Position = vec4(2.0, 2.0, 2.0, 1.0); }
}`;

const EDGE_FRAGMENT = `precision highp float;
in vec3 vColor;
in float vAlpha;
out vec4 outColor;
void main() { if (vAlpha <= 0.0) discard; outColor = vec4(vColor, vAlpha); }`;

/**
 * GPU-backed, data-only neural instrument. Eco defaults are intentionally
 * conservative. Larger modes are capacities, not verified device benchmarks.
 * Creating this class creates a WebGL context; inference is never started here.
 */
export class NeuralScene {
  constructor(container, { onSelect = () => {}, onStats = () => {} } = {}) {
    this.container = container; this.onSelect = onSelect; this.onStats = onStats;
    this.quality = 'eco'; this.selectedLayer = 13; this.selectedUnit = -1;
    this.showEdges = true; this.threshold = 0; this.hasFrame = false;
    this.disposed = false; this.contextLost = false; this.dirty = false;
    this.intervals = []; this.cpuTimes = []; this.frameCount = 0;
    this.lastRender = 0; this.lastStats = 0; this.previousContinuing = false;
    this.timer = null; this.raf = null; this.cameraMode = 'overview';
    this.motion = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color('#050914');
    this.camera = new THREE.PerspectiveCamera(44, 1, 0.1, 250);
    this.camera.position.set(38, 29, 48);
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'low-power', alpha: false });
    try {
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.append(this.renderer.domElement);
    this.renderer.domElement.setAttribute('aria-label', 'Neural architecture. Drag to orbit; click a computational unit to inspect its actual captured value when available.');
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.16;
    this.controls.minDistance = 5; this.controls.maxDistance = 130;
    this.controls.maxPolarAngle = Math.PI * 0.94;
    this._change = () => this.invalidate();
    this._start = () => { this.cameraTween = null; this.invalidate(); };
    this.controls.addEventListener('change', this._change);
    this.controls.addEventListener('start', this._start);
    this.grid = new THREE.GridHelper(80, 40, '#20344a', '#101c2c');
    this.grid.position.y = -5.5; this.grid.material.transparent = true;
    this.grid.material.opacity = 0.25; this.scene.add(this.grid);
    this.labels = new THREE.Group(); this.scene.add(this.labels);
    this.uniforms = {
      uLayers: { value: 28 }, uMLP: { value: 6144 }, uResidual: { value: 2048 }, uUnits: { value: 8192 },
      uPrevious: { value: null }, uCurrent: { value: null }, uActivationWidth: { value: 1024 },
      uBlend: { value: 1 }, uHasFrame: { value: false }, uActivationMax: { value: 1 },
      uPointStride: { value: 8 }, uSelectedLayer: { value: 13 }, uSelectedUnit: { value: -1 },
      uPointSize: { value: 2.2 }, uFocus: { value: 0 }, uWeights: { value: null },
      uWeightWidth: { value: 1 }, uStride: { value: 1 }, uTotal: { value: 1 },
      uWeightLayer: { value: 13 }, uWeightMax: { value: 1 }, uThreshold: { value: 0 },
      uEdgeOpacity: { value: 0.7 }, uActivationEmphasis: { value: true },
    };
    const materialOptions = { uniforms: this.uniforms, glslVersion: THREE.GLSL3, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending };
    this.pointMaterial = new THREE.ShaderMaterial({ ...materialOptions, vertexShader: POINT_VERTEX, fragmentShader: POINT_FRAGMENT });
    this.edgeMaterial = new THREE.ShaderMaterial({ ...materialOptions, vertexShader: EDGE_VERTEX, fragmentShader: EDGE_FRAGMENT });
    // A single vertex/two endpoints are instanced. There is no CPU array of
    // millions of edge objects or explicit endpoint positions.
    this.pointGeometry = new THREE.InstancedBufferGeometry();
    this.pointGeometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    this.pointGeometry.setDrawRange(0, 1);
    this.points = new THREE.Points(this.pointGeometry, this.pointMaterial);
    this.points.frustumCulled = false; this.points.renderOrder = 2; this.scene.add(this.points);
    this.edgeGeometry = new THREE.InstancedBufferGeometry();
    this.edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 0], 3));
    this.edgeGeometry.setDrawRange(0, 2); this.edgeGeometry.instanceCount = 0;
    this.edges = new THREE.LineSegments(this.edgeGeometry, this.edgeMaterial);
    this.edges.frustumCulled = false; this.edges.visible = false; this.scene.add(this.edges);
    this.scratch = new THREE.Vector3(); this.projected = new THREE.Vector3();
    this._down = e => { this.pointerDown = { x: e.clientX, y: e.clientY }; };
    this._click = e => {
      if (this.pointerDown && Math.hypot(e.clientX - this.pointerDown.x, e.clientY - this.pointerDown.y) <= 5) this.pick(e);
    };
    this._lost = e => { e.preventDefault(); this.contextLost = true; this.cancelScheduled(); this.emitStats(true); };
    this._restored = () => { this.contextLost = false; this.previousTexture.needsUpdate = true; this.currentTexture.needsUpdate = true; if (this.weightTexture) this.weightTexture.needsUpdate = true; this.invalidate(); };
    this._visibility = () => { this.previousContinuing = false; if (document.visibilityState === 'hidden') this.cancelScheduled(); else this.invalidate(); };
    this.renderer.domElement.addEventListener('pointerdown', this._down);
    this.renderer.domElement.addEventListener('click', this._click);
    this.renderer.domElement.addEventListener('webglcontextlost', this._lost);
    this.renderer.domElement.addEventListener('webglcontextrestored', this._restored);
    document.addEventListener('visibilitychange', this._visibility);
    this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(container);
    this.setArchitecture({ layers: 28, d_model: 2048, intermediate_size: 6144 });
    this.resize();
    } catch (error) { this.dispose(); throw error; }
  }

  texture(data, width, height) {
    const texture = new THREE.DataTexture(data, width, height, THREE.RedFormat, THREE.FloatType);
    texture.internalFormat = 'R32F'; texture.minFilter = THREE.NearestFilter; texture.magFilter = THREE.NearestFilter;
    texture.generateMipmaps = false; texture.unpackAlignment = 1; texture.needsUpdate = true;
    return texture;
  }

  setArchitecture(value) {
    const next = normalizeArchitecture(value);
    if (this.architecture && ['layers', 'd_model', 'intermediate_size'].every(k => this.architecture[k] === next[k])) return;
    this.architecture = next; this.hasFrame = false; this.rawFrame = null; this.meta = null;
    this.selectedLayer = Math.min(this.selectedLayer, next.layers - 1); this.selectedUnit = -1;
    this.previousTexture?.dispose(); this.currentTexture?.dispose(); this.weightTexture?.dispose();
    this.weightTexture = null; this.weightData = null; this.weightHeader = null; this.edges.visible = false;
    const width = Math.min(1024, this.renderer.capabilities.maxTextureSize, next.activation_count);
    const height = Math.ceil(next.activation_count / width);
    this.previousTexture = this.texture(new Float32Array(width * height), width, height);
    this.currentTexture = this.texture(new Float32Array(width * height), width, height);
    Object.assign(this.uniforms.uLayers, { value: next.layers });
    this.uniforms.uMLP.value = next.intermediate_size; this.uniforms.uResidual.value = next.d_model;
    this.uniforms.uUnits.value = next.units_per_layer; this.uniforms.uActivationWidth.value = width;
    this.uniforms.uPrevious.value = this.previousTexture; this.uniforms.uCurrent.value = this.currentTexture;
    this.uniforms.uHasFrame.value = false; this.uniforms.uSelectedLayer.value = this.selectedLayer;
    this.uniforms.uSelectedUnit.value = -1; this.uniforms.uBlend.value = 1;
    this.clearLabels();
    for (let layer = 0; layer < next.layers; layer++) {
      const canvas = document.createElement('canvas'); canvas.width = 192; canvas.height = 64;
      const ctx = canvas.getContext('2d'); ctx.font = '500 24px ui-monospace, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#a8c8dc';
      ctx.fillText(`BLOCK ${String(layer + 1).padStart(2, '0')}`, 96, 32);
      const map = new THREE.CanvasTexture(canvas); map.colorSpace = THREE.SRGBColorSpace;
      const label = new THREE.Sprite(new THREE.SpriteMaterial({ map, transparent: true, depthWrite: false, depthTest: false }));
      const center = this.layerCenter(layer); label.position.copy(center); label.position.y = 4.1;
      label.scale.set(2.8, 0.93, 1); label.userData.layer = layer; this.labels.add(label);
    }
    this.applyQuality(); this.setLayer(this.selectedLayer); this.invalidate();
  }

  setFrame(values, meta = {}) {
    if (!(values instanceof Float32Array) || values.length !== this.architecture.activation_count) throw new Error('Neural frame must contain the full layer-major MLP + residual capture');
    let maximum = 0;
    for (const value of values) { if (!Number.isFinite(value)) throw new Error('Non-finite neural activation'); maximum = Math.max(maximum, Math.abs(value)); }
    const now = performance.now();
    const oldBlend = this.transitionStart === undefined ? 1 : this.blendAt(now);
    const previous = this.previousTexture.image.data, current = this.currentTexture.image.data;
    if (this.hasFrame) {
      for (let i = 0; i < values.length; i++) previous[i] += (current[i] - previous[i]) * oldBlend;
    } else previous.set(values); // First capture is shown directly, not a fabricated zero measurement.
    current.set(values); this.rawFrame = values.slice(); this.meta = { ...meta };
    this.hasFrame = true; this.uniforms.uHasFrame.value = true;
    this.uniforms.uActivationMax.value = Math.max(maximum, 1e-12);
    this.previousTexture.needsUpdate = true; this.currentTexture.needsUpdate = true;
    this.transitionStart = now; this.uniforms.uBlend.value = this.motion.matches ? 1 : 0;
    this.invalidate();
  }

  setWeights({ header, weights }) {
    const a = this.architecture;
    if (!header || !(weights instanceof Float32Array) || weights.length !== header.count || header.count > NEURAL_LIMITS.edges || !header.count || header.source_count !== a.intermediate_size || header.target_count !== a.d_model || header.total_connections !== a.intermediate_size * a.d_model || !Number.isInteger(header.layer) || header.layer < 0 || header.layer >= a.layers || header.sample_policy !== 'coprime-modular-v1' || header.stride !== samplingStride(header.total_connections, header.count)) throw new Error('Weights do not match the neural architecture');
    let maximum = 0;
    for (const weight of weights) { if (!Number.isFinite(weight)) throw new Error('Non-finite neural weight'); maximum = Math.max(maximum, Math.abs(weight)); }
    const width = Math.min(2 ** Math.ceil(Math.log2(Math.max(1, Math.sqrt(weights.length)))), this.renderer.capabilities.maxTextureSize);
    const height = Math.ceil(weights.length / width);
    if (height > this.renderer.capabilities.maxTextureSize) throw new Error('Weight texture exceeds this WebGL device limit; choose Eco');
    const padded = new Float32Array(width * height); padded.set(weights);
    const texture = this.texture(padded, width, height);
    this.weightTexture?.dispose(); this.weightTexture = texture;
    this.weightData = weights; this.weightHeader = { ...header };
    this.uniforms.uWeights.value = texture; this.uniforms.uWeightWidth.value = width;
    this.uniforms.uStride.value = header.stride; this.uniforms.uTotal.value = header.total_connections;
    this.uniforms.uWeightLayer.value = header.layer; this.uniforms.uWeightMax.value = Math.max(maximum, 1e-12);
    this.applyQuality(); this.invalidate();
  }

  setLayer(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.architecture.layers) throw new Error('Neural layer outside architecture');
    this.selectedLayer = index; this.uniforms.uSelectedLayer.value = index;
    this.selectedUnit = -1; this.uniforms.uSelectedUnit.value = -1;
    this.labels.children.forEach(label => { label.material.color.set(label.userData.layer === index ? '#ffe0ac' : '#6d91ac'); });
    this.updateEdges();
    if (this.cameraMode === 'focus') this.setCamera('focus');
    this.invalidate();
  }

  setQuality(quality) {
    if (!QUALITY[quality]) throw new Error('Unknown neural quality');
    this.quality = quality; this.renderError = null; this.applyQuality(); this.previousContinuing = false; this.invalidate();
  }
  applyQuality() {
    const quality = QUALITY[this.quality];
    this.uniforms.uPointStride.value = quality.pointStride;
    this.pointGeometry.instanceCount = Math.ceil(this.architecture.activation_count / quality.pointStride);
    this.edgeGeometry.instanceCount = Math.min(this.weightHeader?.count || 0, quality.edgeLimit);
    this.updateEdges();
  }
  updateEdges() {
    this.edges.visible = !!(this.showEdges && this.weightHeader && this.weightHeader.layer === this.selectedLayer);
    this.passingEdges = 0;
    if (this.edges.visible) for (let i = 0; i < this.edgeGeometry.instanceCount; i++) if (Math.abs(this.weightData[i]) >= this.threshold) this.passingEdges++;
  }
  setThreshold(value) {
    if (!Number.isFinite(value) || value < 0) throw new Error('Weight threshold must be a nonnegative raw magnitude');
    this.threshold = value; this.uniforms.uThreshold.value = value; this.updateEdges(); this.invalidate();
  }
  setEdgeVisibility(visible) { this.showEdges = !!visible; this.updateEdges(); this.invalidate(); }
  setActivationEmphasis(enabled) { this.uniforms.uActivationEmphasis.value = !!enabled; this.invalidate(); }
  setEdgeOpacity(value) {
    if (!Number.isFinite(value) || value < 0 || value > 2) throw new Error('Invalid edge opacity');
    this.uniforms.uEdgeOpacity.value = value; this.invalidate();
  }

  layerCenter(layer, target = new THREE.Vector3()) {
    const angle = layer / this.architecture.layers * Math.PI * 2 - Math.PI / 2;
    const radius = this.architecture.layers > 1 ? Math.max(12, this.architecture.layers * 0.82) : 0;
    return target.set(Math.cos(angle) * radius, 0, Math.sin(angle) * radius);
  }
  unitPosition(layer, kind, unit, target) {
    const angle = layer / this.architecture.layers * Math.PI * 2 - Math.PI / 2;
    const count = kind === 'mlp' ? this.architecture.intermediate_size : this.architecture.d_model;
    const columns = Math.min(count, kind === 'mlp' ? 96 : 64), rows = Math.ceil(count / columns);
    const x = ((unit % columns + 0.5) / columns - 0.5) * (kind === 'mlp' ? 6 : 5.4);
    const y = ((Math.floor(unit / columns) + 0.5) / rows - 0.5) * (kind === 'mlp' ? 4.8 : 4.2);
    const radius = (this.architecture.layers > 1 ? Math.max(12, this.architecture.layers * 0.82) : 0) + (kind === 'mlp' ? -2.6 : 2.6);
    return target.set(Math.cos(angle) * radius - Math.sin(angle) * x, y, Math.sin(angle) * radius + Math.cos(angle) * x);
  }

  pick(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    this.camera.updateMatrixWorld();
    let nearestLayer = this.selectedLayer, centerDistance = Infinity;
    for (let layer = 0; layer < this.architecture.layers; layer++) {
      this.layerCenter(layer, this.projected).project(this.camera);
      if (this.projected.z < -1 || this.projected.z > 1) continue;
      const distance = Math.hypot((this.projected.x + 1) * rect.width / 2 - x, (1 - this.projected.y) * rect.height / 2 - y);
      if (distance < centerDistance) { centerDistance = distance; nearestLayer = layer; }
    }
    if (centerDistance > 180) return;
    let found = null, closest = 100;
    const a = this.architecture, stride = QUALITY[this.quality].pointStride;
    // Pick only the actual units submitted in the current point subsample.
    for (let local = 0; local < a.units_per_layer; local++) {
      const flat = nearestLayer * a.units_per_layer + local;
      if (flat % stride) continue;
      const kind = local < a.intermediate_size ? 'mlp' : 'residual';
      const unit = kind === 'mlp' ? local : local - a.intermediate_size;
      this.unitPosition(nearestLayer, kind, unit, this.projected).project(this.camera);
      if (this.projected.z < -1 || this.projected.z > 1) continue;
      const dx = (this.projected.x + 1) * rect.width / 2 - x, dy = (1 - this.projected.y) * rect.height / 2 - y;
      const distance = dx * dx + dy * dy;
      if (distance < closest) { closest = distance; found = { layer: nearestLayer, kind, unit, flat }; }
    }
    if (!found) return;
    if (this.selectedLayer !== found.layer) this.setLayer(found.layer);
    this.selectedUnit = found.flat; this.uniforms.uSelectedUnit.value = found.flat;
    this.onSelect({ layer: found.layer, kind: found.kind, unit: found.unit, value: this.hasFrame ? this.rawFrame[found.flat] : null, measured: this.hasFrame });
    this.invalidate();
  }

  setCamera(preset = 'overview') {
    this.cameraMode = preset; let target = new THREE.Vector3(), position;
    if (preset === 'focus') {
      target = this.layerCenter(this.selectedLayer);
      const angle = this.selectedLayer / this.architecture.layers * Math.PI * 2 - Math.PI / 2;
      position = target.clone().add(new THREE.Vector3(Math.cos(angle) * 15 - Math.sin(angle) * 8, 6, Math.sin(angle) * 15 + Math.cos(angle) * 8));
    } else if (preset === 'top') position = new THREE.Vector3(0, 66, 0.1);
    else if (preset === 'front') position = new THREE.Vector3(0, 9, 66);
    else { this.cameraMode = 'overview'; position = new THREE.Vector3(38, 29, 48); }
    this.uniforms.uFocus.value = preset === 'focus' ? 1 : 0;
    this.cameraTween = { from: this.camera.position.clone(), targetFrom: this.controls.target.clone(), position, target, start: performance.now() };
    this.invalidate();
  }
  focusSelection() { this.setCamera('focus'); }
  blendAt(time) { const t = Math.min(1, Math.max(0, (time - this.transitionStart) / 220)); return this.motion.matches ? 1 : t * t * (3 - 2 * t); }

  invalidate() {
    this.dirty = true;
    if (this.disposed || this.contextLost || this.renderError || this.rendering || document.visibilityState === 'hidden' || this.timer !== null || this.raf !== null) return;
    const delay = Math.max(0, 1000 / QUALITY[this.quality].hz - (performance.now() - this.lastRender));
    this.timer = setTimeout(() => {
      this.timer = null;
      if (!this.disposed && !this.contextLost) this.raf = requestAnimationFrame(time => { this.raf = null; this.render(time); });
    }, delay);
  }
  render(time) {
    if (this.disposed || this.contextLost || this.renderError || document.visibilityState === 'hidden') return;
    const started = performance.now(); this.dirty = false; this.rendering = true;
    let continuing = false, controlsChanged = false;
    try {
    if (this.previousContinuing && this.lastRender) { this.intervals.push(time - this.lastRender); if (this.intervals.length > 240) this.intervals.shift(); }
    if (this.hasFrame && this.transitionStart !== undefined) {
      const blend = this.blendAt(time); this.uniforms.uBlend.value = blend;
      if (blend < 1) continuing = true;
    }
    if (this.cameraTween) {
      const tween = this.cameraTween;
      const t = this.motion.matches ? 1 : Math.min(1, Math.max(0, (time - tween.start) / 300));
      const blend = t * t * (3 - 2 * t);
      this.camera.position.lerpVectors(tween.from, tween.position, blend);
      this.controls.target.lerpVectors(tween.targetFrom, tween.target, blend);
      if (t === 1) this.cameraTween = null; else continuing = true;
    }
    controlsChanged = this.controls.update();
    this.renderer.render(this.scene, this.camera); this.frameCount++;
    this.lastSubmission = { units: this.pointGeometry.instanceCount, edges: this.edges.visible ? this.edgeGeometry.instanceCount : 0, passingEdges: this.passingEdges };
    this.cpuTimes.push(performance.now() - started); if (this.cpuTimes.length > 240) this.cpuTimes.shift();
    this.lastRender = time; this.previousContinuing = continuing || controlsChanged;
    this.emitStats();
    } catch (error) {
      this.renderError = error?.message || String(error);
      this.previousContinuing = false; this.dirty = false; this.cancelScheduled();
      try { this.onStats(this.getStats()); } catch { /* A consumer callback cannot prevent resource cleanup. */ }
    } finally { this.rendering = false; }
    if (!this.renderError && (continuing || controlsChanged || this.dirty)) this.invalidate();
  }
  emitStats(force = false) {
    if (force || !this.lastStats || performance.now() - this.lastStats > 500) { this.lastStats = performance.now(); this.onStats(this.getStats()); }
  }
  resize() {
    if (this.disposed) return;
    const width = Math.max(1, this.container.clientWidth), height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false); this.camera.aspect = width / height; this.camera.updateProjectionMatrix(); this.invalidate();
  }
  getStats() {
    if (this.disposed) return { disposed: true, hasFrame: false, drawnUnits: 0, drawnEdges: 0, drawCalls: 0, textures: 0, geometries: 0, uploadedTextureBytes: 0, gpuCapacityVerified: false };
    const percentile = values => { const sorted = [...values].sort((a, b) => a - b); return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] : null; };
    const activeMean = this.intervals.length ? this.intervals.reduce((a, b) => a + b, 0) / this.intervals.length : null;
    return {
      quality: this.quality, frameCap: QUALITY[this.quality].hz, renderOnDemand: true,
      hasFrame: this.hasFrame, measuredUnits: this.hasFrame ? this.architecture.activation_count : 0,
      totalUnits: this.architecture.activation_count, configuredUnits: this.pointGeometry.instanceCount,
      drawnUnits: this.contextLost ? 0 : this.lastSubmission?.units || 0,
      pointStride: QUALITY[this.quality].pointStride, selectedLayer: this.selectedLayer,
      sampledEdges: this.weightHeader?.count || 0, configuredEdges: this.edges.visible ? this.edgeGeometry.instanceCount : 0,
      drawnEdges: this.contextLost ? 0 : this.lastSubmission?.edges || 0,
      thresholdPassingEdges: this.contextLost ? 0 : this.lastSubmission?.passingEdges || 0,
      fullMatrixConnections: this.weightHeader?.total_connections || 0,
      weightsLayer: this.weightHeader?.layer ?? null, frameIndex: this.meta?.index ?? null,
      frames: this.frameCount, activeIntervalSamples: this.intervals.length,
      meanActiveFrameMs: activeMean, p95ActiveFrameMs: percentile(this.intervals), p95CpuSubmitMs: percentile(this.cpuTimes),
      drawCalls: this.renderer.info.render.calls, geometries: this.renderer.info.memory.geometries, textures: this.renderer.info.memory.textures,
      uploadedTextureBytes: this.previousTexture.image.data.byteLength + this.currentTexture.image.data.byteLength + (this.weightTexture?.image.data.byteLength || 0),
      contextLost: this.contextLost, renderError: this.renderError || null, gpuCapacityVerified: false,
    };
  }
  clearLabels() { this.labels?.children.forEach(label => { label.material.map.dispose(); label.material.dispose(); }); this.labels?.clear(); }
  cancelScheduled() { if (this.timer !== null) clearTimeout(this.timer); if (this.raf !== null) cancelAnimationFrame(this.raf); this.timer = null; this.raf = null; }
  dispose() {
    if (this.disposed) return;
    this.disposed = true; this.cancelScheduled(); this.observer?.disconnect();
    this.controls?.removeEventListener('change', this._change); this.controls?.removeEventListener('start', this._start); this.controls?.dispose();
    document.removeEventListener('visibilitychange', this._visibility);
    this.renderer.domElement.removeEventListener('pointerdown', this._down);
    this.renderer.domElement.removeEventListener('click', this._click);
    this.renderer.domElement.removeEventListener('webglcontextlost', this._lost);
    this.renderer.domElement.removeEventListener('webglcontextrestored', this._restored);
    this.clearLabels(); this.pointGeometry?.dispose(); this.edgeGeometry?.dispose(); this.pointMaterial?.dispose(); this.edgeMaterial?.dispose();
    for (const texture of [this.previousTexture, this.currentTexture, this.weightTexture]) if (texture) { texture.dispose(); texture.image.data = null; }
    if (this.uniforms) for (const name of ['uPrevious', 'uCurrent', 'uWeights']) this.uniforms[name].value = null;
    this.previousTexture = null; this.currentTexture = null; this.weightTexture = null;
    this.grid?.geometry.dispose(); this.grid?.material.dispose();
    this.rawFrame = null; this.weightData = null; this.weightHeader = null;
    this.scene?.clear(); this.renderer.dispose(); this.renderer.forceContextLoss(); this.renderer.domElement.remove();
  }
}
