import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const CYAN = new THREE.Color('#54dce8');
const VIOLET = new THREE.Color('#a084f5');
const GOLD = new THREE.Color('#ffd29a');

/** Schematic scene: each node represents one post-block residual vector. */
export class ObservatoryScene {
  constructor(container, { onSelect = () => {}, onHover = () => {} } = {}) {
    this.container = container;
    this.onSelect = onSelect;
    this.onHover = onHover;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#090d17');
    this.scene.fog = new THREE.FogExp2('#090d17', 0.014);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 150);
    this.camera.position.set(15, 14, 21);
    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, powerPreference: 'low-power' });
    this.renderer.setPixelRatio(1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    container.append(this.renderer.domElement);
    this.renderer.domElement.setAttribute('aria-label', 'Interactive 3D model. Drag to orbit, scroll to zoom, click a residual node to inspect its token.');
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 70;
    this.controls.maxPolarAngle = Math.PI * 0.93;
    this.controls.target.set(0, 2.5, 0);
    this.dirty = true;
    this.lastTickTime = 0;
    this._controlChange = () => { this.dirty = true; };
    this.controls.addEventListener('change', this._controlChange);
    this.scene.add(new THREE.HemisphereLight('#c9deff', '#131329', 2));
    const key = new THREE.DirectionalLight('#d6e8ff', 3.5);
    key.position.set(4, 10, 7);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight('#a790ff', 2);
    fill.position.set(-8, 3, -6);
    this.scene.add(fill);
    this.world = new THREE.Group();
    this.scene.add(this.world);
    this.grid = new THREE.GridHelper(44, 44, '#253952', '#142034');
    this.grid.position.y = -1.4;
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.35;
    this.scene.add(this.grid);
    this.nodeGeometry = new THREE.IcosahedronGeometry(0.19, 2);
    this.nodeMaterial = new THREE.MeshStandardMaterial({ color: 'white', roughness: 0.28, metalness: 0.35, emissive: '#1a304c', emissiveIntensity: 0.3 });
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.pickables = [];
    this.positions = [];
    this.intervals = [];
    this.lastTime = 0;
    this._visibility = () => { this.lastTime = 0; };
    document.addEventListener('visibilitychange', this._visibility);
    this.frameCount = 0;
    this.motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.transitions = [];
    this.scratchMatrix = new THREE.Matrix4();
    this.scratchPosition = new THREE.Vector3();
    this.scratchScale = new THREE.Vector3();
    this.scratchQuaternion = new THREE.Quaternion();
    this.scratchColor = new THREE.Color();
    this.cameraGoal = null;
    this.targetGoal = null;
    this.disposed = false;
    this.pointerDown = null;
    this._down = event => { this.pointerDown = { x: event.clientX, y: event.clientY }; this.cameraGoal = null; };
    this._click = event => {
      if (!this.pointerDown || Math.hypot(event.clientX - this.pointerDown.x, event.clientY - this.pointerDown.y) > 5) return;
      const pick = this.pick(event);
      if (pick) this.onSelect(pick);
    };
    this._move = event => {
      const pick = this.pick(event);
      this.renderer.domElement.style.cursor = pick ? 'pointer' : 'grab';
      this.onHover(pick ? { ...pick, x: event.clientX, y: event.clientY } : null);
    };
    this._leave = () => this.onHover(null);
    this.renderer.domElement.addEventListener('pointerdown', this._down);
    this.renderer.domElement.addEventListener('click', this._click);
    this.renderer.domElement.addEventListener('pointermove', this._move);
    this.renderer.domElement.addEventListener('pointerleave', this._leave);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(container);
    this.resize();
    this._animate = time => {
      if (this.disposed) return;
      this.animation = requestAnimationFrame(this._animate);
      if (document.visibilityState !== 'visible' || time - this.lastTickTime < 1000 / 15) return;
      this.lastTickTime = time;
      if (this.controls.update()) this.dirty = true;
      if (!this.dirty && !this.cameraGoal && !this.transitions.length) return;
      const delta = this.lastTime ? time - this.lastTime : 0;
      if (delta > 0 && document.visibilityState === 'visible') {
        this.intervals.push(delta);
        if (this.intervals.length > 600) this.intervals.shift();
      }
      this.lastTime = time;
      this.updatePresentation(time);
      if (this.cameraGoal) {
        const blend = this.motionPreference.matches ? 1 : 1 - Math.exp(-Math.min(delta || 16, 50) / 180);
        this.camera.position.lerp(this.cameraGoal, blend);
        this.controls.target.lerp(this.targetGoal, blend);
        if (this.camera.position.distanceTo(this.cameraGoal) < 0.01) this.cameraGoal = null;
      }
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.frameCount++;
      this.dirty = false;
    };
    this.animation = requestAnimationFrame(this._animate);
  }

  label(text, color = '#b2c6e0', width = 3.2, size = 30) {
    const canvas = document.createElement('canvas');
    let ctx = canvas.getContext('2d');
    ctx.font = `500 ${size}px ui-monospace, monospace`;
    canvas.width = Math.ceil(ctx.measureText(text).width) + 34;
    canvas.height = 72;
    ctx = canvas.getContext('2d');
    ctx.font = `500 ${size}px ui-monospace, monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = 'rgba(9, 15, 26, 0.84)';
    ctx.beginPath(); ctx.roundRect(0, 0, canvas.width, canvas.height, 9); ctx.fill();
    ctx.fillStyle = color;
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false, depthTest: false }));
    sprite.scale.set(width, width * canvas.height / canvas.width, 1);
    return sprite;
  }

  clearWorld() {
    this.world.traverse(obj => {
      if (obj.geometry && obj.geometry !== this.nodeGeometry) obj.geometry.dispose();
      const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
      materials.forEach(material => {
        if (material && material !== this.nodeMaterial) { material.map?.dispose(); material.dispose(); }
      });
      if (obj.isInstancedMesh) obj.dispose();
    });
    this.world.clear();
    this.pickables = [];
  }

  setFrame(frame, selection = {}) {
    this.dirty = true;
    const previous = this.frame;
    const prior = this.pickables.map(mesh => {
      const items = [];
      for (let i = 0; i < mesh.count; i++) {
        mesh.getMatrixAt(i, this.scratchMatrix);
        this.scratchMatrix.decompose(this.scratchPosition, this.scratchQuaternion, this.scratchScale);
        mesh.getColorAt(i, this.scratchColor);
        items.push({ position: this.scratchPosition.clone(), scale: this.scratchScale.x, color: this.scratchColor.clone() });
      }
      return items;
    });
    this.frame = frame;
    this.selection = { layer: 0, head: 0, token: 0, threshold: 0.03, showAttention: true, showResiduals: true, ...selection };
    this.clearWorld();
    this.transitions = [];
    this.connector = null;
    this.presentationPositions = [];
    this.transitionStarted = performance.now();
    if (!frame) return;
    const n = frame.tokens.length;
    const columns = Math.min(n, 12);
    const rows = Math.ceil(n / columns);
    const spacing = 1.08;
    const width = Math.max(5, columns * spacing + 1.9);
    const depth = Math.max(3, rows * spacing + 1.8);
    const matrix = new THREE.Matrix4();
    const color = new THREE.Color();
    const maximum = Math.max(1e-6, ...frame.layers.flatMap(l => l.norms));
    const firstBlock = frame.layers[0].index;
    const blockSpan = Math.max(1, frame.layers[frame.layers.length - 1].index - firstBlock);
    this.positions = frame.layers.map((layer, li) => frame.tokens.map((_, ti) => new THREE.Vector3((ti % columns - (columns - 1) / 2) * spacing, li * 3.1, (Math.floor(ti / columns) - (rows - 1) / 2) * spacing)));
    frame.layers.forEach((layer, li) => {
      const plate = new THREE.Mesh(new THREE.BoxGeometry(width, 0.055, depth), new THREE.MeshStandardMaterial({ color: '#18253d', metalness: 0.25, roughness: 0.5, transparent: true, opacity: li === this.selection.layer ? 0.65 : 0.28, depthWrite: false }));
      plate.position.y = li * 3.1 - 0.28;
      this.world.add(plate);
      const rim = new THREE.LineSegments(new THREE.EdgesGeometry(plate.geometry), new THREE.LineBasicMaterial({ color: li === this.selection.layer ? '#5c8eaa' : '#29364e', transparent: true, opacity: 0.7 }));
      rim.position.copy(plate.position); this.world.add(rim);
      const name = this.label(`BLOCK ${String(layer.index + 1).padStart(2, '0')}`, li === this.selection.layer ? '#8aebf0' : '#8e9db9', 2.0, 30);
      name.position.set(-width / 2 - 1.0, li * 3.1 + 0.16, -depth / 2 - 0.1);
      this.world.add(name);
      const nodes = new THREE.InstancedMesh(this.nodeGeometry, this.nodeMaterial, n);
      nodes.userData.layer = li;
      nodes.visible = this.selection.showResiduals !== false;
      const transitions = [];
      this.presentationPositions[li] = [];
      for (let ti = 0; ti < n; ti++) {
        const ratio = layer.norms[ti] / maximum;
        const selected = li === this.selection.layer && ti === this.selection.token;
        const scale = selected ? 1.65 : 0.7 + 0.7 * ratio;
        color.copy(CYAN).lerp(VIOLET, (layer.index - firstBlock) / blockSpan).multiplyScalar(0.45 + 0.65 * ratio);
        if (selected) color.copy(GOLD);
        const old = previous?.input_ids[ti] === frame.input_ids[ti] ? prior[li]?.[ti] : null;
        const from = old || { position: this.positions[li][ti], scale: 0.2, color: color.clone().multiplyScalar(0.3) };
        const initial = this.motionPreference.matches ? { position: this.positions[li][ti], scale, color } : from;
        matrix.makeScale(initial.scale, initial.scale, initial.scale); matrix.setPosition(initial.position);
        nodes.setMatrixAt(ti, matrix); nodes.setColorAt(ti, initial.color);
        this.presentationPositions[li][ti] = initial.position.clone();
        transitions.push({ from, position: this.positions[li][ti], scale, color: color.clone() });
      }
      this.transitions.push({ mesh: nodes, items: transitions });
      nodes.instanceMatrix.needsUpdate = true; nodes.instanceColor.needsUpdate = true;
      nodes.computeBoundingSphere();
      this.world.add(nodes); this.pickables.push(nodes);
    });
    // Thin traces identify the same token at successive captured blocks. These
    // schematic identity links can span uncaptured blocks; they do not show
    // measured attention, intervening activation states or a causal circuit.
    const traceVertices = [], traceColors = [];
    for (let li = 0; li < frame.layers.length - 1; li++) {
      for (let ti = 0; ti < n; ti++) {
        const a = this.presentationPositions[li][ti], b = this.presentationPositions[li + 1][ti];
        const selected = ti === this.selection.token;
        const c = new THREE.Color(selected ? '#a08d69' : '#253c57');
        traceVertices.push(a.x, a.y, a.z, b.x, b.y, b.z);
        traceColors.push(c.r, c.g, c.b, c.r, c.g, c.b);
      }
    }
    const traceGeometry = new THREE.BufferGeometry();
    traceGeometry.setAttribute('position', new THREE.Float32BufferAttribute(traceVertices, 3));
    traceGeometry.setAttribute('color', new THREE.Float32BufferAttribute(traceColors, 3));
    this.connector = new THREE.LineSegments(traceGeometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, depthWrite: false }));
    this.world.add(this.connector);
    frame.tokens.forEach((token, ti) => {
      const escaped = token.replaceAll(' ', '·').replaceAll('\n', '↵');
      const tokenText = `${ti} ${escaped}`;
      const text = this.label(tokenText, ti === this.selection.token ? '#ffdaaa' : '#b5c8e2', Math.min(1.65, 0.15 * tokenText.length + 0.16), 31);
      text.position.copy(this.positions[0][ti]); text.position.y -= 0.72;
      text.position.z += 0.5;
      this.world.add(text);
    });
    const li = Math.min(this.selection.layer, frame.layers.length - 1);
    const qi = Math.min(this.selection.token, n - 1);
    const hi = Math.min(this.selection.head, frame.layers[li].attention.length - 1);
    const origin = this.positions[li][qi];
    if (this.selection.showAttention) {
      const vertices = [], colors = [];
      const arrows = [];
      const row = frame.layers[li].attention[hi][qi];
      row.forEach((weight, ki) => {
        if (weight < this.selection.threshold || weight <= 0) return;
        const endpoint = this.positions[li][ki];
        const distance = origin.distanceTo(endpoint);
        const middle = origin.clone().add(endpoint).multiplyScalar(0.5);
        middle.y += ki === qi ? 0.8 : 0.45 + Math.min(1.65, distance * 0.18);
        if (ki === qi) middle.x += 0.55;
        const end = endpoint.clone(); end.y += 0.08;
        const curve = new THREE.QuadraticBezierCurve3(origin, middle, end);
        const edgeColor = CYAN.clone().lerp(GOLD, weight).multiplyScalar(0.28 + weight * 1.2);
        let a = curve.getPoint(0);
        for (let s = 1; s <= 24; s++) {
          const b = curve.getPoint(s / 24);
          vertices.push(a.x, a.y, a.z, b.x, b.y, b.z);
          colors.push(edgeColor.r, edgeColor.g, edgeColor.b, edgeColor.r, edgeColor.g, edgeColor.b);
          a = b;
        }
        const arrowPosition = curve.getPoint(0.85);
        const direction = curve.getTangent(0.85).normalize();
        arrows.push({ position: arrowPosition, direction, color: edgeColor, scale: 0.75 + weight * 0.65 });
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      this.world.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 })));
      if (arrows.length) {
        const tips = new THREE.InstancedMesh(new THREE.ConeGeometry(0.052, 0.17, 6), new THREE.MeshBasicMaterial({ color: 'white' }), arrows.length);
        const up = new THREE.Vector3(0, 1, 0);
        arrows.forEach((arrow, index) => {
          const rotation = new THREE.Quaternion().setFromUnitVectors(up, arrow.direction);
          matrix.compose(arrow.position, rotation, new THREE.Vector3(arrow.scale, arrow.scale, arrow.scale));
          tips.setMatrixAt(index, matrix); tips.setColorAt(index, arrow.color);
        });
        tips.instanceMatrix.needsUpdate = true; tips.instanceColor.needsUpdate = true;
        this.world.add(tips);
      }
    }
    // Selection and picking use local capture-array indices. The actual head
    // identity is supplied separately for sparse captures of larger models.
    const displayHead = Number.isInteger(this.selection.displayHead) ? this.selection.displayHead : hi;
    const selectedText = this.label(`QUERY ${qi} · H${displayHead + 1}`, '#ffdaaa', 2.2, 30);
    selectedText.position.set(width / 2 + 1.3, li * 3.1 + 0.2, depth / 2 + 0.1);
    this.world.add(selectedText);
    this.center = new THREE.Vector3(0, (frame.layers.length - 1) * 1.55, 0);
    this.extent = Math.max(width, depth, frame.layers.length * 3.1);
  }

  /** Only visual transforms/colors interpolate; captured tensors stay exact. */
  updatePresentation(time) {
    if (!this.transitions.length) return;
    const t = this.motionPreference.matches ? 1 : Math.min(1, Math.max(0, (time - this.transitionStarted) / 220));
    const blend = t * t * (3 - 2 * t);
    this.transitions.forEach(({ mesh, items }, li) => {
      items.forEach((item, ti) => {
        const position = this.presentationPositions[li][ti];
        position.lerpVectors(item.from.position, item.position, blend);
        const scale = THREE.MathUtils.lerp(item.from.scale, item.scale, blend);
        this.scratchMatrix.makeScale(scale, scale, scale); this.scratchMatrix.setPosition(position);
        mesh.setMatrixAt(ti, this.scratchMatrix);
        this.scratchColor.lerpColors(item.from.color, item.color, blend);
        mesh.setColorAt(ti, this.scratchColor);
      });
      mesh.instanceMatrix.needsUpdate = true; mesh.instanceColor.needsUpdate = true;
      mesh.computeBoundingSphere();
    });
    if (this.connector) {
      const attribute = this.connector.geometry.getAttribute('position');
      let index = 0;
      for (let li = 0; li < this.presentationPositions.length - 1; li++) {
        for (let ti = 0; ti < this.presentationPositions[li].length; ti++) {
          const a = this.presentationPositions[li][ti], b = this.presentationPositions[li + 1][ti];
          attribute.setXYZ(index++, a.x, a.y, a.z); attribute.setXYZ(index++, b.x, b.y, b.z);
        }
      }
      attribute.needsUpdate = true;
    }
    if (t === 1) this.transitions = [];
  }

  pick(event) {
    if (!this.frame) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hit = this.raycaster.intersectObjects(this.pickables.filter(p => p.visible), false)[0];
    return hit && hit.instanceId !== undefined ? { layer: hit.object.userData.layer, token: hit.instanceId } : null;
  }

  setCamera(preset = 'overview') {
    this.dirty = true;
    const center = this.center?.clone() || new THREE.Vector3(0, 3, 0);
    const size = Math.max(12, this.extent || 14);
    const offsets = { overview: new THREE.Vector3(size * 0.85, size * 0.65, size * 1.12), front: new THREE.Vector3(0, size * 0.2, size * 1.65), top: new THREE.Vector3(0, size * 1.6, 0.1) };
    this.targetGoal = center;
    this.cameraGoal = center.clone().add(offsets[preset] || offsets.overview);
  }

  focusSelection() {
    this.dirty = true;
    if (!this.frame) return;
    const { layer, token } = this.selection;
    const point = this.positions[layer]?.[token];
    if (!point) return;
    this.targetGoal = point.clone();
    this.cameraGoal = point.clone().add(new THREE.Vector3(3.5, 3.2, 6));
  }

  resize() {
    this.dirty = true;
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  getStats() {
    const sorted = [...this.intervals].sort((a, b) => a - b);
    const average = sorted.length ? sorted.reduce((a, b) => a + b, 0) / sorted.length : null;
    return { frames: this.frameCount, sampleCount: sorted.length, meanFrameMs: average, p95FrameMs: sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] : null, fps: average ? 1000 / average : null, maxFps: 15, renderOnDemand: true, drawCalls: this.renderer.info.render.calls, triangles: this.renderer.info.render.triangles, geometries: this.renderer.info.memory.geometries, textures: this.renderer.info.memory.textures, nodes: this.frame ? this.frame.tokens.length * this.frame.layers.length : 0, renderer: this.renderer.getContext().getParameter(this.renderer.getContext().RENDERER) };
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.animation);
    this.observer.disconnect();
    document.removeEventListener('visibilitychange', this._visibility);
    this.renderer.domElement.removeEventListener('pointerdown', this._down);
    this.renderer.domElement.removeEventListener('click', this._click);
    this.renderer.domElement.removeEventListener('pointermove', this._move);
    this.renderer.domElement.removeEventListener('pointerleave', this._leave);
    this.controls.removeEventListener('change', this._controlChange);
    this.controls.dispose(); this.clearWorld();
    this.nodeGeometry.dispose(); this.nodeMaterial.dispose();
    this.grid.geometry.dispose(); this.grid.material.dispose();
    this.renderer.dispose(); this.renderer.domElement.remove();
  }
}
