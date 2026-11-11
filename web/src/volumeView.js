// Data view for regular grids: direct volume rendering with Three.js.
//
// A box is drawn around the volume and a fragment shader marches a ray through
// it, front to back, reading a 3D texture of the field values (the structure of
// three.js's own webgl2_materials_texture3d example). A second 3D texture holds,
// per voxel, the colour of the peak whose region contains it (a distinct colour per peak, as
// the dots in the spine) and a weight: when peaks are selected in the spine, voxels of all other
// regions get a small weight, so the selected region stands out inside the volume.
//
// Linking, as in the scatterplot (dataView.js):
//   * ``setSelectedPeaks`` highlights regions chosen in the spine;
//   * clicking the volume reports the voxel under the pointer (``onPick``), found
//     on the CPU with the same transfer function as the shader (``pickVoxel``).
// Small markers show where the visible peaks are.
//
// Colour modes: 'region' (peak colour), 'value' (colour map), and 'iso' (iso-value
// bands): the value range is cut at a list of iso values, and every voxel takes the
// flat colour of the band it falls in, so regions between consecutive isosurfaces are
// told apart. The levels are either the critical values of the visible peaks and
// saddles (the same isovalues as the paper-style spine, ``sharedLevels``) or N even
// bands. Optionally the iso surfaces themselves get extra opacity ('add'), or only
// they are drawn ('only'), which shows the nested shells.
//
// Drag to rotate, wheel to zoom, double-click to reset.

import * as THREE from 'three';
import { peakColor, regionColor } from './spineView.js';
import { sharedLevels } from './contours.js';

export const RAMP = 0.15;        // value range over which a voxel fades in above the threshold
export const DENSITY = 24;       // opacity per unit length (box side = 1) at full density
const DIM = 0.1;                 // weight of unselected regions while a selection exists
export const ISO_BINS = 1024;    // resolution of the iso-band table (value -> band colour, surface weight)
export const ISO_WIDTH = 0.006;  // half-width, in value, of the opacity spike at each iso surface
export const ISO_GAIN = 3;       // opacity of an iso surface relative to a fully faded-in voxel

/**
 * Iso values (sorted, strictly inside (lo, hi)) that cut [lo, hi] into bands.
 * ``source`` 'spine': the critical values of the visible peaks and saddles, thinned as in
 * the paper-style spine (``sharedLevels``); a number n: n even bands.
 */
export function isoLevels(doc, vis, source, { gap = 0.02, maxLevels = 40 } = {}) {
  const [lo, hi] = doc.meta.value_range ?? [0, 1];
  if (source !== 'spine') {
    const n = Math.max(2, Math.round(Number(source) || 8));
    return Array.from({ length: n - 1 }, (_, k) => lo + ((k + 1) * (hi - lo)) / n);
  }
  if (!vis) return [];
  const crit = [...vis.saddles.map((id) => doc.saddles[id].value), ...vis.peaks.map((id) => doc.peaks[id].value)];
  return sharedLevels(crit, lo, hi, { gap, maxLevels }).filter((c) => c > lo && c < hi);
}

/** Band of value ``v``: the number of levels at or below it (0 .. levels.length). */
export function bandOf(v, levels) {
  let k = 0;
  while (k < levels.length && levels[k] <= v) k++;
  return k;
}

/** Colour ([r, g, b] sRGB) of band ``k`` of ``n``: evenly spread over the value colour map. */
export function bandColor(k, n) { return peakColor(n > 1 ? k / (n - 1) : 0.5); }

/**
 * Table over values [0, 1] (the volume texture's range), ``bins`` entries:
 * ``rgba`` (Uint8, for the shader) holds the band colour and, in alpha, how close the
 * value is to an iso level (1 on it, 0 farther than ``width``); ``surf`` is that
 * closeness as floats, for picking on the CPU.
 */
export function isoTable(levels, { bins = ISO_BINS, width = ISO_WIDTH } = {}) {
  const rgba = new Uint8Array(bins * 4), surf = new Float32Array(bins);
  const n = levels.length + 1;
  const cols = Array.from({ length: n }, (_, k) => bandColor(k, n).map((c) => Math.round(c * 255)));
  for (let b = 0; b < bins; b++) {
    const v = (b + 0.5) / bins;
    const k = bandOf(v, levels);
    let d = Infinity;
    for (const c of levels) d = Math.min(d, Math.abs(v - c));
    const w = Math.max(0, 1 - d / width);
    surf[b] = w;
    rgba.set([...cols[k], Math.round(w * 255)], 4 * b);
  }
  return { rgba, surf, bands: n };
}

/** Smooth fade-in of opacity with value: 0 at ``thresh``, 1 at ``thresh + RAMP``. */
export function rampOf(v, thresh) {
  const t = Math.min(1, Math.max(0, (v - thresh) / RAMP));
  return t * t * (3 - 2 * t);
}

/** Ray against the box [-0.5, 0.5]^3: returns [t0, t1] (t0 > t1: no hit). */
export function rayBox(o, d) {
  let t0 = -Infinity, t1 = Infinity;
  for (let k = 0; k < 3; k++) {
    const inv = 1 / d[k];
    const a = (-0.5 - o[k]) * inv, b = (0.5 - o[k]) * inv;
    t0 = Math.max(t0, Math.min(a, b));
    t1 = Math.min(t1, Math.max(a, b));
  }
  return [t0, t1];
}

/**
 * Voxel the user means when pointing along a ray, or -1 if the ray sees nothing.
 * It is the first voxel at which the accumulated opacity reaches ``stop``. A thin
 * feature may never get that opaque, so otherwise the voxel that contributed most
 * is returned, provided the ray accumulated at least ``minAlpha`` (faintly visible).
 * ``surf`` / ``surfMode`` add the iso-surface opacity as the shader does (``isoTable``).
 * @param values  flat voxel values in [0, 1], index = x + nx * (y + ny * z)
 * @param shape   [nz, ny, nx]
 * @param o, d    ray in box coordinates (box = [-0.5, 0.5]^3), d a unit vector
 */
export function pickVoxel(values, shape, o, d, { thresh, density = 1, steps = 384, stop = 0.35, minAlpha = 0.05,
                                                  surf = null, surfMode = 'off' }) {
  const nb = surf?.length ?? 0;
  const useSurf = nb > 0 && surfMode !== 'off', only = useSurf && surfMode === 'only';
  const [nz, ny, nx] = shape;
  const [t0, t1] = rayBox(o, d);
  if (!(t0 <= t1) || t1 < 0) return -1;
  const start = Math.max(t0, 0), dt = (t1 - start) / steps;
  let alpha = 0, best = -1, bestGain = 0;
  for (let s = 0; s < steps; s++) {
    const t = start + (s + 0.5) * dt;
    const x = Math.min(nx - 1, Math.max(0, Math.floor((o[0] + t * d[0] + 0.5) * nx)));
    const y = Math.min(ny - 1, Math.max(0, Math.floor((o[1] + t * d[1] + 0.5) * ny)));
    const z = Math.min(nz - 1, Math.max(0, Math.floor((o[2] + t * d[2] + 0.5) * nz)));
    const i = x + nx * (y + ny * z);
    const v = values[i];
    let op = only ? 0 : rampOf(v, thresh);
    if (useSurf && v >= thresh) op += ISO_GAIN * surf[Math.min(nb - 1, Math.max(0, Math.floor(v * nb)))];
    const sigma = density * DENSITY * op;
    if (sigma > 0) {
      const gain = (1 - alpha) * (1 - Math.exp(-sigma * dt));
      alpha += gain;
      if (alpha >= stop) return i;
      if (gain > bestGain) { bestGain = gain; best = i; }
    }
  }
  return alpha >= minAlpha ? best : -1;
}

const VERTEX = /* glsl */ `
in vec3 position;
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 cameraPos;
out vec3 vOrigin;
out vec3 vDirection;
void main() {
  vOrigin = vec3(inverse(modelMatrix) * vec4(cameraPos, 1.0));
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const FRAGMENT = /* glsl */ `
precision highp float;
precision highp sampler3D;
in vec3 vOrigin;
in vec3 vDirection;
out vec4 outColor;

uniform sampler3D uVolume;   // field value, R8
uniform sampler3D uLabel;    // rgb: colour of the voxel's peak, a: weight (1, or small if not selected)
uniform sampler2D uLut;      // 256 x 1: value -> colour
uniform float uSteps;
uniform float uThresh;
uniform float uDensity;
uniform float uByRegion;     // 1: colour by peak region, 0: by value
uniform float uIso;          // 1: colour by iso-value band (overrides uByRegion)
uniform float uIsoSurf;      // 0: no iso surfaces, 1: added to the volume, 2: surfaces only
uniform sampler2D uIsoLut;   // ISO_BINS x 1: rgb band colour, a: closeness to an iso level
uniform float uLight;        // 1: shade with the gradient of the field
uniform vec3 uVoxel;         // 1 / grid size, for the gradient

vec2 hitBox(vec3 orig, vec3 dir) {
  vec3 invDir = 1.0 / dir;
  vec3 a = (vec3(-0.5) - orig) * invDir;
  vec3 b = (vec3(0.5) - orig) * invDir;
  vec3 tmin = min(a, b);
  vec3 tmax = max(a, b);
  float t0 = max(tmin.x, max(tmin.y, tmin.z));
  float t1 = min(tmax.x, min(tmax.y, tmax.z));
  return vec2(t0, t1);
}

void main() {
  vec3 rayDir = normalize(vDirection);
  vec2 bounds = hitBox(vOrigin, rayDir);
  if (bounds.x > bounds.y) discard;
  bounds.x = max(bounds.x, 0.0);

  vec3 inc = 1.0 / abs(rayDir);
  float delta = min(inc.x, min(inc.y, inc.z)) / uSteps;   // one voxel-ish step along the ray
  vec3 p = vOrigin + bounds.x * rayDir;

  vec4 acc = vec4(0.0);
  for (float t = bounds.x; t < bounds.y; t += delta) {
    vec3 q = p + 0.5;
    float v = texture(uVolume, q).r;
    float s = clamp((v - uThresh) / ${RAMP.toFixed(2)}, 0.0, 1.0);
    s = s * s * (3.0 - 2.0 * s);
    vec4 iso = texture(uIsoLut, vec2(v, 0.5));
    if (uIsoSurf > 1.5) s = 0.0;
    if (uIsoSurf > 0.5 && v >= uThresh) s += ${ISO_GAIN.toFixed(1)} * iso.a;
    vec4 lab = texture(uLabel, q);
    float sigma = uDensity * ${DENSITY.toFixed(1)} * s * lab.a;
    if (sigma > 0.0) {
      float a = 1.0 - exp(-sigma * delta);
      vec3 c = uIso > 0.5 ? iso.rgb : uByRegion > 0.5 ? lab.rgb : texture(uLut, vec2(v, 0.5)).rgb;
      if (uLight > 0.5) {
        vec3 g = vec3(
          texture(uVolume, q + vec3(uVoxel.x, 0.0, 0.0)).r - texture(uVolume, q - vec3(uVoxel.x, 0.0, 0.0)).r,
          texture(uVolume, q + vec3(0.0, uVoxel.y, 0.0)).r - texture(uVolume, q - vec3(0.0, uVoxel.y, 0.0)).r,
          texture(uVolume, q + vec3(0.0, 0.0, uVoxel.z)).r - texture(uVolume, q - vec3(0.0, 0.0, uVoxel.z)).r);
        float gl = length(g);
        if (gl > 1e-4) {
          vec3 n = -g / gl;                                  // faces away from higher values
          c *= 0.5 + 0.5 * abs(dot(n, rayDir));
        }
      }
      acc.rgb += (1.0 - acc.a) * a * c;
      acc.a += (1.0 - acc.a) * a;
      if (acc.a > 0.97) break;
    }
    p += rayDir * delta;
  }
  if (acc.a < 0.005) discard;
  outColor = acc;                                            // premultiplied alpha
}`;

export class VolumeView {
  /**
   * @param {HTMLElement} container
   * @param {{onPick?: (voxelIndex: number|null) => void}} handlers
   * @param {{renderer?: object}} options ``renderer`` replaces WebGL (tests have no GPU)
   */
  constructor(container, handlers = {}, options = {}) {
    this.container = container;
    this.handlers = handlers;
    this.renderer = options.renderer ?? new THREE.WebGLRenderer({ antialias: true });
    if (!options.renderer) this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.touchAction = 'none';

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 20);
    this.home = { theta: 0.7, phi: 1.1, dist: 2.1 };
    this.view = { ...this.home };                  // azimuth, polar angle from +y, distance
    this.colors = { marker: [0.1, 0.1, 0.1], select: [0.15, 0.39, 0.92] };

    this.doc = null;
    this.vis = null;
    this.selectedPeaks = new Set();
    this.colorBy = 'region';
    this.thresh = 0.3;
    this.density = 1;
    this.light = true;
    this.isoSource = 'spine';                      // 'spine' or a number of even bands
    this.isoSurf = 'off';                          // 'off' | 'add' | 'only'
    this.levels = [];
    this.dirty = true;
    this.#bindEvents();
    this.resize();
    this.#loop();
  }

  /** True if ``doc`` is a regular grid this view can draw. */
  static supports(doc) {
    const d = doc.data;
    return doc.meta.kind === 'grid' && Boolean(d?.shape && d.values && doc.region);
  }

  /** Load the volume of ``doc``; returns false (and draws nothing) if it is not a grid. */
  setGraph(doc) {
    this.#dispose();
    this.doc = doc;
    this.vis = null;
    this.selectedPeaks = new Set();
    if (!VolumeView.supports(doc)) return false;

    const [nz, ny, nx] = doc.data.shape;
    this.shape = [nz, ny, nx];
    this.n = nx * ny * nz;
    const sp = doc.data.spacing ?? [1, 1, 1];
    const phys = [nx * sp[0], ny * sp[1], nz * sp[2]];
    const big = Math.max(...phys);
    this.extent = phys.map((p) => p / big);          // box side lengths, longest = 1

    const values = doc.data.values;
    const u8 = new Uint8Array(this.n);
    const hist = new Uint32Array(256);
    for (let i = 0; i < this.n; i++) {
      const b = Math.min(255, Math.max(0, Math.round(values[i] * 255)));
      u8[i] = b; hist[b]++;
    }
    this.thresh = Math.max(0.02, 0.25 * this.#quantile(hist, 0.99));   // a quarter of the bright end

    this.volumeTex = new THREE.Data3DTexture(u8, nx, ny, nz);
    this.volumeTex.format = THREE.RedFormat;
    this.volumeTex.type = THREE.UnsignedByteType;
    this.volumeTex.minFilter = this.volumeTex.magFilter = THREE.LinearFilter;
    this.volumeTex.wrapS = this.volumeTex.wrapT = this.volumeTex.wrapR = THREE.ClampToEdgeWrapping;
    this.volumeTex.unpackAlignment = 1;
    this.volumeTex.needsUpdate = true;

    this.label = new Uint8Array(this.n * 4).fill(255);
    this.labelTex = new THREE.Data3DTexture(this.label, nx, ny, nz);
    this.labelTex.format = THREE.RGBAFormat;
    this.labelTex.type = THREE.UnsignedByteType;
    this.labelTex.minFilter = this.labelTex.magFilter = THREE.LinearFilter;
    this.labelTex.wrapS = this.labelTex.wrapT = this.labelTex.wrapR = THREE.ClampToEdgeWrapping;
    this.labelTex.needsUpdate = true;

    this.lutTex = new THREE.DataTexture(new Uint8Array(256 * 4), 256, 1, THREE.RGBAFormat);
    this.lutTex.minFilter = this.lutTex.magFilter = THREE.LinearFilter;
    this.#fillLut();

    this.isoTex = new THREE.DataTexture(new Uint8Array(ISO_BINS * 4), ISO_BINS, 1, THREE.RGBAFormat);
    this.isoTex.minFilter = this.isoTex.magFilter = THREE.NearestFilter;   // crisp band edges
    this.#fillIso();

    this.material = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      side: THREE.BackSide,                       // draw the far faces so the camera may sit inside
      transparent: true,
      premultipliedAlpha: true,
      depthWrite: false,
      uniforms: {
        uVolume: { value: this.volumeTex }, uLabel: { value: this.labelTex }, uLut: { value: this.lutTex },
        uIsoLut: { value: this.isoTex },
        uIso: { value: this.colorBy === 'iso' ? 1 : 0 }, uIsoSurf: { value: { off: 0, add: 1, only: 2 }[this.isoSurf] },
        cameraPos: { value: new THREE.Vector3() },
        uSteps: { value: Math.max(nx, ny, nz) * 1.5 },
        uThresh: { value: this.thresh }, uDensity: { value: this.density },
        uByRegion: { value: this.colorBy === 'region' ? 1 : 0 }, uLight: { value: this.light ? 1 : 0 },
        uVoxel: { value: new THREE.Vector3(1 / nx, 1 / ny, 1 / nz) },
      },
    });
    this.box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), this.material);
    this.box.scale.set(...this.extent);
    this.scene.add(this.box);

    this.markerGeom = new THREE.BufferGeometry();
    this.markers = new THREE.Points(this.markerGeom, new THREE.PointsMaterial({
      size: 7, sizeAttenuation: false, vertexColors: true, depthTest: false, transparent: true }));
    this.markers.renderOrder = 2;
    this.markers.frustumCulled = false;
    this.scene.add(this.markers);

    this.reset();
    return true;
  }

  /** Where the field value gets visible (0..1); 0.3 hides the lowest ~30 % of values. */
  setThreshold(t) { this.thresh = t; if (this.material) this.material.uniforms.uThresh.value = t; this.dirty = true; }
  getThreshold() { return this.thresh; }

  /** Opacity multiplier (0.1 .. 3). */
  setDensity(k) { this.density = k; if (this.material) this.material.uniforms.uDensity.value = k; this.dirty = true; }

  /** Shade with the gradient of the field (surfaces look solid) or not. */
  setShading(on) { this.light = on; if (this.material) this.material.uniforms.uLight.value = on ? 1 : 0; this.dirty = true; }

  /** Refill the value -> colour table from ``peakColor`` (after ``setColormap``). */
  #fillLut() {
    const lut = this.lutTex.image.data;
    for (let i = 0; i < 256; i++) {
      const [r, g, b] = peakColor(i / 255);
      lut.set([r * 255, g * 255, b * 255, 255], i * 4);
    }
    this.lutTex.needsUpdate = true;
  }

  /** Recolour with the active value colour map. */
  refreshColors() { if (this.lutTex) { this.#fillLut(); this.#fillIso(); } this.dirty = true; }

  /** Colour by 'region' (peak colour), 'value' (field value) or 'iso' (iso-value bands). */
  setColorBy(mode) {
    this.colorBy = mode === 'value' || mode === 'iso' ? mode : 'region';
    if (this.material) {
      this.material.uniforms.uByRegion.value = this.colorBy === 'region' ? 1 : 0;
      this.material.uniforms.uIso.value = this.colorBy === 'iso' ? 1 : 0;
    }
    this.dirty = true;
  }

  /** Where the iso values come from: 'spine' (critical values of the visible peaks / saddles) or a band count. */
  setIsoLevels(source) { this.isoSource = source; if (this.isoTex) this.#fillIso(); this.dirty = true; }

  /** Iso surfaces: 'off', 'add' (extra opacity on each iso value) or 'only' (nothing else drawn). */
  setIsoSurfaces(mode) {
    this.isoSurf = mode === 'add' || mode === 'only' ? mode : 'off';
    if (this.material) this.material.uniforms.uIsoSurf.value = { off: 0, add: 1, only: 2 }[this.isoSurf];
    this.dirty = true;
  }

  /** The current iso values (sorted), for the legend. */
  getIsoLevels() { return this.levels; }

  /** Show the result of ``simplify(doc, t, p_hi)``: voxels take the colour of their (merged) peak. */
  setVisible(vis) {
    this.vis = vis;
    this.selectedPeaks = new Set([...this.selectedPeaks].filter((i) => i < vis.peaks.length));
    if (!this.material) return;
    if (this.isoSource === 'spine') this.#fillIso();          // the levels follow the simplification
    this.#updateLabel();
    this.#updateMarkers();
  }

  /** Highlight these visible peaks (indices into ``vis.peaks``); empty clears. */
  setSelectedPeaks(indices) {
    this.selectedPeaks = new Set(indices);
    if (!this.material) return;
    this.#updateLabel();
    this.#updateMarkers();
  }

  setTheme({ bg, marker, select }) {
    this.renderer.setClearColor(new THREE.Color().setRGB(bg[0], bg[1], bg[2], THREE.SRGBColorSpace), 1);
    if (marker) this.colors.marker = marker;
    if (select) this.colors.select = select;
    if (this.material) this.#updateMarkers();
    this.dirty = true;
  }

  reset() { this.view = { ...this.home }; this.dirty = true; }

  resize() {
    const w = Math.max(this.container.clientWidth, 10), h = Math.max(this.container.clientHeight, 10);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.dirty = true;
  }

  /** Voxel index under client position (x, y), or -1 if the ray sees nothing at the current threshold. */
  pickAt(clientX, clientY) {
    if (!this.material) return -1;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1,
                                  1 - ((clientY - rect.top) / rect.height) * 2);
    this.#placeCamera();
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const e = this.extent;
    const o = [ray.ray.origin.x / e[0], ray.ray.origin.y / e[1], ray.ray.origin.z / e[2]];
    const dv = new THREE.Vector3(ray.ray.direction.x / e[0], ray.ray.direction.y / e[1], ray.ray.direction.z / e[2]).normalize();
    return pickVoxel(this.doc.data.values, this.shape, o, [dv.x, dv.y, dv.z],
                     { thresh: this.thresh, density: this.density, surf: this.isoSurfW, surfMode: this.isoSurf });
  }

  /** Client position of a voxel's centre (for tests and tooltips). */
  voxelToClient([x, y, z]) {
    const [nz, ny, nx] = this.shape, e = this.extent;
    const p = new THREE.Vector3(((x + 0.5) / nx - 0.5) * e[0], ((y + 0.5) / ny - 0.5) * e[1], ((z + 0.5) / nz - 0.5) * e[2]);
    this.#placeCamera();
    this.camera.updateMatrixWorld();
    p.project(this.camera);
    const rect = this.renderer.domElement.getBoundingClientRect();
    return { x: rect.left + (p.x * 0.5 + 0.5) * rect.width, y: rect.top + (0.5 - p.y * 0.5) * rect.height };
  }

  dispose() {
    this.stopped = true;
    this.#dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // ---- internals --------------------------------------------------------
  #dispose() {
    for (const o of [this.box, this.markers]) if (o) { this.scene.remove(o); o.geometry.dispose(); o.material.dispose(); }
    for (const t of [this.volumeTex, this.labelTex, this.lutTex, this.isoTex]) t?.dispose();
    this.box = this.markers = this.material = this.volumeTex = this.labelTex = this.lutTex = this.isoTex = null;
    this.levels = []; this.isoSurfW = null;
  }

  /** Rebuild the iso-band table from the current levels and colour map. */
  #fillIso() {
    this.levels = isoLevels(this.doc, this.vis, this.isoSource);
    const { rgba, surf } = isoTable(this.levels);
    this.isoTex.image.data.set(rgba);
    this.isoTex.needsUpdate = true;
    this.isoSurfW = surf;
    this.dirty = true;
  }

  #quantile(hist, q) {
    let acc = 0;
    const goal = q * this.n;
    for (let b = 0; b < 256; b++) { acc += hist[b]; if (acc >= goal) return b / 255; }
    return 0.5;
  }

  /** Per voxel: colour of its visible peak, and a weight that dims unselected regions. */
  #updateLabel() {
    const { doc, vis, label } = this;
    const table = vis.peaks.map((id) => regionColor(id).map((c) => Math.round(c * 255)));
    const dimmed = this.selectedPeaks.size > 0;
    const weight = (i) => (!dimmed || this.selectedPeaks.has(i) ? 255 : Math.round(DIM * 255));
    const w = vis.peaks.map((_, i) => weight(i));
    const { region } = doc, owner = vis.owner;
    for (let i = 0; i < this.n; i++) {
      const k = owner[region[i]], c = table[k];
      label[4 * i] = c[0]; label[4 * i + 1] = c[1]; label[4 * i + 2] = c[2]; label[4 * i + 3] = w[k];
    }
    this.labelTex.needsUpdate = true;
    this.dirty = true;
  }

  #updateMarkers() {
    const { doc, vis } = this;
    const [nz, ny, nx] = this.shape, e = this.extent;
    const pos = new Float32Array(vis.peaks.length * 3), col = new Float32Array(vis.peaks.length * 3);
    const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    vis.peaks.forEach((id, i) => {
      const [x, y, z] = doc.peaks[id].position ?? [0, 0, 0];
      pos.set([((x + 0.5) / nx - 0.5) * e[0], ((y + 0.5) / ny - 0.5) * e[1], ((z + 0.5) / nz - 0.5) * e[2]], 3 * i);
      const c = (this.selectedPeaks.has(i) ? this.colors.select : this.colors.marker).map(lin);
      col.set(c, 3 * i);
    });
    this.markerGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.markerGeom.setAttribute('color', new THREE.BufferAttribute(col, 3));
    this.dirty = true;
  }

  #placeCamera() {
    const { theta, phi, dist } = this.view;
    this.camera.position.set(dist * Math.sin(phi) * Math.sin(theta), dist * Math.cos(phi), dist * Math.sin(phi) * Math.cos(theta));
    this.camera.lookAt(0, 0, 0);
    this.camera.updateMatrixWorld();
  }

  #bindEvents() {
    const el = this.renderer.domElement;
    let drag = null;
    el.addEventListener('pointerdown', (ev) => {
      el.setPointerCapture?.(ev.pointerId);
      drag = { x: ev.clientX, y: ev.clientY, moved: false, theta: this.view.theta, phi: this.view.phi };
    });
    el.addEventListener('pointermove', (ev) => {
      if (!drag) return;
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (Math.hypot(dx, dy) > 4) drag.moved = true;
      if (drag.moved) {
        this.view.theta = drag.theta - dx * 0.008;
        this.view.phi = Math.min(Math.PI - 0.05, Math.max(0.05, drag.phi - dy * 0.008));
        this.dirty = true;
      }
    });
    el.addEventListener('pointerup', (ev) => {
      if (drag && !drag.moved) {
        const i = this.pickAt(ev.clientX, ev.clientY);
        this.handlers.onPick?.(i < 0 ? null : i);
      }
      drag = null;
    });
    el.addEventListener('dblclick', () => this.reset());
    el.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      this.view.dist = Math.min(6, Math.max(0.9, this.view.dist * Math.exp(ev.deltaY * 0.0012)));
      this.dirty = true;
    }, { passive: false });
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
  }

  #loop() {
    if (this.stopped) return;
    if (this.dirty) {
      this.#placeCamera();
      if (this.material) this.material.uniforms.cameraPos.value.copy(this.camera.position);
      this.renderer.render(this.scene, this.camera);
      this.dirty = false;
    }
    requestAnimationFrame(() => this.#loop());
  }
}
