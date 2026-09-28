import './style.css';
import { icon } from './icons.js';
import { defaults, presets } from './presets/index.js';
import { World, type Diagnostics } from './runtime/world.js';
import {
  PARTICLE_LEVELS,
  particleCount,
  type ParticleLevel,
  type Preset,
  type Values,
} from './types.js';

const REPO = 'https://github.com/dgreenheck/threejs-particle-fluids';
const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <header class="topbar">
    <a class="brand" href="./" aria-label="Three.js Particle Fluids home"><img src="${import.meta.env.BASE_URL}favicon.svg" alt="" width="34" height="34"><span>Three.js <strong>Particle Fluids</strong></span></a>
    <nav class="top-links" aria-label="Project links">
      <a class="top-link" href="${REPO}/tree/main/docs#readme" target="_blank" rel="noopener">${icon('book')}<span>Docs</span></a>
      <a class="top-link" href="${REPO}" target="_blank" rel="noopener">${icon('github')}<span>GitHub</span></a>
    </nav>
  </header>
  <main class="workspace">
  <nav class="gallery" aria-label="Choose a preset">${[...new Set(presets.map((p) => p.group))]
    .map(
      (group) => `
    <section class="preset-group"><h2 class="eyebrow">${group}</h2><div class="preset-grid">${presets
      .filter((preset) => preset.group === group)
      .map(
        (preset) => `
      <button class="preset-card" data-preset="${preset.id}" aria-label="${preset.number}. ${preset.name}" aria-pressed="false" title="${preset.category}" style="--card-accent:${preset.accent}">
        <span class="preset-preview"><img src="${import.meta.env.BASE_URL}previews/${preset.id}.png" alt="" width="862" height="690" loading="lazy"></span>
        <span class="preset-title">${preset.name}</span><span class="preset-number">${preset.number}</span>
      </button>`,
      )
      .join('')}</div></section>`,
    )
    .join('')}</nav>
    <section class="viewport" aria-label="Simulation viewport">
      <div id="canvas-host"></div>
      <aside class="diagnostics" aria-label="Live performance diagnostics">
        <div class="diagnostic-heading"><span class="live-dot"></span><span id="sim-state">INITIALIZING</span></div>
        <dl><div><dt>FPS</dt><dd id="fps">—</dd></div><div><dt>Frame</dt><dd><span id="frame-ms">—</span><small>ms</small></dd></div><div><dt>Particles</dt><dd id="particle-count">—</dd></div></dl>
        <div class="diagnostic-foot"><span id="solver-info">GPU COMPUTE</span><span id="sim-time">0.0 s</span></div>
      </aside>
      <label class="particle-level"><span>Particles</span><select id="particle-level" aria-label="Particle count">${PARTICLE_LEVELS.map(
        (level) =>
          `<option value="${level.id}"${level.id === 'medium' ? ' selected' : ''}>${level.label} · ${level.count.toLocaleString('en-US')}</option>`,
      ).join('')}</select></label>
      <div class="scene-label"><span id="scene-category"></span><h1 id="scene-name"></h1></div>
      <div id="loading" class="loading" role="status"><span class="spinner"></span><strong id="loading-title">Preparing the simulation</strong><span id="loading-copy">Compiling the first frame…</span></div>
      <div id="error" class="error-card" hidden role="alert"><span class="eyebrow">GRAPHICS UNAVAILABLE</span><h2>WebGPU unavailable</h2><p id="error-copy"></p><p>Use a browser with WebGPU and hardware acceleration enabled. Open this demo on localhost or HTTPS.</p><button id="retry" class="primary">Try again ${icon('reset')}</button></div>
      <div class="viewport-bottom">
        <div class="camera-hint" id="interaction-hint">Drag to orbit · Scroll to zoom</div>
        <div class="transport" aria-label="Playback controls">
          <button id="play" class="play-button" title="Pause (Space)" aria-label="Pause simulation">${icon('pause')}</button>
          <button id="restart" class="icon-button" title="Restart (R)" aria-label="Restart simulation">${icon('reset')}</button>
          <span class="toolbar-divider"></span>
          <button id="camera-reset" class="icon-button" title="Reset camera" aria-label="Reset camera">${icon('orbit')}</button>
          <button id="capture" class="icon-button" title="Save image" aria-label="Save image">${icon('camera')}</button>
        </div>
        <button id="mobile-controls" class="mobile-controls" aria-controls="inspector" aria-expanded="false">${icon('tune')} Parameters</button>
      </div>
    </section>
    <aside class="inspector" id="inspector" aria-label="Preset parameters">
      <div class="inspector-top"><span class="eyebrow">PRESET</span><button id="close-controls" class="icon-button" aria-label="Close parameters">${icon('close')}</button><span id="preset-index"></span></div>
      <h2 id="panel-name"></h2><p id="description" class="description"></p>
      <div class="section-label"><span>PARAMETERS</span><button id="defaults" class="text-button">Reset all</button></div>
      <div id="parameters"></div>
      <p class="parameter-note" id="parameter-note">Changes apply in real time.</p>
      <div class="render-settings">
        <div class="section-label"><span>APPEARANCE</span></div>
        <fieldset class="segmented"><legend class="sr-only">Render mode</legend><label><input type="radio" name="render-mode" value="surface" checked><span>Surface</span></label><label><input type="radio" name="render-mode" value="particles"><span>Particles</span></label></fieldset>
        <label class="setting-row switch-row" for="reflections"><span>Reflections</span><input type="checkbox" id="reflections" checked><span class="switch" aria-hidden="true"></span></label>
        <label class="setting-row switch-row" for="ambient-occlusion"><span>Ambient occlusion</span><input type="checkbox" id="ambient-occlusion" checked><span class="switch" aria-hidden="true"></span></label>
        <label class="setting-row switch-row" for="loop"><span>Loop experiment</span><input type="checkbox" id="loop"><span class="switch" aria-hidden="true"></span></label>
        <div class="section-label"><span>ADVANCED</span></div>
        <div class="setting-row"><label for="substeps" title="Solver substeps per frame. Auto scales with the particle count so finer particles stay stable.">Substeps</label><select id="substeps"><option value="auto">Auto</option>${Array.from(
          { length: 16 },
          (_, i) => `<option value="${i + 1}">${i + 1}</option>`,
        ).join('')}</select></div>
      </div>
    </aside>
  </main>

  <div id="toast" role="status" class="toast" hidden></div>
`;

const el = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const host = el('canvas-host');
const memory = new Map<string, Values>();
let selected: Preset =
  presets.find((p) => p.id === new URL(location.href).searchParams.get('preset')) ?? presets[0]!;
let values = defaults(selected);
let world: World | undefined;
let generation = 0;
let queue = Promise.resolve();
let playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
let looping = false;
let particleLevel: ParticleLevel = 'medium';
let particleView = false;
/** Fixed solver substeps, or null to use the preset's count-scaled default. */
let substepOverride: number | null = null;
let ambientOcclusion = true;
let reflections = true;
let loading = true;
let toastTimer = 0;
let structuralTimer = 0;

function toast(message: string): void {
  el('toast').textContent = message;
  el('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    el('toast').hidden = true;
  }, 3000);
}

function syncPlayback(): void {
  el('play').innerHTML = icon(playing ? 'pause' : 'play');
  el('play').setAttribute('aria-label', playing ? 'Pause simulation' : 'Play simulation');
  el('play').title = `${playing ? 'Pause' : 'Play'} (Space)`;
  if (world) world.playing = playing;
}

function stats(data: Diagnostics): void {
  el('fps').textContent = Math.round(data.fps).toString();
  el('frame-ms').textContent = data.frameMs.toFixed(1);
  el('particle-count').textContent = data.particles.toLocaleString();
  el('sim-time').textContent = `${data.time.toFixed(1)} s`;
  el('solver-info').textContent = `${data.substeps} SUBSTEPS · ${data.iterations} ITERATIONS`;
  el('sim-state').textContent = playing
    ? data.limited
      ? 'RUNNING · GPU LIMITED'
      : 'RUNNING'
    : 'PAUSED';
  document.body.classList.toggle('paused', !playing);
}

function showError(error: unknown): void {
  loading = false;
  el('loading').hidden = true;
  el('error').hidden = false;
  el('sim-state').textContent = 'UNAVAILABLE';
  el('error-copy').textContent =
    error instanceof Error ? error.message : 'The simulation could not start.';
  console.error(error);
}

function setBusy(busy: boolean): void {
  loading = busy;
  el('loading').hidden = !busy;
  el('parameters').setAttribute('aria-busy', String(busy));
  for (const id of ['play', 'restart', 'capture', 'camera-reset'])
    el<HTMLButtonElement>(id).disabled = busy;
}

function rebuild(preserveCamera = true): void {
  const token = ++generation;
  const preset = selected;
  const config = { ...values };
  const count = particleCount(preset, particleLevel);
  const camera = preserveCamera ? world?.cameraState() : undefined;
  setBusy(true);
  el('error').hidden = true;
  el('loading-title').textContent = `Preparing ${preset.name.toLowerCase()}`;
  el('loading-copy').textContent = 'Building particles and compiling shaders…';
  el('sim-state').textContent = 'INITIALIZING';
  queue = queue
    .then(async () => {
      if (token !== generation) return;
      const previous = world;
      world = undefined;
      await previous?.dispose();
      const next = new World(
        preset,
        (data) => {
          if (token === generation) stats(data);
        },
        () => rebuild(),
        (error) => {
          if (token === generation) showError(error);
        },
        () => {
          if (token === generation) {
            playing = true;
            syncPlayback();
          }
        },
      );
      try {
        await next.init(host, config, count, camera);
        if (token !== generation) {
          await next.dispose();
          return;
        }
        world = next;
        // A control can change while shaders compile. Reconcile live values before displaying.
        for (const control of preset.controls)
          if (!control.restart) next.experiment.setParameter(control.key, values[control.key]!);
        next.playing = playing;
        next.looping = looping;
        next.ambientOcclusion = ambientOcclusion;
        el('substeps').querySelector('option[value="auto"]')!.textContent =
          `Auto (${next.experiment.substeps})`;
        if (substepOverride !== null) next.experiment.loop.substeps = substepOverride;
        next.experiment.setParticleView?.(particleView);
        next.experiment.setReflections?.(reflections);
        host.replaceChildren(next.canvas);
        el('interaction-hint').textContent = next.experiment.interact
          ? 'Click the liquid to splash · Drag to orbit · Scroll to zoom'
          : 'Drag to orbit · Scroll to zoom';
        setBusy(false);
        syncPlayback();
        next.start();
      } catch (error) {
        await next.dispose();
        if (token === generation) showError(error);
      }
    })
    .catch(showError);
}

function format(value: number, step: number): string {
  const decimals = step.toString().split('.')[1]?.length ?? 0;
  return value.toFixed(decimals);
}

function renderParameters(): void {
  el('parameters').innerHTML = selected.controls
    .map(
      (control) => `
    <div class="parameter"><div class="parameter-label"><label for="control-${control.key}">${control.label}${control.restart ? '<span class="restart-mark" title="Restarts the experiment">↻</span>' : ''}</label><output id="value-${control.key}" for="control-${control.key}">${format(values[control.key]!, control.step)}${control.unit ? `<small>${control.unit}</small>` : ''}</output></div>
      <input type="range" id="control-${control.key}" min="${control.min}" max="${control.max}" step="${control.step}" value="${values[control.key]}" aria-describedby="help-${control.key}" style="--fill:${((values[control.key]! - control.min) / (control.max - control.min)) * 100}%">
      <p class="parameter-help" id="help-${control.key}">${control.description}${control.restart ? ' Restarts the experiment.' : ''}</p></div>`,
    )
    .join('');
  el('parameter-note').textContent = selected.controls.some((control) => control.restart)
    ? 'Live controls · ↻ restarts the experiment'
    : 'Changes apply in real time.';
  for (const control of selected.controls) {
    const input = el<HTMLInputElement>(`control-${control.key}`);
    input.addEventListener('input', () => {
      const value = input.valueAsNumber;
      values[control.key] = value;
      memory.set(selected.id, { ...values });
      el(`value-${control.key}`).innerHTML =
        `${format(value, control.step)}${control.unit ? `<small>${control.unit}</small>` : ''}`;
      input.style.setProperty(
        '--fill',
        `${((value - control.min) / (control.max - control.min)) * 100}%`,
      );
      if (!control.restart) world?.experiment.setParameter(control.key, value);
    });
    if (control.restart)
      input.addEventListener('change', () => {
        clearTimeout(structuralTimer);
        setBusy(true);
        structuralTimer = window.setTimeout(() => rebuild(), 180);
      });
  }
}

function selectPreset(preset: Preset, updateUrl = true): void {
  clearTimeout(structuralTimer);
  memory.set(selected.id, { ...values });
  selected = preset;
  values = { ...(memory.get(preset.id) ?? defaults(preset)) };
  document.documentElement.style.setProperty('--accent', preset.accent);
  for (const id of ['scene-name', 'panel-name']) el(id).textContent = preset.name;
  el('scene-category').textContent = `${preset.number} / ${preset.category}`;
  el('description').textContent = preset.description;
  el('preset-index').textContent = `${preset.number} / ${String(presets.length).padStart(2, '0')}`;
  document.title = `${preset.name} — Three.js Particle Fluids`;
  // Each preset lists its own counts for the shared quality levels.
  for (const option of el<HTMLSelectElement>('particle-level').options) {
    const level = PARTICLE_LEVELS.find((entry) => entry.id === option.value)!;
    option.textContent = `${level.label} · ${particleCount(preset, level.id).toLocaleString('en-US')}`;
  }
  document
    .querySelectorAll<HTMLButtonElement>('[data-preset]')
    .forEach((button) =>
      button.setAttribute('aria-pressed', String(button.dataset['preset'] === preset.id)),
    );
  if (updateUrl) {
    const url = new URL(location.href);
    url.searchParams.set('preset', preset.id);
    history.replaceState(null, '', url);
  }
  renderParameters();
  rebuild(false);
}

document
  .querySelectorAll<HTMLButtonElement>('[data-preset]')
  .forEach((button) =>
    button.addEventListener('click', () =>
      selectPreset(presets.find((p) => p.id === button.dataset['preset'])!),
    ),
  );
el('play').addEventListener('click', () => {
  playing = !playing;
  syncPlayback();
});
el('restart').addEventListener('click', () => rebuild());
el('retry').addEventListener('click', () => rebuild(false));
el('camera-reset').addEventListener('click', () => world?.resetCamera());
el('defaults').addEventListener('click', () => {
  values = defaults(selected);
  memory.set(selected.id, { ...values });
  renderParameters();
  rebuild();
});
el<HTMLSelectElement>('particle-level').addEventListener('change', (event) => {
  particleLevel = (event.target as HTMLSelectElement).value as ParticleLevel;
  rebuild();
});
el<HTMLSelectElement>('substeps').addEventListener('change', (event) => {
  const value = (event.target as HTMLSelectElement).value;
  substepOverride = value === 'auto' ? null : Number(value);
  // Applies live: the solver only re-chains its kernels.
  if (world) world.experiment.loop.substeps = substepOverride ?? world.experiment.substeps;
});
el<HTMLInputElement>('loop').addEventListener('change', (event) => {
  looping = (event.target as HTMLInputElement).checked;
  if (world) world.looping = looping;
});
el<HTMLInputElement>('reflections').addEventListener('change', (event) => {
  reflections = (event.target as HTMLInputElement).checked;
  world?.experiment.setReflections?.(reflections);
});
el<HTMLInputElement>('ambient-occlusion').addEventListener('change', (event) => {
  ambientOcclusion = (event.target as HTMLInputElement).checked;
  if (world) world.ambientOcclusion = ambientOcclusion;
});
document.querySelectorAll<HTMLInputElement>('[name="render-mode"]').forEach((input) =>
  input.addEventListener('change', () => {
    particleView = input.value === 'particles';
    world?.experiment.setParticleView?.(particleView);
  }),
);

function togglePanel(open: boolean): void {
  document.body.classList.toggle('controls-open', open);
  el('mobile-controls').setAttribute('aria-expanded', String(open));
  if (open) el('close-controls').focus();
  else el('mobile-controls').focus();
}
el('mobile-controls').addEventListener('click', () =>
  togglePanel(!document.body.classList.contains('controls-open')),
);
el('close-controls').addEventListener('click', () => togglePanel(false));
el('capture').addEventListener('click', async () => {
  if (!world) return;
  const button = el<HTMLButtonElement>('capture');
  button.disabled = true;
  try {
    const blob = await world.screenshot();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${selected.id}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast('Image saved');
  } catch {
    toast('Image could not be saved. Please try again.');
  } finally {
    button.disabled = false;
  }
});
window.addEventListener('keydown', (event) => {
  const target = event.target as HTMLElement;
  if (event.key === 'Escape' && document.body.classList.contains('controls-open'))
    togglePanel(false);
  if (
    target.closest('input, select, textarea, button, a') ||
    event.metaKey ||
    event.ctrlKey ||
    event.altKey ||
    loading
  )
    return;
  if (event.code === 'Space') {
    event.preventDefault();
    playing = !playing;
    syncPlayback();
  }
  if (event.key.toLowerCase() === 'r') rebuild();
});
window.addEventListener('popstate', () =>
  selectPreset(
    presets.find((p) => p.id === new URL(location.href).searchParams.get('preset')) ?? presets[0]!,
    false,
  ),
);
window.addEventListener('pagehide', () => {
  ++generation;
  void world?.dispose();
});
syncPlayback();
selectPreset(selected);
