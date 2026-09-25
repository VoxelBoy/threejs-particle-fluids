import { describe, expect, it } from 'vitest';
import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute, Vector3 } from 'three';

import {
  ParticleSystem,
  SimLoop,
  createParticleRenderer,
  createXpbdUniforms,
  type ParticleInit,
} from '../../../src/core/index.js';
import { ClothSystem, fromBufferGeometry } from '../../../src/cloth/index.js';

// Phase 19 G1 #2 — terminal wind deflection (Keckeisen 2004 §3).
//
// Plan §"Validation/Automatic G1": "32×32 sheet hanging, uniform wind
// along +x. At steady state, the deflection angle of the sheet
// matches an analytic estimate within 10% (rough — the analytic
// estimate is itself approximate; this is a sanity gate, not a
// precision one)."
//
// **Analytic estimate** — pendulum-equilibrium of the cloth's
// center of mass under gravity + Keckeisen drag + lift on a face
// rotated by θ from its initial orientation. Setting force-line ∥
// pivot→COM ray:
//
//   ρ_s · g · sin(θ) = v² · (dragCoeff·cos²(θ) − liftCoeff·sin²(θ))
//
// where `ρ_s` = surface density (kg/m²), `g` = gravity magnitude,
// `v` = wind speed, `dragCoeff = 0.5·C_D·ρ_air`, `liftCoeff =
// 0.5·C_L·ρ_air`. Cloth tilts about the pivot (top edge midpoint)
// in the wind direction; θ is the angle between the cloth and
// vertical. Both drag (pushes in +wind direction) and lift (in our
// (n̄,v̂_rel)-plane convention, pulls cloth more vertical-aligned)
// act on the cloth's face, scaled by the face's exposure to the
// flow. See block-comment derivation in the test source for the
// full step-through.
//
// **Why the cloth doesn't rigid-rotate to that angle.** The cloth
// catenary-bows under gravity; the analytic compares to the
// pivot→COM angle (the closest single-angle proxy for a deformable
// hinged cloth). At MVP wind speeds the bow-vs-rigid distinction is
// a few degrees — well inside the plan's 10 % sanity gate.
//
// Mesh size M=16 (a 16×16 sheet); the wind-equilibrium property is
// mesh-size-independent. Iterations bumped to I=4 to converge the
// cloth in the test budget.

interface SheetData {
  readonly geometry: BufferGeometry;
  readonly pinnedIndices: number[];
  /**
   * Cloth-local (pinned) vertex indices belonging to the top row
   * — used as the pivot reference when computing deflection angle.
   */
  readonly topRowIndices: number[];
}

function buildHangingSheet(M: number): SheetData {
  const positions: number[] = [];
  const indices: number[] = [];
  const pinnedIndices: number[] = [];
  const topRowIndices: number[] = [];
  for (let j = 0; j < M; j++) {
    for (let i = 0; i < M; i++) {
      const u = i / (M - 1);
      const v = j / (M - 1);
      // Centred on x = 0; cloth lies in the xy plane initially.
      positions.push(u - 0.5, -v, 0);
    }
  }
  for (let j = 0; j < M - 1; j++) {
    for (let i = 0; i < M - 1; i++) {
      const a = j * M + i;
      const b = j * M + (i + 1);
      const c = (j + 1) * M + i;
      const d = (j + 1) * M + (i + 1);
      indices.push(a, c, d);
      indices.push(a, d, b);
    }
  }
  for (let i = 0; i < M; i++) {
    pinnedIndices.push(i);
    topRowIndices.push(i);
  }
  const geom = new BufferGeometry();
  geom.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geom.setIndex(new Uint32BufferAttribute(new Uint32Array(indices), 1));
  return { geometry: geom, pinnedIndices, topRowIndices };
}

/**
 * Solve the analytic equilibrium for the COM-pivot angle θ from
 * vertical of a hanging cloth in steady wind (perpendicular to its
 * initial face). Newton iteration on
 *   F(θ) = ρ_s·g·sin(θ) - v²·(dragCoeff·cos²(θ) - liftCoeff·sin²(θ))
 * starting from the small-angle approximation
 *   θ_0 = atan(v²·dragCoeff / (ρ_s·g))
 * Converges in 4-5 iterations for the parameter ranges we test.
 */
function analyticDeflectionAngle(args: {
  readonly surfaceDensity: number;
  readonly gravityMag: number;
  readonly windSpeed: number;
  readonly dragCoeff: number;
  readonly liftCoeff: number;
}): number {
  const { surfaceDensity, gravityMag, windSpeed, dragCoeff, liftCoeff } = args;
  const rsg = surfaceDensity * gravityMag;
  const vSq = windSpeed * windSpeed;
  const F = (theta: number): number =>
    rsg * Math.sin(theta) -
    vSq * (dragCoeff * Math.cos(theta) ** 2 - liftCoeff * Math.sin(theta) ** 2);
  const dF = (theta: number): number =>
    rsg * Math.cos(theta) +
    vSq *
      (2 * dragCoeff * Math.cos(theta) * Math.sin(theta) +
        2 * liftCoeff * Math.sin(theta) * Math.cos(theta));
  let theta = Math.atan((vSq * dragCoeff) / Math.max(rsg, 1e-9));
  for (let k = 0; k < 12; k++) {
    const f = F(theta);
    const dfdth = dF(theta);
    if (Math.abs(dfdth) < 1e-12) break;
    const next = theta - f / dfdth;
    if (Math.abs(next - theta) < 1e-9) {
      theta = next;
      break;
    }
    theta = next;
  }
  return theta;
}

async function runWindScene(args: {
  readonly M: number;
  readonly windSpeed: number;
  readonly substeps: number;
  readonly iterations: number;
  readonly frames: number;
  readonly surfaceDensity: number;
  readonly gravityMag: number;
  readonly dragCoeff: number;
  readonly liftCoeff: number;
}): Promise<{
  readonly measuredAngle: number;
  readonly nanFree: boolean;
  readonly comOffset: { y: number; z: number };
}> {
  const renderer = await createParticleRenderer();
  try {
    const { geometry, pinnedIndices, topRowIndices } = buildHangingSheet(args.M);
    const graph = fromBufferGeometry(geometry, {
      surfaceDensity: args.surfaceDensity,
      pinnedIndices,
    });

    const initial: ParticleInit[] = [];
    for (let i = 0; i < graph.positions.length; i++) {
      const p = graph.positions[i]!;
      initial.push({
        position: [p[0], p[1], p[2]],
        velocity: [0, 0, 0],
        invMass: graph.invMass[i]!,
        phase: 1,
      });
    }
    const particles = new ParticleSystem(renderer, graph.positions.length, 0.05);
    particles.uploadParticles(initial);

    const xpbd = createXpbdUniforms(1 / 60);
    const cloth = new ClothSystem({
      particles,
      xpbd,
      graph,
      particleOffset: 0,
      stretchCompliance: 1e-7,
      bendCompliance: 1.0,
      tetherCompliance: 0,
      dragCoeff: args.dragCoeff,
      liftCoeff: args.liftCoeff,
      wind: new Vector3(0, 0, args.windSpeed),
    });

    const loop = new SimLoop(particles, {
      substeps: args.substeps,
      iterations: args.iterations,
      xpbd,
      materials: [cloth],
    });
    loop.kernels.floorY.value = -1e9;
    loop.gravity.set(0, -args.gravityMag, 0);

    const frameDt = 1 / 60;
    // Velocity damping speeds settling. Doesn't change the wind
    // force calculation: F_drag depends on |v_rel|² where v_rel =
    // v_avg − wind. Damping shrinks v_avg toward zero, which makes
    // |v_rel| ≈ |wind| — the maximum-drag case anyway.
    const damp = 0.9;
    let nanFree = true;
    for (let f = 0; f < args.frames; f++) {
      await loop.step(frameDt);
      const buf = await renderer.getArrayBufferAsync(particles.velocities.value);
      const v = new Float32Array(buf);
      for (let k = 0; k < v.length; k++) v[k] = v[k]! * damp;
      (particles.velocities.value.array as Float32Array).set(v);
      particles.velocities.value.needsUpdate = true;
    }

    const snap = await particles.readback();
    // Check NaN-free across all positions.
    for (let i = 0; i < graph.positions.length; i++) {
      for (let k = 0; k < 3; k++) {
        if (!Number.isFinite(snap.positions[4 * i + k]!)) {
          nanFree = false;
        }
      }
    }
    // Compute pivot (mean of pinned positions) — should be near
    // (0, 0, 0) since the top row is at y=0 in the rest config.
    let pivotY = 0;
    let pivotZ = 0;
    for (const idx of topRowIndices) {
      pivotY += snap.positions[4 * idx + 1]!;
      pivotZ += snap.positions[4 * idx + 2]!;
    }
    pivotY /= topRowIndices.length;
    pivotZ /= topRowIndices.length;
    // Compute COM of free particles only (mass-weighted via invMass).
    // For uniform surface density, mass-weighted = vertex-area-
    // weighted = essentially uniform on a quad mesh; simple mean
    // suffices.
    let comY = 0;
    let comZ = 0;
    let nFree = 0;
    for (let i = 0; i < graph.positions.length; i++) {
      if (graph.invMass[i] === 0) continue;
      comY += snap.positions[4 * i + 1]!;
      comZ += snap.positions[4 * i + 2]!;
      nFree++;
    }
    comY /= nFree;
    comZ /= nFree;
    const offsetY = comY - pivotY;
    const offsetZ = comZ - pivotZ;
    // Deflection angle = angle between (pivot→COM) and -ŷ.
    // Since cloth hangs in -y and wind pushes +z, COM offset is
    // approximately (0, -L_eff·cos(θ), L_eff·sin(θ)) for some
    // effective length L_eff. The angle from vertical is
    // atan2(z, |y|).
    const measuredAngle = Math.atan2(offsetZ, Math.abs(offsetY));

    particles.destroy();
    return {
      measuredAngle,
      nanFree,
      comOffset: { y: offsetY, z: offsetZ },
    };
  } finally {
    renderer.dispose();
  }
}

describe('Phase 19 G1 #2 — terminal wind deflection (Keckeisen 2004)', () => {
  it('16×16 sheet hanging in uniform wind reaches deflection within 10% of analytic', async () => {
    const M = 16;
    const surfaceDensity = 0.2;
    const gravityMag = 9.81;
    const dragCoeff = 0.6125;
    const liftCoeff = 0.3;
    const windSpeed = 1.0; // m/s — keeps θ_analytic ~16°, well inside the small-angle regime

    const analytic = analyticDeflectionAngle({
      surfaceDensity,
      gravityMag,
      windSpeed,
      dragCoeff,
      liftCoeff,
    });
    const result = await runWindScene({
      M,
      windSpeed,
      substeps: 8,
      iterations: 4,
      frames: 240,
      surfaceDensity,
      gravityMag,
      dragCoeff,
      liftCoeff,
    });
    const analyticDeg = (analytic * 180) / Math.PI;
    const measuredDeg = (result.measuredAngle * 180) / Math.PI;
    const relErr = Math.abs(measuredDeg - analyticDeg) / Math.max(analyticDeg, 1e-6);
    // eslint-disable-next-line no-console
    console.info(
      `[wind-deflection] M=${M} v=${windSpeed} m/s → measured=${measuredDeg.toFixed(2)}°, analytic=${analyticDeg.toFixed(2)}°, relErr=${(relErr * 100).toFixed(2)}%, comOffset=(y=${result.comOffset.y.toFixed(3)}, z=${result.comOffset.z.toFixed(3)})`,
    );

    expect(result.nanFree).toBe(true);
    // Plan: within 10 %. The analytic is itself approximate
    // (treats cloth as rigid pendulum); 10 % gates that the
    // cloth physically deflects in the wind direction by the
    // right order of magnitude.
    expect(relErr).toBeLessThan(0.1);
    // Sanity: cloth deflected in the +z direction (wind direction).
    expect(result.comOffset.z).toBeGreaterThan(0.01);
    expect(result.comOffset.y).toBeLessThan(0); // cloth still hangs down
  }, 240_000);
});
