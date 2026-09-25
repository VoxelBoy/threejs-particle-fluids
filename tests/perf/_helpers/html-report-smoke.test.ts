// Phase Perf — HTML report generator smoke test.
//
// Loads the static `report-template.html`, runs `generateHtmlReport`
// against a fixture `PerfReportJson`, and asserts the produced HTML
// contains every expected structural element.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { generateHtmlReport } from './report-generator.js';
import type { PerfReportJson } from './types.js';

const here = dirname(fileURLToPath(import.meta.url));
const TEMPLATE = readFileSync(join(here, 'report-template.html'), 'utf8');

function fixtureReport(): PerfReportJson {
  return {
    version: 1,
    commit: 'abc123def4567890',
    date: '2026-04-25T12:34:56Z',
    platform: { gpu: 'Apple M1 Pro', browser: 'Chrome 147', os: 'Darwin 24.6.0' },
    timing_method: 'per-kernel-pass',
    scenes: [
      {
        id: 'fluid-10k',
        particle_count: 10000,
        substeps: 4,
        iterations: 2,
        frames_warmup: 20,
        frames_measure: 50,
        frame_total_ms: { p10: 4.1, p50: 4.2, p90: 4.5 },
        frame_step_ms: { p10: 1.5, p50: 1.6, p90: 1.7 },
        dispatch_count: 80,
        kernels: [
          {
            name: 'hashGrid.cellIndexAndHistogram',
            dispatches_per_frame: 4,
            min_ms: 0.03,
            p10_ms: 0.04,
            p50_ms: 0.05,
            p90_ms: 0.06,
            max_ms: 0.07,
            samples: 800,
          },
          {
            name: 'fluid.lambda',
            dispatches_per_frame: 8,
            min_ms: 0.28,
            p10_ms: 0.3,
            p50_ms: 0.32,
            p90_ms: 0.35,
            max_ms: 0.41,
            samples: 1600,
          },
        ],
      },
      {
        id: 'fluid-contact-100k',
        particle_count: 100000,
        substeps: 4,
        iterations: 2,
        frames_warmup: 20,
        frames_measure: 50,
        frame_total_ms: { p10: 18.5, p50: 19.0, p90: 19.6 },
        frame_step_ms: { p10: 7.8, p50: 8.0, p90: 8.4 },
        dispatch_count: 80,
        kernels: [
          {
            name: 'fluid.positionDelta',
            dispatches_per_frame: 8,
            min_ms: 0.4,
            p10_ms: 0.42,
            p50_ms: 0.45,
            p90_ms: 0.48,
            max_ms: 0.55,
            samples: 1600,
          },
        ],
      },
    ],
  };
}

describe('Phase Perf — HTML report generator', () => {
  it('substitutes inline data and produces a self-contained report', () => {
    const data = fixtureReport();
    const html = generateHtmlReport(TEMPLATE, data);

    expect(html).not.toContain('__PARTICLE_FLUIDS_PERF_DATA__');
    expect(html).toContain('"fluid-10k"');
    expect(html).toContain('"fluid-contact-100k"');
    expect(html).toContain('"hashGrid.cellIndexAndHistogram"');
    expect(html).toContain('"fluid.lambda"');
    expect(html).toContain('"fluid.positionDelta"');

    expect(html).not.toMatch(/__PARTICLE_FLUIDS_PERF_DATA__/);
    expect(html).not.toContain('"</script>"');

    expect(html).toContain('id="baseline-input"');
    expect(html).toContain('type="file"');
    expect(html).toContain('Copy table as markdown');
    expect(html).toContain('Compare to baseline');
    expect(html).toContain('Frame contrib');

    expect(html).toContain('"frame_step_ms"');
    expect(html).toContain('Frame step');

    expect(html).toMatch(/^<!doctype html>/i);
    expect(html).toContain('<html');
    expect(html).toContain('</html>');

    expect(html).not.toMatch(/<link\s+[^>]*href=/);
    expect(html).not.toMatch(/<script\s+[^>]*src=/);
  });

  it('rejects templates without exactly one placeholder', () => {
    const data = fixtureReport();

    expect(() => generateHtmlReport('<html><body>no placeholder here</body></html>', data)).toThrow(
      /expected exactly one/,
    );

    expect(() =>
      generateHtmlReport(
        '<html>__PARTICLE_FLUIDS_PERF_DATA__ and __PARTICLE_FLUIDS_PERF_DATA__</html>',
        data,
      ),
    ).toThrow(/expected exactly one/);
  });
});
