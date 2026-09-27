// Fills the static report template with a fixture report and checks that
// the result is a self-contained page carrying the data and its controls.

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
    version: 2,
    commit: 'abc123def4567890',
    date: '2026-04-25T12:34:56Z',
    platform: { gpu: 'apple metal-3', browser: 'Chrome 147', os: 'darwin 24.6.0' },
    timing_method: 'timestamp',
    scenes: [
      {
        id: 'fluid-10k',
        particle_count: 10000,
        substeps: 4,
        iterations: 2,
        frames_warmup: 60,
        frames_measure: 100,
        frame_gpu_ms: { min: 0.5, p10: 0.51, p50: 0.52, p90: 0.8, max: 1.1 },
        frame_step_ms: { min: 1.1, p10: 1.2, p50: 1.3, p90: 1.9, max: 2.4 },
      },
      {
        id: 'softbody-100k',
        particle_count: 100000,
        substeps: 4,
        iterations: 2,
        frames_warmup: 60,
        frames_measure: 100,
        frame_gpu_ms: { min: 6.6, p10: 6.7, p50: 7.1, p90: 7.6, max: 8.2 },
        frame_step_ms: { min: 7.5, p10: 7.7, p50: 7.9, p90: 8.6, max: 9.9 },
        contact_count: { p10: 4990, p50: 5000, p90: 5010 },
      },
    ],
  };
}

describe('HTML report generator', () => {
  it('inlines the report data into a self-contained page', () => {
    const html = generateHtmlReport(TEMPLATE, fixtureReport());

    expect(html).not.toContain('__PARTICLE_FLUIDS_PERF_DATA__');
    expect(html).toContain('"fluid-10k"');
    expect(html).toContain('"softbody-100k"');
    expect(html).toContain('"frame_gpu_ms"');
    expect(html).toContain('"frame_step_ms"');
    expect(html).toContain('"contact_count"');

    // Controls and table headings the page renders.
    expect(html).toContain('id="baseline-input"');
    expect(html).toContain('type="file"');
    expect(html).toContain('Compare to baseline');
    expect(html).toContain('Copy table as markdown');
    expect(html).toContain('GPU ms/frame');
    expect(html).toContain('Wall-clock ms/frame');
    expect(html).toContain('Contacts/substep');

    expect(html).toMatch(/^<!doctype html>/i);
    expect(html).toContain('</html>');
    // Works offline from file://: no external stylesheets or scripts.
    expect(html).not.toMatch(/<link\s+[^>]*href=/);
    expect(html).not.toMatch(/<script\s+[^>]*src=/);
  });

  it('escapes "<" so data cannot close the inline script', () => {
    const report = fixtureReport();
    const html = generateHtmlReport(TEMPLATE, {
      ...report,
      platform: { ...report.platform, browser: '</script><b>x</b>' },
    });
    expect(html).not.toContain('</script><b>');
    expect(html).toContain('\\u003c/script>');
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
