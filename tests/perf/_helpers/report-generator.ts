// Builds the HTML report: inlines a `PerfReportJson` into
// `report-template.html` at its single `__PARTICLE_FLUIDS_PERF_DATA__`
// placeholder. The result is one self-contained file (no external
// dependencies) that works offline from `file://`.

import type { PerfReportJson } from './types.js';

const PLACEHOLDER = '__PARTICLE_FLUIDS_PERF_DATA__';

export function generateHtmlReport(template: string, data: PerfReportJson): string {
  const occurrences = template.split(PLACEHOLDER).length - 1;
  if (occurrences !== 1) {
    throw new Error(
      `generateHtmlReport: expected exactly one '${PLACEHOLDER}' in template, found ${occurrences}`,
    );
  }
  // JSON is a valid JavaScript expression, so it can be inlined as-is. Escape
  // `<` so no string in the data can close the surrounding <script> element.
  const json = JSON.stringify(data).replace(/</g, '\\u003c');
  // A replacer function keeps `$` sequences in the data from being read as
  // replacement patterns.
  return template.replace(PLACEHOLDER, () => json);
}
