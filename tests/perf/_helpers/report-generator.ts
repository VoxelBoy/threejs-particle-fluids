// Phase Perf — HTML report generator. Inlines a `PerfReportJson` object
// into the static `report-template.html` via a single
// `__PARTICLE_FLUIDS_PERF_DATA__` placeholder substitution. Returns the final
// self-contained HTML — vanilla HTML/CSS/JS, no external dependencies,
// works fully offline from `file://`.
//

import type { PerfReportJson } from './types.js';

const PLACEHOLDER = '__PARTICLE_FLUIDS_PERF_DATA__';

export function generateHtmlReport(template: string, data: PerfReportJson): string {
  const occurrences = (template.match(/__PARTICLE_FLUIDS_PERF_DATA__/g) ?? []).length;
  if (occurrences !== 1) {
    throw new Error(
      `generateHtmlReport: expected exactly one '${PLACEHOLDER}' in template, found ${occurrences}`,
    );
  }
  // Embed data as a JSON literal — safe because JSON is a strict subset of
  // JavaScript expressions in this position. The template is trusted (it
  // is committed alongside the harness); the data is typed as
  // `PerfReportJson` so it cannot smuggle in functions or `<script>`-class
  // payloads. As a defense in depth against accidentally embedding the
  // closing `</script>` sequence inside a string, we escape `<` to `\u003c`
  // — JSON.stringify doesn't escape it by default but it is still
  // syntactically valid JSON.
  const json = JSON.stringify(data).replace(/</g, '\\u003c');
  return template.replace(PLACEHOLDER, json);
}
