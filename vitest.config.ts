import { defineConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';
const suite = process.env['VITEST_SUITE'];
const browser = suite === 'gpu' || suite === 'perf';
export default defineConfig({
  optimizeDeps: {
    include: browser
      ? [
          'three/src/constants.js',
          'three/examples/jsm/controls/OrbitControls.js',
          'three/examples/jsm/environments/RoomEnvironment.js',
          'three/examples/jsm/geometries/RoundedBoxGeometry.js',
        ]
      : [],
  },
  test: {
    include:
      suite === 'perf'
        ? ['tests/perf/**/*.gpu.perf.ts']
        : browser
          ? ['tests/**/*.gpu.test.ts']
          : ['tests/**/!(*.gpu).test.ts'],
    fileParallelism: !browser,
    ...(browser
      ? {
          browser: {
            enabled: true,
            provider: playwright({
              launchOptions: { channel: 'chrome', args: ['--enable-unsafe-webgpu'] },
            }),
            instances: [{ browser: 'chromium' }],
            headless: true,
            screenshotFailures: false,
          },
        }
      : { environment: 'node' as const }),
  },
});
