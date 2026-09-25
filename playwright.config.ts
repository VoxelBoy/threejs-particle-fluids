import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/demo',
  fullyParallel: false,
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 30_000 },
  use: {
    baseURL: 'http://127.0.0.1:5192',
    channel: 'chrome',
    launchOptions: { args: ['--enable-unsafe-webgpu'] },
    viewport: { width: 1440, height: 980 },
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'npm run dev -- --port 5192 --strictPort',
    url: 'http://127.0.0.1:5192',
    reuseExistingServer: !process.env['CI'],
  },
});
