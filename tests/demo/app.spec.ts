import { expect, test, type Page } from '@playwright/test';
import { stat } from 'node:fs/promises';

const presetIds = [
  'tidal-chamber',
  'crown-impact',
  'liquid-marble',
  'amber-cascade',
  'floating-forms',
  'elastic-studies',
  'silk-in-motion',
  'vortex-plume',
];

async function ready(page: Page): Promise<void> {
  await expect(page.locator('#loading')).toBeHidden({ timeout: 90_000 });
  await expect(page.locator('#error')).toBeHidden();
  await expect(page.locator('#canvas-host canvas')).toHaveCount(1);
  await expect(page.locator('#fps')).not.toHaveText('—');
}

test('all eight presets render, advance, and switch without browser errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  await page.goto('/');
  await expect(page.locator('[data-preset]')).toHaveCount(8);
  for (const id of presetIds) {
    await page.locator(`[data-preset="${id}"]`).click();
    await ready(page);
    await expect(page.locator(`[data-preset="${id}"]`)).toHaveAttribute('aria-pressed', 'true');
    await expect
      .poll(async () => parseFloat(await page.locator('#sim-time').innerText()))
      .toBeGreaterThan(0.6);
    expect(
      parseInt((await page.locator('#particle-count').innerText()).replaceAll(',', '')),
    ).toBeGreaterThan(100);
    await page.getByLabel('Particles', { exact: true }).check();
    await page.getByLabel('Surface', { exact: true }).check();
  }
  expect(errors).toEqual([]);
});

test('pause, live controls, rebuilds, captures, and rapid navigation preserve a usable app', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/?preset=crown-impact');
  await ready(page);
  await page.getByLabel('Loop experiment').uncheck();
  await page.getByRole('button', { name: 'Pause simulation' }).click();
  await expect(page.locator('#sim-state')).toHaveText('PAUSED');
  const pausedTime = await page.locator('#sim-time').innerText();
  await page.waitForTimeout(350);
  await expect(page.locator('#sim-time')).toHaveText(pausedTime);
  await page.getByLabel('Gravity', { exact: true }).fill('4.2');
  await expect(page.locator('#value-gravity')).toContainText('4.2');
  await page.getByLabel('Drop height').fill('1.3');
  await ready(page);
  await expect(page.getByLabel('Gravity', { exact: true })).toHaveValue('4.2');
  await page.locator('#canvas-host canvas').focus();
  await page.keyboard.press('Space');
  await expect(page.locator('#sim-state')).toHaveText(/RUNNING/);
  await page.getByLabel('Quality', { exact: true }).selectOption('high');
  await ready(page);
  await page.setViewportSize({ width: 1200, height: 820 });
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Save image', exact: true }).click();
  const image = await download;
  expect(image.suggestedFilename()).toBe('crown-impact.png');
  // A cleared WebGPU backbuffer produces a tiny empty PNG even when download succeeds.
  expect((await stat((await image.path())!)).size).toBeGreaterThan(10_000);
  await page.getByRole('button', { name: 'Reset all', exact: true }).click();
  await ready(page);
  await expect(page.getByLabel('Gravity', { exact: true })).toHaveValue('7');
  await page.getByLabel('Quality', { exact: true }).selectOption('balanced');
  for (const id of ['silk-in-motion', 'vortex-plume', 'liquid-marble'])
    await page.locator(`[data-preset="${id}"]`).click();
  await ready(page);
  await expect(page.locator('#panel-name')).toHaveText('Liquid marble');
  await page.getByRole('button', { name: 'Disturb', exact: true }).click();
  await page.getByRole('button', { name: 'Restart simulation' }).click();
  await ready(page);
  expect(errors).toEqual([]);
});

test('mobile controls, reduced motion, and deep links work', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/?preset=silk-in-motion');
  await ready(page);
  await expect(page.locator('#sim-state')).toHaveText('PAUSED');
  await expect(page.locator('#scene-name')).toHaveText('Silk in motion');
  await expect(page.locator('#inspector')).toBeHidden();
  await page.getByRole('button', { name: 'Parameters', exact: true }).click();
  await expect(page.locator('#inspector')).toBeVisible();
  await page.getByLabel('Wind speed', { exact: true }).fill('2.1');
  await expect(page.locator('#value-wind')).toContainText('2.1');
  await page.keyboard.press('Escape');
  await expect(page.locator('#inspector')).toBeHidden();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('missing WebGPU produces a helpful recovery state', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { get: () => undefined }));
  await page.goto('/');
  await expect(page.locator('#error')).toBeVisible();
  await expect(page.locator('#error-copy')).toContainText('WebGPU');
  await expect(page.getByRole('button', { name: 'Try again' })).toBeEnabled();
});
