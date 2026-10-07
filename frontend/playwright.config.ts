import { defineConfig } from '@playwright/test';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

const baseURL = process.env['SENTILENSE_TEST_URL'] ?? 'http://127.0.0.1:8000';
const executablePath = process.env['CHROMIUM_PATH'] ?? (existsSync('/usr/bin/chromium') ? '/usr/bin/chromium' : undefined);

export default defineConfig({
  testDir: './e2e',
  timeout: 60000,
  expect: { timeout: 15000 },
  workers: 1,
  reporter: [['list'], ['json', { outputFile: '../artifacts/frontend-e2e.json' }]],
  use: {
    baseURL,
    browserName: 'chromium',
    launchOptions: { executablePath, args: ['--no-sandbox', '--disable-dev-shm-usage'] },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1360, height: 1000 } } },
    { name: 'mobile', use: { viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true } },
  ],
  webServer: process.env['SENTILENSE_TEST_URL'] ? undefined : {
    command: 'uv run --frozen --extra api python scripts/serve.py',
    cwd: resolve('..'),
    url: `${baseURL}/api/health`,
    reuseExistingServer: true,
    timeout: 30000,
  },
});
