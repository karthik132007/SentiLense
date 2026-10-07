import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

// Deterministic UI coverage; fixture scores come from the saved tweet model.
// Backend relevance and inference are tested separately with TestClient.
const topic = JSON.parse(readFileSync(resolve('e2e/topic-fixture.json'), 'utf-8'));

test('topic evidence, uncertainty, filters and export are explicit', async ({ page }, info) => {
  await page.route('**/api/analyze/keyword', route => route.fulfill({ json: topic }));
  await page.goto('/');
  await expect(page.getByText('Model connected', { exact: true })).toBeVisible();
  await page.getByLabel('What topic would you like to explore?').fill('Data centers');
  await page.getByRole('button', { name: 'Search and analyze', exact: true }).click();
  await expect(page.getByTestId('sentiment-result')).toHaveText('Neutral');
  await expect(page.getByRole('heading', { name: 'Topic passage analysis for “Data centers”' })).toBeVisible();
  await expect(page.getByText('These scores describe passage tone;', { exact: false })).toBeVisible();
  await expect(page.getByText('Neutral is a trained tone class;', { exact: false })).toBeVisible();
  await expect(page.getByText('Only passages matching the complete topic phrase are scored.', { exact: false })).toBeVisible();
  const card = page.getByTestId('keyword-source');
  await expect(card).toHaveCount(1);
  await expect(card).toContainText('1 of 1 extracted passages match');
  await card.getByText('View analyzed topic passages', { exact: true }).click();
  await expect(card.locator('details .segment')).toContainText(topic.sources[0].results[0].text);
  await page.getByRole('button', { name: 'Positive', exact: true }).click();
  await expect(card).toHaveCount(0);
  await page.getByRole('button', { name: 'Neutral', exact: true }).click();
  await expect(card).toHaveCount(1);
  await page.getByText("1 sources couldn't be included", { exact: true }).click();
  await expect(page.getByText('Off-topic source:', { exact: false })).toBeVisible();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download analysis JSON' }).click();
  const download = await downloadPromise;
  const exported = JSON.parse(readFileSync((await download.path())!, 'utf-8'));
  expect(exported.result).toEqual(topic);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('heading', { name: 'Topic passage analysis for “Data centers”' }).click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: resolve(`../artifacts/topic-evidence-${info.project.name}.png`), fullPage: true });
});

test('off-topic results leave no summary or export and allow retry', async ({ page }) => {
  await page.route('**/api/analyze/keyword', route => route.fulfill({
    status: 422,
    json: { detail: 'No readable webpages with sufficient content matching this topic were found. No topic score was produced. Try another phrase or analyze a specific URL.' },
  }));
  await page.goto('/');
  await expect(page.getByText('Model connected', { exact: true })).toBeVisible();
  await page.getByLabel('What topic would you like to explore?').fill('Data centers');
  await page.getByRole('button', { name: 'Search and analyze', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('No topic score was produced');
  await expect(page.getByTestId('sentiment-result')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Download analysis JSON' })).toHaveCount(0);
  await expect(page.getByTestId('keyword-source')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Search and analyze', exact: true })).toBeEnabled();
});
