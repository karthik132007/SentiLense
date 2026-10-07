import { expect, test } from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const examples = [
  ['I absolutely love this!', 'positive'],
  ['I hate this product.', 'negative'],
  ['The product arrived yesterday.', 'neutral'],
] as const;
let browserErrors: string[] = [];

test.beforeEach(async ({ page }) => {
  browserErrors = [];
  page.on('pageerror', error => browserErrors.push(error.message));
  await page.goto('/');
  await expect(page.getByText('Model connected', { exact: true })).toBeVisible();
});

test.afterEach(async () => {
  expect(browserErrors, 'The application must not emit JavaScript runtime errors').toEqual([]);
});

test('workspace navigation, keyboard tabs, and a fresh analysis', async ({ page }, info) => {
  await page.getByRole('tab', { name: 'Keyword' }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Text', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('tab', { name: 'Text', exact: true })).toBeFocused();
  await page.keyboard.press('End');
  await expect(page.getByRole('tab', { name: 'Webpage' })).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('Home');
  await expect(page.getByRole('tab', { name: 'Keyword' })).toHaveAttribute('aria-selected', 'true');
  await page.getByLabel('What topic would you like to explore?').fill('antigravity');
  await page.getByRole('button', { name: 'New analysis', exact: true }).click();
  await expect(page.getByLabel('What topic would you like to explore?')).toHaveValue('');
  await expect(page.getByLabel('What topic would you like to explore?')).toBeFocused();
  await expect(page.getByRole('button', { name: 'Search and analyze', exact: true })).toBeDisabled();
  await expect(page.getByRole('heading', { name: 'Your results appear here' })).toBeVisible();
  if (info.project.name === 'desktop') {
    await page.getByText('How sentiment works', { exact: false }).click();
    await expect(page.getByText('The SVM + RoBERTa fusion model estimates negative, neutral, and positive tone.', { exact: false })).toBeVisible();
    await page.getByText('How sentiment works', { exact: false }).click();
  }
  const fonts = await page.evaluate(async () => { await document.fonts.ready; return document.fonts.check('14px Inter'); });
  expect(fonts).toBe(true);
  await page.getByRole('heading', { name: 'Analyze sentiment.', exact: true }).click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: resolve(`../artifacts/frontend-workspace-${info.project.name}.png`), fullPage: true });
});

test('real text predictions, uncertainty explanation, and JSON export', async ({ page }, info) => {
  await page.getByRole('tab', { name: 'Text', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Analyze text', exact: true })).toBeDisabled();
  let lastPrediction: unknown;
  for (const [text, label] of examples) {
    await page.getByRole('button', { name: text, exact: true }).click();
    const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/analyze/text') && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Analyze text', exact: true }).click();
    const response = await responsePromise;
    expect(response.status()).toBe(200);
    const data = await response.json();
    expect(data.sentiment).toBe(label);
    expect(data.scores.positive + data.scores.neutral + data.scores.negative).toBeCloseTo(1);
    await expect(page.locator('.score-label').getByText('Neutral', { exact: true })).toBeVisible();
    await expect(page.getByTestId('sentiment-result')).toHaveText(label.charAt(0).toUpperCase() + label.slice(1));
    await expect(page.getByTestId('confidence-result')).toHaveText(`${(data.confidence * 100).toFixed(1)}%`);
    lastPrediction = data;
  }
  await expect(page.getByText('Neutral is a trained class;', { exact: false })).toBeVisible();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download analysis JSON' }).click();
  const download = await downloadPromise;
  const path = await download.path();
  expect(path).toBeTruthy();
  const exported = JSON.parse(readFileSync(path!, 'utf-8'));
  expect(exported.input).toBe(examples[2][0]);
  expect(exported.result).toEqual(lastPrediction);
  const width = await page.evaluate(() => ({ body: document.documentElement.scrollWidth, viewport: innerWidth }));
  expect(width.body).toBeLessThanOrEqual(width.viewport);
  await page.screenshot({ path: resolve(`../artifacts/frontend-${info.project.name}.png`), fullPage: true });
});

test('reported negative sentence uses the saved SVM through the live API', async ({ page }, info) => {
  await page.getByRole('tab', { name: 'Text', exact: true }).click();
  const text = 'This is fucking trash subject i ever saw in my life, only one with loose brain screw will opt it as minor subject';
  await page.getByLabel('What would you like to understand?').fill(text);
  const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/analyze/text') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Analyze text', exact: true }).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const data = await response.json();
  const savedComparison = JSON.parse(readFileSync(resolve('../artifacts/svm_comparison.json'), 'utf-8'));
  expect(data).toEqual(savedComparison.reported_sentence.fusion);
  expect(data.sentiment).toBe('negative');
  expect(data.scores.negative).toBeGreaterThan(data.scores.positive);
  expect(data.scores.negative).toBeGreaterThan(data.scores.neutral);
  await expect(page.getByTestId('sentiment-result')).toHaveText('Negative');
  await expect(page.getByText('SVM + RoBERTa fusion · English text', { exact: true })).toBeVisible();
  await expect(page.getByTestId('confidence-result')).toHaveText(`${(data.confidence * 100).toFixed(1)}%`);
  writeFileSync(resolve(`../artifacts/svm-negative-text-${info.project.name}.json`), JSON.stringify({ text, prediction: data }, null, 2));
  await page.getByRole('heading', { name: 'Analyze sentiment.', exact: true }).click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: resolve(`../artifacts/svm-negative-text-${info.project.name}.png`), fullPage: true });
});

test('live keyword search, scraped sources, filtering, and JSON export', async ({ page }, info) => {
  await expect(page.getByRole('tab', { name: 'Keyword' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: 'Search and analyze', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'antigravity', exact: true }).click();
  const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/analyze/keyword') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Search and analyze', exact: true }).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const data = await response.json();
  expect(data.keyword).toBe('antigravity');
  expect(data.statistics.sources_analyzed).toBeGreaterThan(0);
  expect(data.statistics.sources_analyzed).toBeLessThanOrEqual(5);
  expect(data.sources.length).toBe(data.statistics.sources_analyzed);
  await expect(page.getByRole('heading', { name: 'Topic passage analysis for “antigravity”' })).toBeVisible();
  await expect(page.getByTestId('sentiment-result')).toHaveText(data.overall_sentiment.charAt(0).toUpperCase() + data.overall_sentiment.slice(1));
  await expect(page.getByTestId('confidence-result')).toHaveText(`${(data.confidence * 100).toFixed(1)}%`);
  const cards = page.getByTestId('keyword-source');
  await expect(cards).toHaveCount(data.sources.length);
  for (let index = 0; index < data.sources.length; index++) {
    const source = data.sources[index];
    await expect(cards.nth(index).getByRole('link')).toHaveAttribute('href', source.url);
    await expect(cards.nth(index).getByRole('heading')).toContainText(source.title);
    expect(source.statistics.analyzed_units).toBe(source.results.length);
    expect(source.results.length).toBeGreaterThan(0);
  }
  const positiveMean = data.sources.reduce((sum: number, source: { scores: { positive: number } }) => sum + source.scores.positive, 0) / data.sources.length;
  expect(data.scores.positive).toBeCloseTo(positiveMean, 10);
  await cards.first().getByText('View analyzed topic passages', { exact: true }).click();
  await expect(cards.first().locator('details .segment').first()).toContainText(data.sources[0].results[0].text);
  await cards.first().getByText('View analyzed topic passages', { exact: true }).click();
  const label = data.sources[0].overall_sentiment;
  await page.getByRole('button', { name: label.charAt(0).toUpperCase() + label.slice(1), exact: true }).click();
  await expect(cards).toHaveCount(data.sources.filter((source: { overall_sentiment: string }) => source.overall_sentiment === label).length);
  await page.getByRole('button', { name: 'All', exact: true }).click();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download analysis JSON' }).click();
  const download = await downloadPromise;
  const exported = JSON.parse(readFileSync((await download.path())!, 'utf-8'));
  expect(exported.input).toBe('antigravity');
  expect(exported.result).toEqual(data);
  const width = await page.evaluate(() => ({ body: document.documentElement.scrollWidth, viewport: innerWidth }));
  expect(width.body).toBeLessThanOrEqual(width.viewport);
  await page.screenshot({ path: resolve(`../artifacts/frontend-keyword-${info.project.name}.png`), fullPage: true });
});

test('reported Data centers query returns relevant live sources', async ({ page }, info) => {
  test.setTimeout(180000);
  await page.getByLabel('What topic would you like to explore?').fill('Data centers');
  const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/analyze/keyword') && response.request().method() === 'POST', { timeout: 170000 });
  await page.getByRole('button', { name: 'Search and analyze', exact: true }).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const data = await response.json();
  expect(data.keyword).toBe('Data centers');
  expect(data.statistics.sources_analyzed).toBeGreaterThan(0);
  expect(data.sources.length).toBe(data.statistics.sources_analyzed);
  const cards = page.getByTestId('keyword-source');
  await expect(cards).toHaveCount(data.sources.length);
  await expect(page.getByRole('heading', { name: 'Topic passage analysis for “Data centers”' })).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
  for (const source of data.sources) {
    expect(source.topic_evidence.matched_passages).toBeGreaterThan(0);
    expect(source.results.length).toBeGreaterThan(0);
    for (const unit of source.results) expect(unit.text).toMatch(/data[\s-]+cent(?:er|re)s?\b/i);
  }
  await cards.first().getByText('View analyzed topic passages', { exact: true }).click();
  await expect(cards.first().locator('details .segment').first()).toContainText(data.sources[0].results[0].text);
  writeFileSync(resolve(`../artifacts/live-data-centers-${info.project.name}.json`), JSON.stringify(data, null, 2));
  await page.getByRole('heading', { name: 'Topic passage analysis for “Data centers”' }).click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: resolve(`../artifacts/live-data-centers-${info.project.name}.png`), fullPage: true });
});

test('live webpage extraction, model results, statistics, and filtering', async ({ page }, info) => {
  await page.getByRole('tab', { name: 'Webpage' }).click();
  await page.getByLabel('A public webpage URL').fill('https://example.com');
  const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/analyze/url') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Analyze webpage', exact: true }).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const data = await response.json();
  expect(data.statistics.analyzed_units).toBe(data.results.length);
  expect(data.results.length).toBeGreaterThan(0);
  await expect(page.getByRole('heading', { name: data.title, exact: true })).toBeVisible();
  await expect(page.getByTestId('sentiment-result')).toHaveText(data.overall_sentiment.charAt(0).toUpperCase() + data.overall_sentiment.slice(1));
  for (const unit of data.results) await expect(page.getByText(unit.text, { exact: true })).toBeVisible();
  const emptyLabel = ['positive', 'neutral', 'negative'].find(label => !data.results.some((unit: { sentiment: string }) => unit.sentiment === label));
  if (emptyLabel) {
    await page.getByRole('button', { name: emptyLabel.charAt(0).toUpperCase() + emptyLabel.slice(1), exact: true }).click();
    await expect(page.getByText('No segments with this sentiment.', { exact: false })).toBeVisible();
    await page.getByRole('button', { name: 'All', exact: true }).click();
    await expect(page.getByText(data.results[0].text, { exact: true })).toBeVisible();
  }
  await page.screenshot({ path: resolve(`../artifacts/frontend-webpage-${info.project.name}.png`), fullPage: true });
});

test('invalid input and a real rejected webpage request recover cleanly', async ({ page }) => {
  await page.getByRole('tab', { name: 'Webpage' }).click();
  await page.getByLabel('A public webpage URL').fill('not-a-url');
  await expect(page.getByRole('button', { name: 'Analyze webpage', exact: true })).toBeDisabled();
  await page.getByLabel('A public webpage URL').fill('http://127.0.0.1/private');
  const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/analyze/url') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Analyze webpage', exact: true }).click();
  expect((await responsePromise).status()).toBe(400);
  await expect(page.getByRole('alert')).toContainText("We couldn't access this webpage");
  await expect(page.getByRole('button', { name: 'Analyze webpage', exact: true })).toBeEnabled();
  await page.getByRole('tab', { name: 'Text', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('reported sarcastic movie review uses the document model and reads negative', async ({ page }, info) => {
  await page.getByRole('tab', { name: 'Webpage' }).click();
  await page.getByLabel('A public webpage URL').fill('https://www.rediff.com/movies/review/the-paradise-movie-review-torture-porn/20260925.htm');
  const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/analyze/url') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Analyze webpage', exact: true }).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const data = await response.json();
  expect(data.overall_sentiment).toBe('negative');
  expect(data.analysis_model).toBe('imdb_review');
  expect(data.aggregation).toBe('full_review_document');
  expect(data.scores.negative).toBeGreaterThan(0.5);
  await expect(page.getByTestId('sentiment-result')).toHaveText('Negative');
  await expect(page.getByTestId('confidence-result')).toHaveText(`${(data.confidence * 100).toFixed(1)}%`);
  await expect(page.getByText('Movie-review SVM · complete document', { exact: true })).toBeVisible();
  expect(data.results.some((unit: { text: string }) => unit.text.includes('Our Services'))).toBe(false);
  const width = await page.evaluate(() => ({ body: document.documentElement.scrollWidth, viewport: innerWidth }));
  expect(width.body).toBeLessThanOrEqual(width.viewport);
  await page.screenshot({ path: resolve(`../artifacts/frontend-review-${info.project.name}.png`), fullPage: true });
});
