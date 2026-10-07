import { ChangeDetectionStrategy, Component, computed, DestroyRef, inject, OnDestroy, signal } from '@angular/core';
import { DecimalPipe, PercentPipe, TitleCasePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { finalize, Observable, Subscription } from 'rxjs';
import { SentimentApiService } from '../../services/sentiment-api.service';
import { analysisErrorMessage } from '../../services/analysis-errors';
import { WorkspaceNavigation } from '../../components/workspace-navigation/workspace-navigation.component';
import { PolarityResult } from '../../components/polarity-result/polarity-result.component';
import { AnalysisMode, KeywordAnalysisResponse, Sentiment, TextAnalysisResponse, UrlAnalysisResponse } from '../../models/sentiment.models';

@Component({
  selector: 'app-analysis-page',
  imports: [FormsModule, PercentPipe, DecimalPipe, TitleCasePipe, WorkspaceNavigation, PolarityResult],
  templateUrl: './analysis.page.html',
  styleUrl: './analysis.page.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AnalysisPage implements OnDestroy {
  private readonly api = inject(SentimentApiService);
  private readonly destroyRef = inject(DestroyRef);
  private request?: Subscription;
  private healthRequest?: Subscription;

  readonly mode = signal<AnalysisMode>('keyword');
  readonly keyword = signal('');
  readonly maxSources = signal(5);
  readonly sourceLimits = [3, 5];
  readonly keywordExamples = ['antigravity', 'electric vehicles', 'remote work'];
  readonly text = signal('');
  readonly url = signal('');
  readonly busy = signal(false);
  readonly connection = signal<'checking' | 'ready' | 'offline'>('checking');
  readonly error = signal('');
  readonly textResult = signal<TextAnalysisResponse | null>(null);
  readonly pageResult = signal<UrlAnalysisResponse | null>(null);
  readonly keywordResult = signal<KeywordAnalysisResponse | null>(null);
  readonly analyzedInput = signal('');
  readonly filter = signal<'all' | Sentiment>('all');
  readonly examples = ['I absolutely love this!', 'I hate this product.', 'The product arrived yesterday.'];
  readonly sentiments: Sentiment[] = ['positive', 'neutral', 'negative'];

  sentimentLabel(sentiment: Sentiment): string {
    return sentiment === 'neutral' && this.keywordResult() && this.scores()?.neutral === undefined ? 'Uncertain' : sentiment;
  }

  readonly valid = computed(() => {
    if (this.mode() === 'keyword') return this.keyword().trim().length > 0 && this.keyword().length <= 200;
    if (this.mode() === 'text') return this.text().trim().length > 0 && this.text().length <= 10000;
    try {
      const value = new URL(this.url().trim());
      return ['http:', 'https:'].includes(value.protocol) && !!value.hostname && this.url().length <= 2048;
    } catch { return false; }
  });
  readonly summary = computed(() => {
    const topic = this.keywordResult();
    if (topic) return { sentiment: topic.overall_sentiment, confidence: topic.confidence };
    const text = this.textResult();
    if (text) return { sentiment: text.sentiment, confidence: text.confidence };
    const page = this.pageResult();
    return page ? { sentiment: page.overall_sentiment, confidence: page.confidence } : null;
  });
  readonly scores = computed(() => this.keywordResult()?.scores ?? this.textResult()?.scores ?? this.pageResult()?.scores ?? null);
  readonly units = computed(() => this.pageResult()?.results.filter(unit => this.filter() === 'all' || unit.sentiment === this.filter()) ?? []);
  readonly sources = computed(() => this.keywordResult()?.sources.filter(source => this.filter() === 'all' || source.overall_sentiment === this.filter()) ?? []);

  constructor() { this.checkConnection(); }

  checkConnection(): void {
    this.healthRequest?.unsubscribe();
    this.connection.set('checking');
    this.healthRequest = this.api.healthCheck().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: health => this.connection.set(health.status === 'ok' && health.model_loaded && health.model_backend === 'sentiment_model' ? 'ready' : 'offline'),
      error: () => this.connection.set('offline'),
    });
  }

  setMode(mode: AnalysisMode): void {
    if (mode === this.mode()) return;
    this.request?.unsubscribe();
    this.busy.set(false);
    this.mode.set(mode);
    this.clearResults();
  }

  onTabKeydown(event: KeyboardEvent): void {
    const modes: AnalysisMode[] = ['keyword', 'text', 'url'];
    const current = modes.indexOf(this.mode());
    const next = event.key === 'ArrowRight' ? (current + 1) % modes.length
      : event.key === 'ArrowLeft' ? (current + modes.length - 1) % modes.length
      : event.key === 'Home' ? 0 : event.key === 'End' ? modes.length - 1 : null;
    if (next === null) return;
    event.preventDefault();
    this.setMode(modes[next]);
    document.getElementById(`${modes[next]}-tab`)?.focus();
  }

  useExample(value: string): void {
    this.setMode('text');
    this.text.set(value);
    this.clearResults();
  }

  clearResults(): void {
    this.error.set('');
    this.textResult.set(null);
    this.pageResult.set(null);
    this.keywordResult.set(null);
    this.filter.set('all');
  }

  newAnalysis(): void {
    this.request?.unsubscribe();
    this.busy.set(false);
    this.keyword.set('');
    this.text.set('');
    this.url.set('');
    this.analyzedInput.set('');
    this.clearResults();
    document.getElementById('main-content')?.scrollIntoView({ block: 'start' });
    document.getElementById(`${this.mode()}-input`)?.focus();
  }

  submit(): void {
    if (!this.valid() || this.busy() || this.connection() !== 'ready') return;
    this.request?.unsubscribe();
    this.clearResults();
    const mode = this.mode();
    const input = (mode === 'keyword' ? this.keyword() : mode === 'text' ? this.text() : this.url()).trim();
    this.analyzedInput.set(input);
    this.busy.set(true);
    const operation: Observable<TextAnalysisResponse | UrlAnalysisResponse | KeywordAnalysisResponse> = mode === 'keyword'
      ? this.api.analyzeKeyword(input, this.maxSources()) : mode === 'text'
        ? this.api.analyzeText(input) : this.api.analyzeUrl(input);
    this.request = operation.pipe(takeUntilDestroyed(this.destroyRef), finalize(() => this.busy.set(false))).subscribe({
      next: result => {
        if (mode === 'keyword') this.keywordResult.set(result as KeywordAnalysisResponse);
        else if (mode === 'text') this.textResult.set(result as TextAnalysisResponse);
        else this.pageResult.set(result as UrlAnalysisResponse);
      },
      error: error => this.error.set(analysisErrorMessage(error, mode)),
    });
  }

  download(): void {
    const result = this.keywordResult() ?? this.textResult() ?? this.pageResult();
    if (!result) return;
    const file = new Blob([JSON.stringify({ input: this.analyzedInput(), result }, null, 2)], { type: 'application/json' });
    const objectUrl = URL.createObjectURL(file);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = 'sentilense-analysis.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  }

  ngOnDestroy(): void { this.request?.unsubscribe(); this.healthRequest?.unsubscribe(); }
}
