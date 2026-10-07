import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { PercentPipe, TitleCasePipe } from '@angular/common';
import { Sentiment, TextAnalysisResponse } from '../../models/sentiment.models';

@Component({
  selector: 'app-polarity-result',
  imports: [PercentPipe, TitleCasePipe],
  templateUrl: './polarity-result.component.html',
  styleUrl: './polarity-result.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class PolarityResult {
  readonly result = input.required<{ sentiment: Sentiment; confidence: number }>();
  readonly scores = input<TextAnalysisResponse['scores'] | null>(null);
  readonly context = input('TEXT SENTIMENT');
  readonly uncertaintyLabel = input('Neutral');
  readonly trainedNeutral = computed(() => this.scores()?.neutral !== undefined);
}
