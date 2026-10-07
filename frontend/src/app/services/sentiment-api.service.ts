import { HttpClient } from '@angular/common/http';
import { inject, Injectable, InjectionToken } from '@angular/core';
import { Observable, timeout } from 'rxjs';
import { environment } from '../../environments/environment';
import {
  HealthResponse, TextAnalysisRequest, TextAnalysisResponse,
  UrlAnalysisRequest, UrlAnalysisResponse,
  KeywordAnalysisResponse,
} from '../models/sentiment.models';

export const API_BASE_URL = new InjectionToken<string>('API_BASE_URL', {
  providedIn: 'root',
  factory: () => environment.apiBaseUrl,
});

@Injectable({ providedIn: 'root' })
export class SentimentApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL).replace(/\/+$/, '');

  healthCheck(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/api/health`).pipe(timeout(8000));
  }

  analyzeText(text: string): Observable<TextAnalysisResponse> {
    const body: TextAnalysisRequest = { text };
    return this.http.post<TextAnalysisResponse>(`${this.baseUrl}/api/analyze/text`, body)
      .pipe(timeout(60000));
  }

  analyzeKeyword(keyword: string, maxSources: number): Observable<KeywordAnalysisResponse> {
    return this.http.post<KeywordAnalysisResponse>(`${this.baseUrl}/api/analyze/keyword`, { keyword, max_sources: maxSources })
      .pipe(timeout(180000));
  }

  analyzeUrl(url: string): Observable<UrlAnalysisResponse> {
    const body: UrlAnalysisRequest = { url };
    return this.http.post<UrlAnalysisResponse>(`${this.baseUrl}/api/analyze/url`, body)
      .pipe(timeout(120000));
  }
}
