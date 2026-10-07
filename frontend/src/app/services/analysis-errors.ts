import { HttpErrorResponse } from '@angular/common/http';
import { TimeoutError } from 'rxjs';
import { AnalysisMode } from '../models/sentiment.models';

export function analysisErrorMessage(error: unknown, mode: AnalysisMode): string {
  if (error instanceof TimeoutError) {
    return 'Analysis is taking longer than expected. Please try again.';
  }
  if (mode === 'keyword' && error instanceof HttpErrorResponse && error.status > 0) {
    const body: unknown = error.error;
    if ([400, 404, 422, 503].includes(error.status) && body && typeof body === 'object' && 'detail' in body && typeof body.detail === 'string') {
      return body.detail;
    }
    if (error.status >= 400 && error.status < 500) return 'Enter a keyword or short phrase and try again.';
  }
  if (!(error instanceof HttpErrorResponse) || error.status === 0 || error.status >= 500) {
    return 'The sentiment service is currently unavailable. Please try again shortly.';
  }
  if (error.status === 429) {
    return 'Too many requests. Please wait a moment before analyzing again.';
  }
  if (error.status === 413) {
    return 'This content is too long to analyze. Try a shorter text or webpage.';
  }
  if (mode === 'url') {
    return "We couldn't access this webpage. Check the URL and try a publicly accessible page.";
  }
  return "We couldn't analyze this text. Check your input and try again.";
}
