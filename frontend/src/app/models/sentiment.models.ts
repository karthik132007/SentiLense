export type Sentiment = 'positive' | 'neutral' | 'negative';
export type AnalysisMode = 'keyword' | 'text' | 'url';

export interface TextAnalysisRequest {
  text: string;
}

export interface TextAnalysisResponse {
  sentiment: Sentiment;
  confidence: number;
  scores: {
    positive: number;
    negative: number;
    neutral?: number;
  };
}

export interface UrlAnalysisRequest {
  url: string;
}

export interface AnalysisUnit {
  text: string;
  sentiment: Sentiment;
  confidence: number;
  scores: TextAnalysisResponse['scores'];
}

export interface UrlAnalysisResponse {
  url: string;
  title: string;
  domain: string;
  overall_sentiment: Sentiment;
  confidence: number;
  scores: TextAnalysisResponse['scores'];
  distribution: {
    positive: number;
    neutral: number;
    negative: number;
  };
  statistics: {
    characters: number;
    words: number;
    analyzed_units: number;
  };
  results: AnalysisUnit[];
  analysis_model: 'imdb_review' | 'sentiment140' | 'tweeteval' | 'svm_roberta_fusion';
  aggregation: 'full_review_document' | 'mean_segment_probability' | 'mean_topic_passage_probability';
}

export interface KeywordAnalysisResponse {
  keyword: string;
  search_provider: string;
  searched_at_utc: string;
  overall_sentiment: Sentiment;
  confidence: number;
  scores: TextAnalysisResponse['scores'];
  distribution: UrlAnalysisResponse['distribution'];
  statistics: UrlAnalysisResponse['statistics'] & {
    search_results_found: number;
    sources_analyzed: number;
    sources_skipped: number;
    requested_sources: number;
  };
  aggregation: string;
  sources: (UrlAnalysisResponse & {
    topic_evidence: {
      method: string;
      title_matches: boolean;
      matched_passages: number;
      extracted_passages: number;
      extracted_words: number;
    };
  })[];
  skipped_sources: { url: string; title: string; reason: string }[];
}

// The health payload is documented in backend/README.md.
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_backend: string;
}
