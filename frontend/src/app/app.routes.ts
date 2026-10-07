import { Routes } from '@angular/router';
import { AnalysisPage } from './pages/analysis/analysis.page';

export const routes: Routes = [
  { path: '', component: AnalysisPage },
  { path: '**', redirectTo: '' },
];
