import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

@Component({
  selector: 'app-workspace-navigation',
  templateUrl: './workspace-navigation.component.html',
  styleUrl: './workspace-navigation.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class WorkspaceNavigation {
  readonly sourceTarget = input<string | null>(null);
  readonly reset = output<void>();
}
