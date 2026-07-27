import { createRequire } from 'node:module';
import { describe, expect, it } from 'vitest';

const require = createRequire(import.meta.url);
const { formatSaveSceneName, formatSaveTimestamp } = require(
  '../../vendor/dendrynexus-ten/lib/ui/save-label.js',
) as {
  formatSaveSceneName(scene: string): string;
  formatSaveTimestamp(timestamp: string): string;
};

describe('old-shell save labels', () => {
  it('removes post_event, nested scene suffixes, and authoring punctuation', () => {
    expect(formatSaveSceneName('parlament_digital.post_event')).toBe('Parlament Digital');
    expect(formatSaveSceneName('post_event.events_choice')).toBe('Events Choice');
    expect(formatSaveSceneName('parlament_inheritance_tax.vote_result')).toBe(
      'Parlament Inheritance Tax',
    );
  });

  it('sanitizes old stored timestamps when displaying them', () => {
    expect(formatSaveTimestamp('post_event.events_choice\n(7/26/2026, 2:30 PM)')).toBe(
      'Events Choice\n(7/26/2026, 2:30 PM)',
    );
  });
});
