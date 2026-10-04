import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import AchievementGallery from '../src/components/viz/AchievementGallery.vue';
import { useGameStore } from '../src/stores/game';
import { i18n, setLocale } from '../src/i18n';

// Real Pinia + a hand-authored "compiled game" object (same pattern
// glossary.prose.test.ts uses for game.data.glossary — convertJSONToGame is
// a plain JSON.parse, so it does not care that this bypassed the real
// source/data -> compiler route). AchievementGallery needs BOTH the
// REGISTRY (off the game store, static) and the per-id unlocked bit (off
// `q`, passed as a prop by WidgetHost) — see the component's own header
// comment for why those are two different sources.
const GAME = {
  info: { title: 'Test', ifid: 'test-game', languages: ['en', 'ca'] },
  scenes: {
    root: {
      id: 'root',
      type: 'scene',
      title: 'Root',
      content: [{ type: 'paragraph', content: ['Hi.'] }],
      options: [],
    },
  },
  qualities: {},
  qdisplays: {},
  data: {
    achievements: {
      achievements: [
        { id: 'a', name: 'Achievement A', description: 'Do A.', stars: 1, image: 'img/a.png' },
        { id: 'b', name: 'Achievement B', description: 'Do B.', stars: 5, image: 'img/b.png' },
        { id: 'secret', name: 'Secret Achievement', description: 'Discover this.', stars: 5, image: 'img/secret.png', hidden: true },
      ],
    },
  },
};

function mountGallery(props: { scope?: 'ever' | 'playthrough'; q?: Record<string, unknown> }) {
  const pinia = createPinia();
  setActivePinia(pinia);
  const game = useGameStore();
  game.initFromText(JSON.stringify(GAME));
  game.newGame();
  return mount(AchievementGallery, { props, global: { plugins: [pinia, i18n] } });
}

describe('AchievementGallery', () => {
  beforeEach(() => {
    localStorage.clear();
    setLocale('en');
  });

  afterEach(() => vi.useRealTimers());

  it('renders ordinary registry entries in order without exposing locked hidden achievements', () => {
    const w = mountGallery({ q: {} });
    const rows = w.findAll('[data-test^="achievement-row-"]');
    expect(rows).toHaveLength(2);
    expect(rows[0].attributes('data-test')).toBe('achievement-row-a');
    expect(rows[1].attributes('data-test')).toBe('achievement-row-b');
    expect(w.text()).not.toContain('Secret Achievement');
  });

  it('marks a row unlocked when scope="ever" and Q.achievement_<id> is truthy', () => {
    const w = mountGallery({ scope: 'ever', q: { achievement_a: 1 } });
    expect(w.get('[data-test="achievement-row-a"]').classes()).toContain(
      'achievement-row--unlocked',
    );
    expect(w.get('[data-test="achievement-row-b"]').classes()).toContain('achievement-row--locked');
  });

  it('defaults to scope="ever" when no scope prop is given', () => {
    const w = mountGallery({ q: { achievement_b: 1 } });
    expect(w.get('[data-test="achievement-row-b"]').classes()).toContain(
      'achievement-row--unlocked',
    );
  });

  it('renders only Q.game_achievement_<id> unlocks when scope="playthrough"', () => {
    // achievement_a is set (ever unlocked) but game_achievement_a is not
    // (not unlocked THIS playthrough) — playthrough scope must ignore the
    // ever-unlocked quality entirely.
    const w = mountGallery({ scope: 'playthrough', q: { achievement_a: 1, game_achievement_b: 1 } });
    expect(w.find('[data-test="achievement-row-a"]').exists()).toBe(false);
    expect(w.get('[data-test="achievement-row-b"]').classes()).toContain(
      'achievement-row--unlocked',
    );
    expect(w.find('.achievement-row--locked').exists()).toBe(false);
  });

  it('renders an empty gallery when this playthrough has no achievements', () => {
    const w = mountGallery({ scope: 'playthrough', q: { achievement_a: 1 } });
    expect(w.findAll('[data-test^="achievement-row-"]')).toHaveLength(0);
  });

  it('renders the correct filled/empty star split', () => {
    const w = mountGallery({ q: {} });
    const rowA = w.get('[data-test="achievement-row-a"]');
    expect(rowA.findAll('.star--filled')).toHaveLength(1);
    expect(rowA.findAll('.star--empty')).toHaveLength(4);
    const rowB = w.get('[data-test="achievement-row-b"]');
    expect(rowB.findAll('.star--filled')).toHaveLength(5);
    expect(rowB.findAll('.star--empty')).toHaveLength(0);
  });

  it('shows the real unlocked and registry totals', () => {
    const w = mountGallery({ q: { achievement_a: 1 } });
    expect(w.get('[data-test="achievement-count"]').text()).toBe('1 / 2 unlocked');
    expect(w.get('[data-test="achievement-percentage"]').text()).toBe('50%');
    expect(w.get('.achievement-gallery-progress-fill').attributes('style')).toContain('width: 50%');
  });

  it('reveals unlocked hidden achievements and separates their bonus count above 100%', () => {
    const w = mountGallery({ q: { achievement_a: 1, achievement_b: 1, achievement_secret: 1 } });
    expect(w.get('[data-test="achievement-row-secret"]').classes()).toContain('achievement-row--unlocked');
    expect(w.get('[data-test="achievement-count"]').text()).toBe('2+1 / 2 unlocked');
    expect(w.get('[data-test="achievement-percentage"]').text()).toBe('150%');
    expect(w.get('.achievement-gallery-progress-fill').attributes('style')).toContain('width: 100%');
    expect(w.get('[role="progressbar"]').attributes('aria-valuetext')).toBe('150% — 2+1 / 2 unlocked');
  });

  it('updates hidden visibility and completion when unlock qualities change', async () => {
    const w = mountGallery({ q: {} });
    await w.setProps({ q: { achievement_secret: 1 } });
    expect(w.find('[data-test="achievement-row-secret"]').exists()).toBe(true);
    expect(w.get('[data-test="achievement-count"]').text()).toBe('0+1 / 2 unlocked');
    expect(w.get('[data-test="achievement-percentage"]').text()).toBe('50%');
    await w.setProps({ q: {} });
    expect(w.find('[data-test="achievement-row-secret"]').exists()).toBe(false);
    expect(w.get('[data-test="achievement-count"]').text()).toBe('0 / 2 unlocked');
  });

  it('counts hidden unlocks only in the selected playthrough scope', () => {
    const w = mountGallery({ scope: 'playthrough', q: {
      achievement_a: 1, achievement_secret: 1, game_achievement_b: 1,
    } });
    expect(w.find('[data-test="achievement-row-secret"]').exists()).toBe(false);
    expect(w.get('[data-test="achievement-count"]').text()).toBe('1 / 2 unlocked');
    expect(w.get('[data-test="achievement-percentage"]').text()).toBe('50%');
  });

  it('includes hidden achievements earned this run in the playthrough summary', () => {
    const w = mountGallery({ scope: 'playthrough', q: { game_achievement_secret: 1 } });
    expect(w.findAll('[data-test^="achievement-row-"]')).toHaveLength(1);
    expect(w.get('[data-test="achievement-count"]').text()).toBe('0+1 / 2 unlocked');
  });

  it('shows zero completion even when the playthrough list has no rows', () => {
    const w = mountGallery({ scope: 'playthrough', q: { achievement_secret: 1 } });
    expect(w.findAll('[data-test^="achievement-row-"]')).toHaveLength(0);
    expect(w.get('[data-test="achievement-count"]').text()).toBe('0 / 2 unlocked');
    expect(w.get('[data-test="achievement-percentage"]').text()).toBe('0%');
  });

  it('renders relative labels below 24 hours from the engine-owned ledger', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-08-15T12:00:00.000Z'));
    localStorage.setItem('dnt:test-game:achievements', JSON.stringify({
      a: { unlockedAt: '2026-08-15T11:58:00.000Z' },
      b: { unlockedAt: '2026-08-15T04:00:00.000Z' },
    }));
    const w = mountGallery({ q: { achievement_a: 1, achievement_b: 1 } });
    expect(w.get('[data-test="achievement-date-a"]').text()).toBe('2 min. ago');
    expect(w.get('[data-test="achievement-date-b"]').text()).toBe('8 h. ago');
    expect(w.get('[data-test="achievement-date-a"]').element.parentElement?.classList)
      .toContain('achievement-row-meta');
    expect(w.find('[data-test="achievement-row-a"] .achievement-row-meta .achievement-row-stars').exists())
      .toBe(true);
  });

  it('renders locale-formatted absolute dates after 24 hours', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-08-15T12:00:00.000Z'));
    localStorage.setItem('dnt:test-game:achievements', JSON.stringify({
      a: { unlockedAt: '2025-10-15T12:00:00.000Z' },
    }));
    const w = mountGallery({ q: { achievement_a: 1 } });
    expect(w.get('[data-test="achievement-date-a"]').text()).toBe(
      new Intl.DateTimeFormat('en', { day: 'numeric', month: 'short', year: 'numeric' })
        .format(new Date('2025-10-15T12:00:00.000Z')),
    );
  });

  it('labels legacy numeric and malformed records with an unknown date', () => {
    localStorage.setItem('dnt:test-game:achievements', JSON.stringify({ a: 1, b: { unlockedAt: 'nope' } }));
    const w = mountGallery({ q: { achievement_a: 1, achievement_b: 1 } });
    expect(w.get('[data-test="achievement-date-a"]').text()).toBe('Unlock date unknown');
    expect(w.get('[data-test="achievement-date-b"]').text()).toBe('Unlock date unknown');
  });

  it('renders name, description and image', () => {
    const w = mountGallery({ q: {} });
    const rowA = w.get('[data-test="achievement-row-a"]');
    expect(rowA.text()).toContain('Achievement A');
    expect(rowA.text()).toContain('Do A.');
    // Registry paths are web-root-relative; the component resolves them
    // against BASE_URL ('/' under vitest), same as HandCard/GlossaryTerm.
    expect(rowA.get('img').attributes('src')).toBe(`${import.meta.env.BASE_URL}img/a.png`);
  });

  it('shows an undefined percentage without dividing by zero when there is no registry', () => {
    const pinia = createPinia();
    setActivePinia(pinia);
    const game = useGameStore();
    game.initFromText(
      JSON.stringify({
        scenes: { root: { id: 'root', type: 'scene', title: 'Root', content: [], options: [] } },
        qualities: {},
        qdisplays: {},
      }),
    );
    game.newGame();
    const w = mount(AchievementGallery, { props: { q: {} }, global: { plugins: [pinia, i18n] } });
    expect(w.find('[data-test="achievement-gallery"]').exists()).toBe(true);
    expect(w.findAll('[data-test^="achievement-row-"]')).toHaveLength(0);
    expect(w.get('[data-test="achievement-count"]').text()).toBe('0 / 0 unlocked');
    expect(w.get('[data-test="achievement-percentage"]').text()).toBe('—%');
    expect(w.get('.achievement-gallery-progress-fill').attributes('style')).toContain('width: 0%');
  });

  it('does not throw when q is undefined (WidgetHost always passes it, but stay defensive)', () => {
    expect(() => mountGallery({})).not.toThrow();
  });
});
