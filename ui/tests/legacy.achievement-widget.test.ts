import { beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const WIDGETS_JS = readFileSync(
  path.join(__dirname, '..', '..', 'out', 'html', 'widgets.js'),
  'utf8',
);

const registry = [
  { id: 'a', name: 'A', description: 'Do A.', stars: 1, image: 'img/a.png' },
  { id: 'b', name: 'B', description: 'Do B.', stars: 2, image: 'img/b.png' },
];

function mount(scope: 'ever' | 'playthrough', q: Record<string, unknown>): HTMLElement {
  document.body.innerHTML =
    `<main><div data-widget="achievement-gallery" data-props='{"scope":"${scope}"}'></div></main>`;
  const root = document.querySelector('main')!;
  (window as any).mountWidgets(root, q);
  return root.querySelector('[data-widget="achievement-gallery"]')!;
}

describe('the old shell achievement-gallery handler', () => {
  beforeEach(() => {
    (window as any).dendryUI = { game: { data: { achievements: { achievements: registry } } } };
    (window as any).initCataloniaPolls = vi.fn();
    (window as any).initCatLocalMap = vi.fn();
    (window as any).initCongresoMap = vi.fn();
    (window as any).initCatCoalitions = vi.fn();
    (window as any).initCongresoPartyTour = vi.fn();
    window.eval(WIDGETS_JS);
  });

  it('keeps the full locked/unlocked catalogue for scope="ever"', () => {
    const gallery = mount('ever', { achievement_a: 1 });
    expect(gallery.querySelectorAll('.achievement')).toHaveLength(2);
    expect(gallery.querySelectorAll('.achievement--unlocked')).toHaveLength(1);
    expect(gallery.querySelectorAll('.achievement--locked')).toHaveLength(1);
  });

  it('renders only achievements completed in this playthrough', () => {
    const gallery = mount('playthrough', { achievement_a: 1, game_achievement_b: 1 });
    expect(gallery.querySelectorAll('.achievement')).toHaveLength(1);
    expect(gallery.textContent).toContain('B');
    expect(gallery.textContent).not.toContain('A');
    expect(gallery.querySelector('.achievement--locked')).toBeNull();
  });

  it('clears stale rows when this playthrough has no achievements', () => {
    const gallery = mount('playthrough', { game_achievement_a: 1 });
    expect(gallery.querySelectorAll('.achievement')).toHaveLength(1);

    (window as any).mountWidgets(document.querySelector('main'), {});
    expect(gallery.querySelectorAll('.achievement')).toHaveLength(0);
  });
});
