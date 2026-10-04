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
  { id: 'secret', name: 'Secret Achievement', description: 'Discover this.', stars: 5, image: 'img/secret.png', hidden: true },
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
    expect(gallery.textContent).not.toContain('Secret Achievement');
    expect(gallery.querySelector('.achievement-count')?.textContent).toBe('1 / 2');
    expect(gallery.querySelector('.achievement-percentage')?.textContent).toBe('50%');
    expect((gallery.querySelector('.achievement-progress-fill') as HTMLElement).style.width).toBe('50%');
  });

  it('reveals unlocked hidden achievements and shows their separate bonus above 100%', () => {
    const gallery = mount('ever', { achievement_a: 1, achievement_b: 1, achievement_secret: 1 });
    expect(gallery.querySelectorAll('.achievement--unlocked')).toHaveLength(3);
    expect(gallery.textContent).toContain('Secret Achievement');
    expect(gallery.querySelector('.achievement-count')?.textContent).toBe('2+1 / 2');
    expect(gallery.querySelector('.achievement-percentage')?.textContent).toBe('150%');
    expect((gallery.querySelector('.achievement-progress-fill') as HTMLElement).style.width).toBe('100%');
    expect(gallery.querySelector('[role="progressbar"]')?.getAttribute('aria-valuetext')).toBe('150% — 2+1 / 2');
  });

  it('refreshes hidden visibility and drops the bonus when the widget is mounted again', () => {
    const gallery = mount('ever', { achievement_secret: 1 });
    expect(gallery.textContent).toContain('Secret Achievement');
    expect(gallery.querySelector('.achievement-count')?.textContent).toBe('0+1 / 2');
    (window as any).mountWidgets(document.querySelector('main'), {});
    expect(gallery.textContent).not.toContain('Secret Achievement');
    expect(gallery.querySelector('.achievement-count')?.textContent).toBe('0 / 2');
  });

  it('renders only achievements completed in this playthrough', () => {
    const gallery = mount('playthrough', { achievement_a: 1, game_achievement_b: 1 });
    expect(gallery.querySelectorAll('.achievement')).toHaveLength(1);
    expect(gallery.textContent).toContain('B');
    expect(gallery.textContent).not.toContain('A');
    expect(gallery.querySelector('.achievement--locked')).toBeNull();
  });

  it('includes only hidden achievements earned in this playthrough', () => {
    const gallery = mount('playthrough', { achievement_a: 1, achievement_secret: 1, game_achievement_secret: 1 });
    expect(gallery.querySelectorAll('.achievement')).toHaveLength(1);
    expect(gallery.textContent).toContain('Secret Achievement');
    expect(gallery.querySelector('.achievement-count')?.textContent).toBe('0+1 / 2');
    expect(gallery.querySelector('.achievement-percentage')?.textContent).toBe('50%');
  });

  it('clears stale rows when this playthrough has no achievements', () => {
    const gallery = mount('playthrough', { game_achievement_a: 1 });
    expect(gallery.querySelectorAll('.achievement')).toHaveLength(1);

    (window as any).mountWidgets(document.querySelector('main'), {});
    expect(gallery.querySelectorAll('.achievement')).toHaveLength(0);
    expect(gallery.querySelector('.achievement-count')?.textContent).toBe('0 / 2');
    expect(gallery.querySelector('.achievement-percentage')?.textContent).toBe('0%');
  });

  it('avoids division by zero when every registry entry is hidden', () => {
    (window as any).dendryUI.game.data.achievements.achievements = registry.filter((a) => a.hidden);
    const gallery = mount('ever', { achievement_secret: 1 });
    expect(gallery.querySelector('.achievement-count')?.textContent).toBe('0+1 / 0');
    expect(gallery.querySelector('.achievement-percentage')?.textContent).toBe('—%');
    expect((gallery.querySelector('.achievement-progress-fill') as HTMLElement).style.width).toBe('0%');
  });
});
