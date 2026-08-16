import { beforeEach, describe, expect, it } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { useGameStore } from '../src/stores/game';
import { useBriefStore } from '../src/stores/brief';
import { useShellStore } from '../src/stores/shell';

const GAME = join(__dirname, '..', '..', 'out', 'game.json');

describe('Library live-scene behavior', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    const game = useGameStore();
    game.initFromText(readFileSync(GAME, 'utf8'));
    game.newGame();
  });

  it('discovers the authored root by role and keeps article navigation live', () => {
    const game = useGameStore();
    const brief = useBriefStore();
    expect(brief.libraryId).toBe('library');

    brief.openLibrary();
    expect(game.frame?.sceneId).toBe('library.menu');
    expect(brief.libraryAtIndex).toBe(true);
    expect(brief.libraryIndexChoices.map((choice) => choice.id)).toContain('library.catalan_system_lib');

    const article = brief.libraryIndexChoices.findIndex((choice) => choice.id === 'library.catalan_system_lib');
    brief.chooseLibraryIndex(article);
    expect(game.frame?.sceneId).toBe('library.catalan_system_lib');
    expect(brief.libraryAtIndex).toBe(false);
    expect(game.frame?.html).toContain('Catalan Political System');
    expect(game.frame?.html).not.toContain('Herein you may find all varieties');

    brief.chooseLibraryArticle(0);
    expect(game.frame?.sceneId).toBe('library.menu');
    expect(brief.libraryAtIndex).toBe(true);
  });

  it.each([
    ['Desk', 'main', 'desk'],
    ['dossier', 'debug_card', 'card'],
    ['event', '2012diada', 'event'],
    ['generic page', 'root.start', 'main-menu-item'],
  ])('closes through backSpecialScene to the exact %s origin', (_label, origin, role) => {
    const game = useGameStore();
    const brief = useBriefStore();
    game.goToScene(origin);
    const exactOrigin = game.frame?.sceneId;
    brief.openLibrary();
    brief.closeLibrary();
    expect(game.frame?.sceneId).toBe(exactOrigin);
    expect(game.effectiveRole).toBe(role);
  });

  it('switching to another Brief tab closes Library first and selects that tab', () => {
    const game = useGameStore();
    const brief = useBriefStore();
    game.goToScene('main');
    brief.openLibrary();
    brief.select('status_new.chamber');
    expect(game.frame?.sceneId).toBe('main');
    expect(brief.activeTab).toBe('status_new.chamber');
  });

  it('remains a stable pausable surface at both index and article depth', () => {
    const game = useGameStore();
    const brief = useBriefStore();
    const shell = useShellStore();
    brief.openLibrary();
    expect(shell.canPause('page', game.effectiveRole, false)).toBe(true);
    brief.chooseLibraryIndex(0);
    expect(shell.canPause('page', game.effectiveRole, false)).toBe(true);
  });

  it('restores an article as Library after a serialized pause-pane detour', () => {
    const game = useGameStore();
    const brief = useBriefStore();
    game.goToScene('main');
    brief.openLibrary();
    brief.chooseLibraryIndex(0);
    const article = game.frame?.sceneId;
    const pausedState = game.captureState();
    game.goToScene('about');
    game.restoreState(pausedState);
    expect(game.frame?.sceneId).toBe(article);
    expect(game.effectiveRole).toBe('library-item');
    brief.closeLibrary();
    expect(game.frame?.sceneId).toBe('main');
  });
});
