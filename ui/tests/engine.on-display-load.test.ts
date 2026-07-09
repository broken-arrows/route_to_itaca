import { describe, it, expect } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

// Minimal compiled-shape game: root has onDisplay (NOT onArrival) that
// increments a counter, so we can tell display-time code from arrival-time.
const game = {
  scenes: {
    root: {
      id: 'root',
      type: 'scene',
      title: 'Root',
      newPage: true,
      onDisplay: [{ $code: 'Q.shown = (Q.shown || 0) + 1;' }],
      content: [{ type: 'paragraph', content: ['Hello.'] }],
      options: [{ id: '@other' }],
    },
    other: {
      id: 'other',
      type: 'scene',
      title: 'Other',
      newPage: true,
      content: [{ type: 'paragraph', content: ['Other scene.'] }],
      options: [],
    },
  },
  qualities: {},
  qdisplays: {},
  tagLookup: {},
};

describe('onDisplay re-fires on save load', () => {
  it('runs onDisplay again when state is loaded via setState', () => {
    const adapter = DendryAdapter.fromJSONText(JSON.stringify(game));
    adapter.beginGame();
    expect(adapter.qualities.shown).toBe(1); // fired on entry
    const saved = adapter.exportStateJSON();
    adapter.importStateJSON(saved);
    expect(adapter.qualities.shown).toBe(2); // fired again on load
  });
});

// Second, compiled-shape game: root's onDisplay bumps Q.shown, and the
// '@advanced' option is only viewable once Q.shown >= 2. On normal entry
// (beginGame), onDisplay fires once, so shown === 1 and 'advanced' is
// correctly hidden. On load (setState), onDisplay must fire AGAIN before
// choices are compiled (exactly as it does on normal entry, via
// displaySceneContent -> _compileChoices) so shown becomes 2 and 'advanced'
// becomes visible. If onDisplay ran AFTER choice compilation (the bug this
// locks against), choices would be compiled while shown is still 1 and
// 'advanced' would wrongly stay absent post-load.
// Root also carries an always-visible '@other' option. Without it, the
// scene would have zero choosable options while shown < 2, which trips
// DendryEngine's "no choosable options" path into an unwanted gameOver
// (engine.js __changeScene, ~line 1111) on the very first entry -- that
// would short-circuit setState's isGameOver() branch on load and mask the
// ordering bug entirely. '@other' keeps the scene choosable throughout so
// the test isolates the onDisplay/compile-choices ordering specifically.
const gameWithConditionalChoice = {
  scenes: {
    root: {
      id: 'root',
      type: 'scene',
      title: 'Root',
      newPage: true,
      onDisplay: [{ $code: 'Q.shown = (Q.shown || 0) + 1;' }],
      content: [{ type: 'paragraph', content: ['Hello.'] }],
      options: [
        { id: '@other' },
        { id: '@advanced', viewIf: { $code: 'return (Q.shown || 0) >= 2;' } },
      ],
    },
    other: {
      id: 'other',
      type: 'scene',
      title: 'Other',
      content: [{ type: 'paragraph', content: ['Other scene.'] }],
    },
    advanced: {
      id: 'advanced',
      type: 'scene',
      title: 'Advanced',
      content: [{ type: 'paragraph', content: ['Advanced scene.'] }],
    },
  },
  qualities: {},
  qdisplays: {},
  tagLookup: {},
};

describe('onDisplay runs before choice compilation on load', () => {
  it('makes a viewIf-gated choice visible after load once onDisplay has re-fired', () => {
    const adapter = DendryAdapter.fromJSONText(JSON.stringify(gameWithConditionalChoice));

    const initial = adapter.beginGame();
    expect(adapter.qualities.shown).toBe(1);
    // Not yet visible: shown is only 1, viewIf requires >= 2.
    expect(initial.choices.some((c) => c.id === 'advanced')).toBe(false);

    const saved = adapter.exportStateJSON(); // persists shown = 1
    const loaded = adapter.importStateJSON(saved);

    // With the fix, onDisplay re-runs (shown -> 2) BEFORE choices are
    // compiled, so 'advanced' is now present in the loaded frame's choices.
    expect(loaded.choices.some((c) => c.id === 'advanced')).toBe(true);
  });
});
