import { describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { nextTick } from 'vue';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { readFileSync } from 'node:fs';
import { i18n } from '../src/i18n';
import GameView from '../src/views/GameView.vue';
import { useGameStore } from '../src/stores/game';
import { useSettingsStore } from '../src/stores/settings';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';
import { DendryAdapter } from '../src/engine/adapter';

async function compile(root: string, other: Record<string, string> = {}) {
  const files = [
    { name: 'info.dry', contents: 'title: Test\nauthor: Test\n' },
    ...Object.entries({ root, ...other }).map(([id, contents]) => ({ name: id + '.scene.dry', contents })),
  ];
  return new Promise<string>((resolve, reject) => compileGame(files, (error, game) => {
    if (error) return reject(error);
    convertGameToJSON(game, 0, (err, json) => err ? reject(err) : resolve(json!));
  }));
}

describe('conditional scene images', () => {
  it('captures the first matching portrait and keeps it with the displayed page', async () => {
    const game = DendryAdapter.fromJSONText(await compile(`title: Start
on-arrival: {! Q.leader = 1; !}
face-image: first.jpg if {! const leader = Q.leader; if (leader) return true; !}; second.jpg if leader; fallback.jpg
set-bg: hall.jpg if leader
set-sprites: topLeft: a.jpg, bottomRight: b.jpg if leader; none

Start.

- @next
- choice-image: override.jpg if leader

@next
title: Next
choice-image: target.jpg
face-image: other.jpg if other
set-bg: other.jpg if other
on-arrival: {! Q.leader = 0; !}

Next.

- @root
`));
    const frame = game.beginGame();
    expect(frame.faceImage).toBe('first.jpg');
    expect(frame.choices[0].choiceImage).toBe('override.jpg');
    expect(frame.sprites).toEqual({ topleft: 'a.jpg', bottomright: 'b.jpg' });
    expect(game.choose(0)).toMatchObject({ faceImage: 'first.jpg', bg: 'hall.jpg' });
    const saved = game.exportStateJSON();
    expect(game.refreshLocale('ca', {})?.faceImage).toBe('first.jpg');
    expect(game.importStateJSON(saved)).toMatchObject({ faceImage: 'first.jpg', sprites: frame.sprites });
  });

  it('uses destination art when an option has no match and links relative visit references', async () => {
    const game = DendryAdapter.fromJSONText(await compile(`title: Start

Start.

- @next
- choice-image: already.jpg if @next

@next
title: Next
count-visits-max: 2
choice-image: other.jpg if other; default.jpg

Next.

- @root
`));
    expect(game.beginGame().choices[0].choiceImage).toBe('default.jpg');
    game.choose(0);
    expect(game.choose(0).choices[0].choiceImage).toBe('already.jpg');
  });

  it('resolves deck, pinned and drawn card art and replaces or clears a sprite layout', async () => {
    const game = DendryAdapter.fromJSONText(await compile(
      'title: Start\nset-sprites: topRight: old.jpg\n\n- @desk\n', {
      desk: 'title: Desk\nrole: desk\nmax-cards: 1\nset-sprites: bottomLeft: new.jpg if {! return true; !}; none\n\n- @deck\n- @pin\n',
      deck: 'title: Deck\nrole: deck\ncard-image: no.jpg if missing; deck.jpg\n\n- @card\n',
      card: 'title: Card\nrole: card\ncard-image: card.jpg if {! return true; !}\n\n- @desk\n',
      pin: 'title: Pinned\nrole: pinned-advisor\ncard-image: no.jpg if missing; pin.jpg\nset-sprites: none if {! return true; !}\n\n- @desk\n',
    }));
    game.beginGame();
    const desk = game.choose(0);
    expect(desk.decks[0].image).toBe('deck.jpg');
    expect(desk.pinned[0].image).toBe('pin.jpg');
    expect(desk.sprites).toEqual({ bottomleft: 'new.jpg' });
    expect(game.drawCard('deck').frame.hand[0].image).toBe('card.jpg');
    expect(game.goToScene('pin').sprites).toEqual({});
  });

  it('restores the original layout and portrait when returning from a special scene', async () => {
    const game = DendryAdapter.fromJSONText(await compile(
      'title: Start\non-arrival: {! Q.active = 1; !}\nface-image: first.jpg if active\nset-bg: hall.jpg if active\nset-sprites: topRight: old.jpg if active; none\n\nStart.\n\n- @special\n', {
      special: 'title: Special\nis-special: true\nnew-page: true\non-arrival: {! Q.active = 0; !}\nset-bg: changed.jpg\nset-sprites: none\n\nSpecial.\n\n- @backSpecialScene: Back\n',
    }));
    const start = game.beginGame();
    expect(game.choose(0).sprites).toEqual({});
    game.importStateJSON(game.exportStateJSON());
    expect(game.choose(0)).toMatchObject({ faceImage: start.faceImage, bg: start.bg, sprites: start.sprites });
  });

  it('renders selected images on Desk and honours the image preference', async () => {
    const pinia = createPinia();
    setActivePinia(pinia);
    const game = useGameStore();
    game.initFromText(await compile(`title: Start
role: intro
face-image: face.jpg if {! return true; !}
set-bg: hall.jpg if {! return true; !}
set-sprites: topRight: right.jpg if {! return true; !}

Start.

- @root
- choice-image: choice.jpg if {! return true; !}
`));
    game.newGame();
    const view = mount(GameView, { global: { plugins: [pinia, i18n] } });
    try {
      const images = () => view.findAll('img').map(img => img.attributes('src')?.split('/').pop());
      expect(images()).toEqual(expect.arrayContaining(['face.jpg', 'choice.jpg', 'right.jpg', 'hall.jpg']));
      useSettingsStore().eventImages = false;
      await nextTick();
      expect(images()).toContain('hall.jpg');
      expect(images()).not.toContain('right.jpg');
      expect(images()).not.toContain('choice.jpg');
    } finally {
      view.unmount();
    }
  });

  it('replaces and clears Classic sprite corners, including legacy saved maps', () => {
    const require = createRequire(import.meta.url);
    new Function(readFileSync(resolve(import.meta.dirname, '../../out/html/jquery-1.11.1.min.js'), 'utf8'))();
    const BrowserUI = require('../../vendor/dendrynexus-ten/lib/ui/browser.js');
    const ui = Object.create(BrowserUI.prototype);
    ui.spriteLocs = { topLeft: 1, topRight: 1, bottomLeft: 1, bottomRight: 1 };
    document.body.innerHTML = Object.keys(ui.spriteLocs).map(loc => '<div id="' + loc + 'Sprite"></div>').join('');
    ui.show_portraits = true;
    ui.setSprites([['bottomRight', 'one.jpg']]);
    expect(document.querySelector('#bottomRightSprite img')?.getAttribute('src')).toBe('one.jpg');
    ui.setSprites({ topLeft: 'two.jpg' });
    expect(document.querySelector('#bottomRightSprite img')).toBeNull();
    expect(document.querySelector('#topLeftSprite img')?.getAttribute('src')).toBe('two.jpg');
    ui.setSprites('none');
    expect(document.querySelector('img')).toBeNull();
  });

  it.each(['a.jpg if', 'fallback.jpg; a.jpg if leader', 'a.jpg if {! return (; !}'])(
    'rejects an invalid conditional value %s', async value => {
      await expect(compile('title: Start\nface-image: ' + value + '\n\nStart.\n')).rejects.toThrow();
    },
  );
});
