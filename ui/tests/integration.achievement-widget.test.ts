import { describe, it, expect } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { mount } from '@vue/test-utils';
import { setActivePinia, createPinia } from 'pinia';
import { DendryAdapter } from '../src/engine/adapter';
import { useGameStore } from '../src/stores/game';
import Prose from '../src/components/Prose.vue';

// The Desk-ending-mounts-the-gallery question (task-8-brief ambiguity
// resolution 3): game_over.scene.dry's @achievements /
// @achievements_this_playthrough scenes are ordinary content (roles
// undefined / 'ending'), reached through GameView's PaperPage branch, which
// renders `game.frame.html` through <Prose> (PaperPage.vue) exactly like
// the desk's own dossier prose does. Prose hosts ANY `[data-widget]` marker
// generically — it does not know or care which variant (page/event/ending)
// is showing. This test proves that chain end to end against the REAL
// compiled scene, not a hand-rolled HTML fixture: if the widget marker's
// exact shape ever drifts (attribute quoting, JSON escaping,
// data-props key), this fails loudly instead of the gallery silently not
// mounting on the Desk.
const GAME_JSON = path.join(__dirname, '..', '..', 'out', 'game.json');
const HAVE_GAME = existsSync(GAME_JSON);

(HAVE_GAME ? describe : describe.skip)(
  'the Desk actually mounts AchievementGallery for game_over.scene.dry\'s real scenes',
  () => {
    const realText = HAVE_GAME ? readFileSync(GAME_JSON, 'utf8') : '';

    function mountSceneProse(sceneId: string) {
      const pinia = createPinia();
      setActivePinia(pinia);
      // Two adapters, deliberately: `game` (the store) boots the NORMAL way
      // so AchievementGallery's own `useGameStore().achievements` resolves;
      // a second, throwaway adapter reaches the specific ending/menu scene
      // directly (the game store exposes no goToScene — nothing else in
      // the Desk needs one) purely to obtain that scene's real compiled
      // `frame.html`, the exact string Prose would receive in the app.
      const game = useGameStore();
      game.initFromText(realText);
      game.newGame();
      const probe = DendryAdapter.fromJSONText(realText);
      probe.beginGame();
      const frame = probe.goToScene(sceneId);
      const wrapper = mount(Prose, { props: { html: frame.html }, global: { plugins: [pinia] } });
      return wrapper;
    }

    it('@achievements (the main gallery, scope=ever)', () => {
      const wrapper = mountSceneProse('game_over.achievements');
      expect(wrapper.find('[data-test="achievement-gallery"]').exists()).toBe(true);
      expect(wrapper.findAll('[data-test^="achievement-row-"]').length).toBe(13);
    });

    it('@achievements_this_playthrough (scope=playthrough)', () => {
      const wrapper = mountSceneProse('game_over.achievements_this_playthrough');
      expect(wrapper.find('[data-test="achievement-gallery"]').exists()).toBe(true);
      expect(wrapper.findAll('[data-test^="achievement-row-"]').length).toBe(13);
    });
  },
);
