<script setup lang="ts">
// Newspaper is a presentation of the engine's existing newspaper choice list.
// It deliberately owns no queue, ordering, cap, filler, or transition logic:
// the live frame remains authoritative and each headline selects the ordinary
// choice at the same index.
import { computed } from 'vue';
import { useGameStore } from '../../stores/game';
import { useDeskStore } from '../../stores/desk';
import type { ChoiceView } from '../../engine/types';
import Clipboard from '../brief/Clipboard.vue';
import Prose from '../Prose.vue';

const game = useGameStore();
const desk = useDeskStore();
const choices = computed<ChoiceView[]>(() => game.frame?.choices ?? []);

function choose(index: number, choice: ChoiceView): void {
  if (choice.canChoose) desk.chooseNewspaperStory(index);
}
</script>

<template>
  <div class="news-desk" data-test="newspaper">
    <Clipboard />
    <main class="news-region">
      <article class="newspaper-sheet">
        <header class="masthead">
          <div class="edition-rule" aria-hidden="true"></div>
          <h1><Prose tag="span" :html="game.frame?.title ?? ''" /></h1>
          <!-- Edition context (currently the authored month/year heading) is
               ordinary scene prose; the masthead itself comes from title:. -->
          <Prose class="newspaper-prose" :html="game.frame?.html ?? ''" />
          <div class="masthead-rule" aria-hidden="true"></div>
        </header>

        <section class="story-list" aria-label="News stories">
          <article
            v-for="(choice, index) in choices"
            :key="choice.id"
            class="story"
            :class="{ locked: !choice.canChoose }"
            :data-test="`newspaper-story-${index}`"
            role="button"
            :aria-disabled="choice.canChoose ? 'false' : 'true'"
            :tabindex="choice.canChoose ? 0 : -1"
            @click="choose(index, choice)"
            @keydown.enter.prevent="choose(index, choice)"
            @keydown.space.prevent="choose(index, choice)"
          >
            <span class="story-signal" aria-hidden="true">!</span>
            <div class="story-copy">
              <h2><Prose tag="span" :html="choice.title" /></h2>
              <p v-if="choice.subtitle"><Prose tag="span" :html="choice.subtitle" /></p>
            </div>
          </article>
        </section>
      </article>
    </main>
  </div>
</template>

<style scoped>
.news-desk {
  display: grid;
  grid-template-columns: clamp(var(--brief-min), var(--brief-fluid), var(--brief-max)) minmax(0, 1fr);
  width: 100%;
  height: 100%;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}
.news-region {
  min-width: 0;
  min-height: 0;
  overflow-y: auto;
  padding: clamp(42px, 8vh, 88px) clamp(34px, 6vw, 108px);
  background:
    repeating-linear-gradient(0deg, rgba(90, 70, 40, 0.025) 0 1px, transparent 1px 6px),
    radial-gradient(110% 100% at 50% 0%, #e2d9c4, #d2c7ac);
}
.newspaper-sheet {
  width: min(920px, 100%);
  margin: 0 auto;
  padding: clamp(26px, 3vw, 46px);
  color: var(--ink-1);
  background: #faf9f3;
  border: 1px solid #d7d0be;
  box-shadow: 0 14px 28px rgba(56, 45, 25, 0.22);
  transform: rotate(0.45deg);
}
.edition-rule { border-top: 1px solid #585248; }
.masthead-rule {
  height: 5px;
  margin-top: 14px;
  border-top: 2px solid #4f4a41;
  border-bottom: 1px solid #4f4a41;
}
.newspaper-prose {
  font-family: var(--font-news);
  text-align: center;
}
.masthead > h1 {
  margin: 14px 0 8px;
  font-family: var(--font-news);
  font-size: clamp(34px, 4vw, 54px);
  line-height: 1;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.newspaper-prose :deep(h1) {
  margin: 5px 0;
  font-family: var(--font-news);
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.newspaper-prose :deep(p) { margin: 6px 0; }
.story-list { margin-top: 16px; }
.story {
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr);
  gap: 14px;
  align-items: start;
  padding: 18px 4px;
  border-bottom: 1px solid #d9d1bd;
  cursor: pointer;
  transition: background 120ms ease, translate 120ms ease;
}
.story:not(.locked):hover,
.story:not(.locked):focus-visible {
  background: rgba(140, 29, 29, 0.045);
  translate: 4px 0;
  outline: none;
}
.story.locked { cursor: not-allowed; opacity: 0.42; }
.story-signal {
  display: grid;
  place-items: center;
  width: 24px;
  height: 24px;
  margin-top: 3px;
  border-radius: 50%;
  color: #fff;
  background: var(--accent-red);
  font-family: var(--font-title);
  font-size: 14px;
  font-weight: 800;
}
.story h2 {
  margin: 0;
  font-family: var(--font-news);
  font-size: clamp(22px, 2vw, 31px);
  line-height: 1.05;
}
.story p {
  margin: 6px 0 0;
  color: #686156;
  font-family: var(--font-news);
  font-size: 15px;
  font-style: italic;
  line-height: 1.35;
}
@media (max-width: 920px) {
  .news-desk { grid-template-columns: minmax(330px, 42vw) minmax(0, 1fr); }
  .news-region { padding: 30px 20px; }
}
</style>
