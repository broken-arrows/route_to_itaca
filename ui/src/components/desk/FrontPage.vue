<script setup lang="ts">
// A generic event document. Prose, images, widget markers and options all
// arrive through the ordinary frame contract; this surface knows nothing
// about event ids or stages and retains no record of earlier answers.
import { computed, ref, watch } from 'vue';
import { useGameStore } from '../../stores/game';
import { useDeskStore } from '../../stores/desk';
import { useSettingsStore } from '../../stores/settings';
import type { ChoiceView } from '../../engine/types';
import Clipboard from '../brief/Clipboard.vue';
import Prose from '../Prose.vue';

const game = useGameStore();
const desk = useDeskStore();
const settings = useSettingsStore();
const choices = computed<ChoiceView[]>(() => game.frame?.choices ?? []);
const assetBase = import.meta.env.BASE_URL;
const imageBroken = ref(false);
watch(
  () => game.frame?.sceneId,
  () => { imageBroken.value = false; },
);

function choose(index: number, choice: ChoiceView): void {
  if (choice.canChoose) desk.chooseEventChoice(index);
}
</script>

<template>
  <div class="news-desk" data-test="front-page">
    <Clipboard />
    <main class="news-region">
      <article class="front-page-sheet">
        <header class="paper-header">
          <span class="red-rule" aria-hidden="true"></span>
        </header>
        <img
          v-if="settings.eventImages && game.frame?.faceImage && !imageBroken"
          class="event-image"
          :src="`${assetBase}${game.frame.faceImage}`"
          alt=""
          @error="imageBroken = true"
        />
        <Prose class="event-prose" :html="game.frame?.html ?? ''" />
        <section v-if="choices.length" class="event-options" aria-label="Choices">
          <article
            v-for="(choice, index) in choices"
            :key="choice.id"
            class="event-choice"
            :class="{ locked: !choice.canChoose }"
            :data-test="`event-choice-${index}`"
            role="button"
            :aria-disabled="choice.canChoose ? 'false' : 'true'"
            :tabindex="choice.canChoose ? 0 : -1"
            @click="choose(index, choice)"
            @keydown.enter.prevent="choose(index, choice)"
            @keydown.space.prevent="choose(index, choice)"
          >
            <h2><Prose tag="span" :html="choice.title" /></h2>
            <p v-if="choice.subtitle"><Prose tag="span" :html="choice.subtitle" /></p>
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
  padding: clamp(28px, 4vh, 48px) clamp(28px, 4vw, 64px);
  background:
    repeating-linear-gradient(0deg, rgba(90, 70, 40, 0.025) 0 1px, transparent 1px 6px),
    radial-gradient(110% 100% at 50% 0%, #e2d9c4, #d2c7ac);
}
.front-page-sheet {
  width: min(1080px, 100%);
  margin: 0 auto;
  padding: clamp(28px, 3.5vw, 52px);
  color: var(--ink-1);
  background: #faf9f3;
  border: 1px solid #d7d0be;
  box-shadow: 0 12px 26px rgba(56, 45, 25, 0.2);
}
.paper-header {
  padding-bottom: 9px;
  border-bottom: 1px solid #514c43;
}
.red-rule { height: 3px; background: var(--accent-red); }
.event-prose {
  margin-top: 22px;
  font-family: var(--font-news);
  font-size: clamp(16px, 1.3vw, 19px);
  line-height: 1.55;
}
.event-image {
  float: right;
  width: min(36%, 340px);
  max-height: 250px;
  margin: 22px 0 18px 30px;
  object-fit: contain;
  border: 1px solid #d9d0bb;
  filter: saturate(0.82) contrast(0.96);
}
.event-prose :deep(h1) {
  margin: 8px 0 20px;
  font-family: var(--font-news);
  font-size: clamp(34px, 4vw, 54px);
  line-height: 1.02;
}
.event-prose :deep(h2), .event-prose :deep(h3) { font-family: var(--font-news); }
.event-prose :deep(img) { max-width: 100%; height: auto; }
.event-options {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr));
  gap: 14px;
  clear: both;
  margin-top: 24px;
  padding-top: 14px;
  border-top: 3px double #aaa18d;
}
.event-choice {
  min-height: 82px;
  padding: 16px 18px;
  background: rgba(255, 255, 255, 0.52);
  border: 1px solid #d9d0bb;
  cursor: pointer;
  transition: border-color 120ms ease, box-shadow 120ms ease, translate 120ms ease;
}
.event-choice:not(.locked):hover,
.event-choice:not(.locked):focus-visible {
  border-color: var(--accent-red);
  box-shadow: 0 7px 16px rgba(70, 45, 30, 0.14);
  translate: 0 -3px;
  outline: none;
}
.event-choice.locked { cursor: not-allowed; opacity: 0.4; }
.event-choice h2 {
  margin: 0;
  font-family: var(--font-news);
  font-size: 22px;
  line-height: 1.1;
}
.event-choice p {
  margin: 9px 0 0;
  color: #686156;
  font-family: var(--font-news);
  font-size: 14px;
  font-style: italic;
  line-height: 1.4;
}
@media (max-width: 920px) {
  .news-desk { grid-template-columns: minmax(330px, 42vw) minmax(0, 1fr); }
  .news-region { padding: 20px 14px; }
}
</style>
