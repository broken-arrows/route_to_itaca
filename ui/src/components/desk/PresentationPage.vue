<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { useGameStore } from '../../stores/game';
import { useShellStore } from '../../stores/shell';
import { useDeskStore } from '../../stores/desk';
import { useSettingsStore } from '../../stores/settings';
import Prose from '../Prose.vue';

const game = useGameStore();
const shell = useShellStore();
const desk = useDeskStore();
const settings = useSettingsStore();
const { t } = useI18n();
const region = ref<HTMLElement | null>(null);
const backgroundBroken = ref(false);
const imageBroken = ref(false);
const base = import.meta.env.BASE_URL;
const background = computed(() => game.frame?.bg);
const faceImage = computed(() => game.frame?.faceImage);
const imageContent = computed(() => game.frame?.faceImageContent ?? {
  beforeHtml: '', afterHtml: game.frame?.html ?? '',
});
const choices = computed(() => game.frame?.choices ?? []);
const hasAuthoredHeading = computed(() => /<h[1-6][\s>]/i.test(game.frame?.html ?? ''));

function focusHeading(): void {
  const heading = region.value?.querySelector<HTMLElement>('h1, h2, h3');
  heading?.setAttribute('tabindex', '-1');
  heading?.focus({ preventScroll: true });
}

watch(background, () => { backgroundBroken.value = false; });
watch(faceImage, () => { imageBroken.value = false; });
watch(() => game.frame?.html, async () => {
  await nextTick();
  if (region.value) region.value.scrollTop = 0;
  if (!shell.transitioning) focusHeading();
}, { immediate: true });
watch(() => shell.transitioning, async (busy, previous) => {
  if (previous && !busy) {
    await nextTick();
    focusHeading();
  }
});

function choose(index: number): void {
  if (!choices.value[index]?.canChoose || shell.blocksEngineChoices) return;
  try { game.choose(index); }
  catch (error) {
    console.error('[presentation] engine action failed:', error);
    desk.nudge('desk.toast.engineError');
  }
}

function returnToTitle(): void {
  if (shell.transitioning || !game.frame?.gameOver) return;
  if (!game.newGame()) return;
  const hub = game.roleHub('title-hub');
  if (hub) shell.enterTitle(hub);
}
</script>

<template>
  <section class="presentation-page" :class="`presentation-${game.effectiveRole}`" data-test="presentation-page">
    <img v-if="background && !backgroundBroken" class="presentation-background" :src="`${base}${background}`" alt="" data-test="presentation-background" @error="backgroundBroken = true" />
    <main ref="region" class="presentation-region" data-test="presentation-region">
      <article class="presentation-paper">
        <header class="presentation-heading">
          <span v-if="game.effectiveRole === 'ending'" class="ended-stamp">{{ t('desk.page.ended') }}</span>
          <h1 v-if="!hasAuthoredHeading"><Prose tag="span" :html="game.frame?.title ?? ''" /></h1>
          <span class="presentation-rule" aria-hidden="true" />
        </header>
        <div class="presentation-prose">
          <Prose v-if="imageContent.beforeHtml" :html="imageContent.beforeHtml" />
          <div class="presentation-scene-text">
            <img v-if="faceImage && settings.eventImages && !imageBroken" class="presentation-image" :src="`${base}${faceImage}`" alt="" data-test="presentation-face-image" @error="imageBroken = true" />
            <Prose :html="imageContent.afterHtml" />
          </div>
        </div>
        <nav v-if="choices.length && !game.frame?.gameOver" class="presentation-choices">
          <button v-for="(choice, index) in choices" :key="choice.id" type="button" :disabled="!choice.canChoose || shell.transitioning" :data-test="`presentation-choice-${index}`" @click="choose(index)">
            <span class="choice-title"><Prose tag="span" :html="choice.title" /></span>
            <span v-if="choice.subtitle" class="choice-subtitle"><Prose tag="span" :html="choice.subtitle" /></span>
          </button>
        </nav>
        <button v-if="game.frame?.gameOver" type="button" class="return-title" data-test="return-to-title" :disabled="shell.transitioning" @click="returnToTitle">{{ t('shell.title.return') }}</button>
      </article>
    </main>
  </section>
</template>

<style scoped>
.presentation-page { position: absolute; inset: 0; overflow: hidden; color: var(--ink-1); background: #302c25; }
.presentation-background { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; filter: brightness(.48) saturate(.65); }
.presentation-region { position: relative; height: 100%; overflow: auto; padding: 60px 88px; }
.presentation-paper { width: min(860px, 100%); min-height: 380px; margin: 0 auto; padding: 34px 48px 42px; background: #faf9f3; border: 1px solid #d7d0be; box-shadow: 0 18px 46px #0006; }
.presentation-heading { outline: none; }
.presentation-heading:focus-visible { outline: 2px solid var(--accent-slate); outline-offset: 8px; }
.presentation-heading h1, .presentation-prose :deep(h1) { font: var(--type-heading); letter-spacing: var(--type-tracking); }
.presentation-heading h1 { margin: 0 0 18px; }
.presentation-rule { display: block; height: 6px; margin-bottom: 24px; border-top: 2px solid #4f4a41; border-bottom: 1px solid #4f4a41; }
.presentation-prose { font: var(--type-reading); letter-spacing: var(--type-tracking); overflow-wrap: anywhere; }
.presentation-prose :deep(h1) { margin: 24px 0 18px; }
.presentation-prose :deep(p) { margin: 0 0 18px; }
.presentation-prose :deep(blockquote) { font: var(--type-reading); letter-spacing: var(--type-tracking); margin: 0 0 22px; padding: 0 20px; border-left: 2px solid #b7ac94; }
.presentation-prose :deep(img) { max-width: 100%; height: auto; }
.presentation-scene-text { display: flow-root; }
.presentation-image { float: right; width: 120px; max-height: 120px; object-fit: contain; margin: 0 0 16px 24px; }
.presentation-choices { margin-top: 28px; border-top: 1px solid #585248; }
.presentation-choices button { display: block; width: 100%; padding: 16px 8px; border: 0; border-bottom: 1px solid #d9d1bd; background: transparent; color: inherit; text-align: left; cursor: pointer; }
.choice-title { font: var(--type-control); letter-spacing: var(--type-tracking); display: block; }
.choice-subtitle { font: var(--type-compact); letter-spacing: var(--type-tracking); display: block; margin-top: 6px; }
.presentation-choices button:hover:not(:disabled), .presentation-choices button:focus-visible { background: #eee9db; outline: 2px solid var(--accent-slate); outline-offset: -2px; }
.presentation-choices button:disabled { opacity: .5; cursor: not-allowed; }
.ended-stamp { font: var(--type-compact); letter-spacing: var(--type-tracking); display: inline-block; margin-bottom: 18px; padding: 5px 12px; border: 2px solid #4a5b6a; color: #4a5b6a; }
.return-title { font: var(--type-control); letter-spacing: var(--type-tracking); margin-top: 24px; padding: 14px 22px; border: 1px solid #585248; background: #eee9db; color: inherit; cursor: pointer; }
.return-title:focus-visible { outline: 3px solid var(--accent-slate); outline-offset: 3px; }
@media (max-height: 760px) { .presentation-region { padding: 42px 76px; } .presentation-paper { padding: 28px 40px 34px; } }
</style>
