<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import RibbonStack from '../menu/RibbonStack.vue';
import StepAsidePane from '../menu/StepAsidePane.vue';
import Prose from '../Prose.vue';
import { useGameStore } from '../../stores/game';
import { useSettingsStore } from '../../stores/settings';
import { useShellStore } from '../../stores/shell';
import { splitAuthoredPane } from '../menu/authoredPane';
import SaveLoadPane from '../menu/SaveLoadPane.vue';
import OptionsPane from '../menu/OptionsPane.vue';

const emit = defineEmits<{ closed: [] }>();
const game = useGameStore();
const shell = useShellStore();
const settings = useSettingsStore();
const { t } = useI18n();
const ribbons = ref<InstanceType<typeof RibbonStack> | null>(null);
const pane = ref<{ focusHeading: () => Promise<boolean> | boolean | void } | null>(null);
const lastRibbon = ref('resume');
const items = computed(() => [
  { id: 'resume', title: t('shell.pause.resume'), tone: 'gold' as const },
  {
    id: 'load',
    title: t('shell.pause.saveLoad'),
    disabled: game.savesDisabled,
    disabledReason: game.savesDisabled ? t('shell.saveManager.reasons.ironmanRun') : undefined,
  },
  { id: 'options', title: t('shell.pause.settings') },
  { id: 'achievements', title: t('shell.pause.achievements') },
  { id: 'exit', title: t('shell.pause.exit'), tone: 'red' as const },
]);
const visibleChoices = computed(() =>
  (game.frame?.choices ?? []).map((choice, index) => ({ choice, index }))
    .filter(({ choice }) => !choice.tags.includes('shell-return')),
);
const authoredPane = computed(() => splitAuthoredPane(game.frame?.html ?? ''));

async function focusPane(): Promise<void> { await nextTick(); pane.value?.focusHeading(); }
async function focusRibbon(): Promise<void> { await nextTick(); ribbons.value?.focusItem(lastRibbon.value); }

function restoreAchievementOrigin(): void {
  if (!shell.pauseReturnState) return;
  const state = shell.pauseReturnState;
  shell.pauseReturnState = null;
  game.restoreState(state);
}

async function resume(): Promise<void> {
  restoreAchievementOrigin();
  shell.resume();
  await nextTick();
  emit('closed');
}

async function select(id: string): Promise<void> {
  lastRibbon.value = id;
  if (id === 'resume') return resume();
  restoreAchievementOrigin();
  if (id === 'exit') {
    shell.showExitConfirmation();
    return focusPane();
  }
  if (id === 'achievements') {
    const target = shell.titleChoices.find((choice) => choice.tags.includes('shell-achievements'));
    if (!target?.canChoose) return;
    shell.pauseReturnState = game.captureState();
    game.goToScene(target.id);
    shell.openPausePane('achievements');
    return focusPane();
  }
  shell.openPausePane(id === 'load' ? 'load' : 'options');
  return focusPane();
}

async function back(): Promise<void> {
  restoreAchievementOrigin();
  shell.backToPauseMenu();
  await focusRibbon();
}

async function exitToTitle(): Promise<void> {
  if (!game.newGame()) return;
  const hub = game.roleHub('title-hub');
  if (!hub) return;
  shell.enterTitle(hub);
}

async function choose(index: number): Promise<void> {
  game.chooseFromShell(index);
  await focusPane();
}

watch(() => shell.overlay, async (value) => {
  if (value === 'pause-menu') await focusRibbon();
  else await focusPane();
}, { immediate: true });

defineExpose({ resume, back });
</script>

<template>
  <section class="pause-overlay" :class="{ 'has-pane': shell.overlay !== 'pause-menu' }" role="dialog" aria-modal="true" :aria-label="t('shell.pause.menuLabel')" data-test="pause-overlay">
    <header class="pause-heading">
      <h1>{{ t('shell.pause.paused') }}</h1>
      <p>{{ t('shell.pause.escResumes') }}</p>
    </header>
    <RibbonStack
      ref="ribbons"
      :items="items"
      :active-id="shell.overlay === 'pause-menu' ? null : lastRibbon"
      :animations="settings.animations"
      :aria-label="t('shell.pause.menuLabel')"
      @select="select"
    >
      <template #title="{ item }"><Prose tag="span" :html="item.title" /></template>
      <template #subtitle="{ item }"><Prose tag="span" :html="item.subtitle ?? ''" /></template>
    </RibbonStack>
    <SaveLoadPane
      v-if="shell.overlay === 'pause-pane' && shell.pane === 'load'"
      ref="pane"
      mode="pause"
      :animations="settings.animations"
      :close-label="t('shell.back')"
      @close="back"
      @loaded="resume"
    />
    <StepAsidePane
      v-else-if="shell.overlay === 'pause-pane' || shell.overlay === 'exit-confirm'"
      ref="pane"
      :title="shell.overlay === 'exit-confirm' ? t('shell.exit.title') : shell.pane === 'load' ? t('shell.load.pauseTitle') : shell.pane === 'options' ? t('shell.options.pauseTitle') : (game.frame?.title || t('shell.pause.achievements'))"
      :title-hidden="shell.pane === 'achievements' && !authoredPane.titleHtml"
      :animations="settings.animations"
      :close-label="t('shell.back')"
      @close="back"
    >
      <template v-if="shell.pane === 'achievements' && authoredPane.titleHtml" #title>
        <Prose tag="span" :html="authoredPane.titleHtml" />
      </template>
      <template v-if="shell.overlay === 'exit-confirm'">
        <p>{{ t('shell.exit.body') }}</p>
        <div class="pane-actions">
          <button type="button" class="danger" @click="exitToTitle">{{ t('shell.exit.confirm') }}</button>
          <button type="button" @click="back">{{ t('shell.cancel') }}</button>
        </div>
      </template>
      <OptionsPane v-if="shell.overlay !== 'exit-confirm' && shell.pane === 'options'" />
      <template v-if="shell.overlay !== 'exit-confirm' && shell.pane !== 'load' && shell.pane !== 'options'">
        <Prose :html="authoredPane.bodyHtml" />
        <div v-if="visibleChoices.length" class="authored-choices">
          <button v-for="entry in visibleChoices" :key="entry.index" type="button" :disabled="!entry.choice.canChoose" @click="choose(entry.index)">
            <Prose tag="span" :html="entry.choice.title" />
            <small v-if="entry.choice.subtitle"><Prose tag="span" :html="entry.choice.subtitle" /></small>
          </button>
        </div>
      </template>
    </StepAsidePane>
  </section>
</template>

<style scoped>
.pause-overlay { position: absolute; inset: 0; z-index: 80; display: grid; grid-template-columns: minmax(280px, 680px); justify-content: center; align-items: center; padding: clamp(24px, 5vw, 72px); overflow: hidden; }
.pause-heading { position: absolute; top: clamp(22px, 4vh, 56px); left: clamp(22px, 4vw, 72px); }
.pause-heading h1 { margin: 0; font-family: var(--font-title); font-size: clamp(1.15rem, 2vw, 1.8rem); letter-spacing: .08em; text-transform: uppercase; }
.pause-heading p { margin: 6px 0 0; color: color-mix(in srgb, var(--ink-0) 62%, transparent); font-size: .82rem; }
.pause-overlay.has-pane { grid-template-columns: minmax(260px, .72fr) minmax(420px, 1.28fr); gap: clamp(20px, 4vw, 64px); align-items: stretch; }
.pane-actions, .authored-choices { display: grid; gap: 10px; margin-top: 24px; }
.pane-actions button, .authored-choices button { padding: 12px 16px; border: 1px solid #27231c; background: #f5efd9; color: inherit; text-align: left; font: inherit; }
.pane-actions .danger { color: #8c1717; border-color: #8c1717; }
button:focus-visible { outline: 3px solid #a21f1f; outline-offset: 3px; }
.authored-choices small { display: block; opacity: .72; }
@media (max-width: 800px) { .pause-overlay.has-pane { grid-template-columns: 1fr; grid-template-rows: auto minmax(420px, 1fr); overflow: auto; } }
</style>
