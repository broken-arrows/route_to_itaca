<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue';
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

const game = useGameStore();
const shell = useShellStore();
const settings = useSettingsStore();
const { t } = useI18n();
const ribbons = ref<InstanceType<typeof RibbonStack> | null>(null);
const pane = ref<{ focusHeading: () => Promise<boolean> | boolean | void } | null>(null);
const pendingRiskyContinue = ref(false);
const lastRibbon = ref<string | null>(null);

const slots = computed(() => game.listSlots());
const auto = computed(() => slots.value.find((entry) => entry.slot === 'auto-1' && entry.status === 'ready'));
const authored = computed(() => shell.titleChoices);
const activeAuthoredId = computed(() => {
  const id = game.frame?.sceneId;
  return authored.value.find((choice) => choice.id === id)?.id ?? null;
});
const activeId = computed(() => {
  if (shell.pane === 'load') return 'host-load';
  if (shell.pane === 'options') return 'host-options';
  if (activeAuthoredId.value) return `authored:${activeAuthoredId.value}`;
  return shell.pane === 'authored' ? lastRibbon.value : null;
});
const gameTitle = computed(() => {
  const [name, ...rest] = (game.info?.title ?? '').split(/\s+-\s+/);
  return { name, subtitle: rest.join(' - ') };
});
const items = computed(() => [
  ...(auto.value
    ? [{
        id: 'host-continue',
        title: t('shell.title.continue'),
        subtitle: auto.value.savedAt ? new Date(auto.value.savedAt).toLocaleString() : undefined,
        tone: 'gold' as const,
      }]
    : []),
  {
    id: 'host-load',
    title: t('shell.title.load'),
    subtitle: t('shell.title.saveCount', { count: slots.value.filter((entry) => entry.status === 'ready').length }),
  },
  ...authored.value.map((choice) => ({
    id: `authored:${choice.id}`,
    title: choice.title,
    subtitle: choice.subtitle,
    disabled: !choice.canChoose,
    disabledReason: choice.subtitle,
  })),
  { id: 'host-options', title: t('shell.title.options') },
]);

const visibleChoices = computed(() =>
  (game.frame?.choices ?? []).map((choice, index) => ({ choice, index }))
    .filter(({ choice }) => !choice.tags.includes('shell-return')),
);
const authoredPane = computed(() => splitAuthoredPane(game.frame?.html ?? ''));

async function focusPane(): Promise<void> {
  await nextTick();
  pane.value?.focusHeading();
}

function refreshHub(): void {
  const hub = game.roleHub('title-hub');
  if (hub) shell.refreshTitleHub(hub);
}

async function continueGame(allowRisk = false): Promise<void> {
  const result = game.loadSlot('auto-1', allowRisk);
  if (result.status === 'loaded') {
    pendingRiskyContinue.value = false;
    shell.beginPlaying();
    return;
  }
  if (result.status === 'confirmation-required') {
    pendingRiskyContinue.value = true;
    shell.openTitlePane('load');
    await focusPane();
  }
}

async function select(id: string): Promise<void> {
  lastRibbon.value = id;
  if (id === 'host-continue') return continueGame();
  pendingRiskyContinue.value = false;
  if (id === 'host-load') {
    shell.openTitlePane('load');
    return focusPane();
  }
  if (id === 'host-options') {
    shell.openTitlePane('options');
    return focusPane();
  }
  const target = authored.value.find((choice) => `authored:${choice.id}` === id);
  if (!target?.canChoose) return;
  if (target.tags.includes('starts-game')) {
    if (!game.newGame()) return;
    const hub = game.roleHub('title-hub');
    const start = hub?.choices.find((choice) => choice.tags.includes('starts-game'));
    if (!hub || !start?.canChoose) return;
    shell.refreshTitleHub(hub);
    shell.beginPlaying();
    game.goToScene(start.id);
    return;
  }
  game.goToScene(target.id);
  shell.openTitlePane('authored');
  await focusPane();
}

async function choose(index: number): Promise<void> {
  game.chooseFromShell(index);
  const hub = game.roleHub('title-hub');
  if (game.effectiveRole === 'title-hub' && hub) {
    shell.enterTitle(hub);
    await nextTick();
    await ribbons.value?.focusItem(lastRibbon.value);
  } else {
    await focusPane();
  }
}

async function closePane(): Promise<void> {
  pendingRiskyContinue.value = false;
  if (shell.titleHubId) game.goToScene(shell.titleHubId);
  refreshHub();
  shell.overlay = 'closed';
  shell.pane = null;
  await nextTick();
  await ribbons.value?.focusItem(lastRibbon.value);
}

watch(() => game.frame, refreshHub, { flush: 'sync' });
onMounted(async () => { await nextTick(); await ribbons.value?.focusFirst(); });
</script>

<template>
  <section class="title-shell" :class="{ 'has-pane': shell.overlay === 'title-pane' }" data-test="title-shell">
    <div v-if="shell.overlay !== 'title-pane'" class="title-identity">
      <p v-if="gameTitle.subtitle" class="title-kicker">{{ gameTitle.subtitle }}</p>
      <h1>{{ gameTitle.name }}</h1>
      <span class="title-rule" aria-hidden="true" />
    </div>
    <RibbonStack
      ref="ribbons"
      :items="items"
      :active-id="activeId"
      :animations="settings.animations"
      :aria-label="t('shell.title.menuLabel')"
      @select="select"
    >
      <template #title="{ item }"><Prose tag="span" :html="item.title" /></template>
      <template #subtitle="{ item }"><Prose tag="span" :html="item.subtitle ?? ''" /></template>
    </RibbonStack>
    <SaveLoadPane
      v-if="shell.overlay === 'title-pane' && shell.pane === 'load' && !pendingRiskyContinue"
      ref="pane"
      mode="title"
      :animations="settings.animations"
      :close-label="t('shell.close')"
      @close="closePane"
      @loaded="shell.beginPlaying()"
    />
    <StepAsidePane
      v-else-if="shell.overlay === 'title-pane'"
      ref="pane"
      :title="pendingRiskyContinue ? t('shell.continueWarning.title') : shell.pane === 'load' ? t('shell.load.title') : shell.pane === 'options' ? t('shell.options.title') : (game.frame?.title || '')"
      :title-hidden="shell.pane === 'authored' && !authoredPane.titleHtml"
      :animations="settings.animations"
      :close-label="t('shell.close')"
      @close="closePane"
    >
      <template v-if="shell.pane === 'authored' && authoredPane.titleHtml" #title>
        <Prose tag="span" :html="authoredPane.titleHtml" />
      </template>
      <template v-if="pendingRiskyContinue">
        <p>{{ t('shell.continueWarning.body') }}</p>
        <div class="pane-actions">
          <button type="button" @click="continueGame(true)">{{ t('shell.continueWarning.confirm') }}</button>
          <button type="button" @click="closePane">{{ t('shell.cancel') }}</button>
        </div>
      </template>
      <OptionsPane v-if="!pendingRiskyContinue && shell.pane === 'options'" />
      <template v-if="!pendingRiskyContinue && shell.pane !== 'load' && shell.pane !== 'options'">
        <Prose :html="authoredPane.bodyHtml" />
        <div v-if="visibleChoices.length" class="authored-choices">
          <button
            v-for="entry in visibleChoices"
            :key="`${game.frame?.sceneId}:${entry.index}`"
            type="button"
            :disabled="!entry.choice.canChoose"
            @click="choose(entry.index)"
          >
            <Prose tag="span" :html="entry.choice.title" />
            <small v-if="entry.choice.subtitle"><Prose tag="span" :html="entry.choice.subtitle" /></small>
          </button>
        </div>
      </template>
    </StepAsidePane>
  </section>
</template>

<style scoped>
.title-shell { position: relative; height: 100%; min-height: 0; display: grid; grid-template-columns: minmax(300px, .9fr) minmax(460px, 1.1fr); grid-template-rows: minmax(0, 1fr); grid-template-areas: 'pane ribbons'; gap: clamp(28px, 5vw, 84px); align-items: center; padding: clamp(24px, 5vw, 72px); overflow: hidden; }
.title-shell.has-pane { grid-template-columns: minmax(420px, 1.28fr) minmax(260px, .72fr); align-items: stretch; }
.title-shell :deep(.ribbon-stack) { grid-area: ribbons; align-self: center; }
.title-shell :deep(.step-aside-pane), .title-identity { grid-area: pane; }
.title-identity { align-self: center; min-width: 0; }
.title-kicker { margin: 0 0 18px; color: color-mix(in srgb, var(--accent-gold) 72%, var(--ink-0)); font-size: clamp(.7rem, .9vw, .92rem); font-weight: 700; letter-spacing: .28em; text-transform: uppercase; }
.title-identity h1 { max-width: 10ch; margin: 0; font-family: var(--font-title); font-size: clamp(3.5rem, 7.5vw, 8rem); font-weight: 800; letter-spacing: -.045em; line-height: .92; text-transform: uppercase; }
.title-rule { display: block; width: min(70%, 460px); margin-top: 44px; border-top: 2px solid var(--accent-red); }
.authored-choices, .pane-actions { display: grid; gap: 10px; margin-top: 24px; }
.authored-choices button, .pane-actions button { padding: 12px 16px; border: 1px solid var(--ink, #27231c); background: #f5efd9; color: inherit; text-align: left; font: inherit; }
.authored-choices button:focus-visible, .pane-actions button:focus-visible { outline: 3px solid #a21f1f; outline-offset: 3px; }
.authored-choices small { display: block; margin-top: 3px; opacity: .72; }
@media (max-width: 800px) { .title-shell, .title-shell.has-pane { grid-template-columns: 1fr; grid-template-areas: 'pane' 'ribbons'; grid-template-rows: auto auto; overflow: auto; } .title-identity h1 { font-size: clamp(3rem, 14vw, 5rem); } }
</style>
