<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import type { SaveSlotEntry } from '../../engine/types';
import { useGameStore, type LoadSlotResult } from '../../stores/game';
import StepAsidePane from './StepAsidePane.vue';

type PaneMode = 'title' | 'pause';
type OperationError = { code: string; message?: string };
type SlotOperationResult = {
  ok: boolean;
  slot?: string;
  status?: string;
  error?: OperationError;
};

interface SaveManagementApi {
  savesDisabled: boolean;
  createManualSave: () => SlotOperationResult;
  overwriteManualSave: (slot: string, confirmed?: boolean) => SlotOperationResult;
  importManualSave: (serialized: string) => SlotOperationResult;
}

const props = withDefaults(defineProps<{
  mode: PaneMode;
  animations?: boolean;
  closeLabel?: string;
}>(), {
  animations: true,
  closeLabel: undefined,
});

const emit = defineEmits<{
  close: [];
  loaded: [slot: string];
  exported: [slot: string, serialized: string];
}>();

const game = useGameStore();
// Phase 5B extends the store with policy-aware manual operations. Keeping this
// cast local documents the component boundary while the store remains the sole
// owner of allocation, persistence validation, and ironman policy.
const management = game as typeof game & SaveManagementApi;
const { t, locale } = useI18n();
const pane = ref<InstanceType<typeof StepAsidePane> | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);
const selectedSlot = ref<string | null>(null);
const revision = ref(0);
const errorCode = ref<string | null>(null);
const confirmation = ref<{ kind: 'load' | 'overwrite'; slot: string } | null>(null);
const rowElements = new Map<string, HTMLButtonElement>();

const entries = computed(() => {
  void revision.value;
  const supplied = game.listSlots();
  const auto1 = supplied.find((entry) => entry.slot === 'auto-1');
  const auto2 = supplied.find((entry) => entry.slot === 'auto-2');
  return [auto1, auto2, ...supplied.filter((entry) => entry.slot !== 'auto-1' && entry.slot !== 'auto-2')]
    .filter((entry): entry is SaveSlotEntry => Boolean(entry));
});
const selected = computed(() => entries.value.find((entry) => entry.slot === selectedSlot.value) ?? null);
const isPausedIronman = computed(() => props.mode === 'pause' && management.savesDisabled);
const selectedIsAuto = computed(() => selected.value?.slot === 'auto-1' || selected.value?.slot === 'auto-2');
const selectedIsManual = computed(() => /^manual-\d+$/.test(selected.value?.slot ?? ''));

const loadDisabledReason = computed(() => {
  if (!selected.value) return t('shell.saveManager.reasons.select');
  if (isPausedIronman.value) return t('shell.saveManager.reasons.ironmanLoad');
  if (selected.value.status !== 'ready') return t('shell.saveManager.reasons.notLoadable');
  return '';
});
const exportDisabledReason = computed(() => {
  if (!selected.value) return t('shell.saveManager.reasons.select');
  if (selected.value.status === 'unreadable') return t('shell.saveManager.reasons.notExportable');
  return '';
});
const deleteDisabledReason = computed(() => {
  if (!selected.value) return t('shell.saveManager.reasons.select');
  if (selectedIsAuto.value) return t('shell.saveManager.reasons.autoProtected');
  return '';
});
const overwriteDisabledReason = computed(() => {
  if (!selected.value) return t('shell.saveManager.reasons.select');
  if (isPausedIronman.value) return t('shell.saveManager.reasons.ironmanSave');
  if (!selectedIsManual.value) return t('shell.saveManager.reasons.manualOnly');
  return '';
});
const actionReasons = computed(() => [...new Set([
  loadDisabledReason.value,
  exportDisabledReason.value,
  deleteDisabledReason.value,
  props.mode === 'pause' ? overwriteDisabledReason.value : '',
  isPausedIronman.value ? t('shell.saveManager.reasons.ironmanSave') : '',
].filter(Boolean))].join(' '));

watch(entries, (current) => {
  if (!current.some((entry) => entry.slot === selectedSlot.value)) {
    selectedSlot.value = current[0]?.slot ?? null;
  }
}, { immediate: true });

function setRowElement(slot: string, element: Element | null): void {
  if (element instanceof HTMLButtonElement) rowElements.set(slot, element);
  else rowElements.delete(slot);
}

function refresh(): void {
  revision.value++;
}

async function selectAndFocus(slot: string): Promise<void> {
  selectedSlot.value = slot;
  await nextTick();
  const row = rowElements.get(slot);
  row?.focus();
  row?.scrollIntoView?.({ block: 'nearest' });
}

function operationError(result: SlotOperationResult | undefined): string | null {
  if (result?.status === 'confirmation-required') return null;
  return result && !result.ok ? result.error?.code ?? 'operation-failed' : null;
}

async function acceptCreated(result: SlotOperationResult): Promise<void> {
  errorCode.value = operationError(result);
  if (!result.ok || !result.slot) return;
  refresh();
  await selectAndFocus(result.slot);
}

async function createSave(): Promise<void> {
  if (props.mode !== 'pause' || isPausedIronman.value) return;
  await acceptCreated(management.createManualSave());
}

async function requestOverwrite(): Promise<void> {
  if (props.mode !== 'pause' || overwriteDisabledReason.value || !selected.value) return;
  const result: SlotOperationResult = management.overwriteManualSave(selected.value.slot, false);
  errorCode.value = operationError(result);
  if (result.status === 'confirmation-required') {
    confirmation.value = { kind: 'overwrite', slot: selected.value.slot };
  } else if (result.ok) {
    refresh();
    await selectAndFocus(selected.value.slot);
  }
}

async function confirmOverwrite(): Promise<void> {
  const pending = confirmation.value;
  if (!pending || pending.kind !== 'overwrite') return;
  const result = management.overwriteManualSave(pending.slot, true);
  confirmation.value = null;
  errorCode.value = operationError(result);
  if (result.ok) {
    refresh();
    await selectAndFocus(pending.slot);
  }
}

function loadSelected(allowRisk = false): void {
  if (loadDisabledReason.value || !selected.value) return;
  const slot = selected.value.slot;
  const result: LoadSlotResult = game.loadSlot(slot, allowRisk);
  if (result.status === 'confirmation-required') {
    confirmation.value = { kind: 'load', slot };
    return;
  }
  if (result.status === 'loaded') {
    confirmation.value = null;
    errorCode.value = null;
    emit('loaded', slot);
    return;
  }
  errorCode.value = result.error?.code ?? result.status;
}

function confirmLoad(): void {
  const pending = confirmation.value;
  if (!pending || pending.kind !== 'load') return;
  selectedSlot.value = pending.slot;
  loadSelected(true);
}

function exportSelected(): void {
  if (exportDisabledReason.value || !selected.value) return;
  const slot = selected.value.slot;
  const result = game.exportSlot(slot);
  if (!result.ok) {
    errorCode.value = result.error.code;
    return;
  }
  errorCode.value = null;
  emit('exported', slot, result.data);
  if (typeof URL.createObjectURL !== 'function') return;
  const url = URL.createObjectURL(new Blob([result.data], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `route-to-itaca-${slot}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

async function deleteSelected(): Promise<void> {
  if (deleteDisabledReason.value || !selected.value) return;
  const result = game.removeSlot(selected.value.slot);
  if (!result.ok) {
    errorCode.value = result.error.code;
    return;
  }
  errorCode.value = null;
  refresh();
  await nextTick();
}

async function importSerialized(serialized: string): Promise<void> {
  if (isPausedIronman.value) return;
  await acceptCreated(management.importManualSave(serialized));
}

async function onFileSelected(event: Event): Promise<void> {
  const input = event.currentTarget as HTMLInputElement;
  const file = input.files?.[0];
  input.value = '';
  if (!file) return;
  try {
    await importSerialized(await file.text());
  } catch {
    errorCode.value = 'file-read-failed';
  }
}

function rowTitle(entry: SaveSlotEntry): string {
  if (entry.slot === 'auto-1') return t('shell.saveManager.autoLatest');
  if (entry.slot === 'auto-2') return t('shell.saveManager.autoPrevious');
  const match = /^manual-(\d+)$/.exec(entry.slot);
  return match ? t('shell.saveManager.manual', { number: match[1] }) : entry.slot;
}

function formatDate(value?: string): string {
  if (!value) return t('shell.saveManager.unknownDate');
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf())
    ? t('shell.saveManager.unknownDate')
    : new Intl.DateTimeFormat(locale.value, { dateStyle: 'medium', timeStyle: 'short' }).format(parsed);
}

function statusLabel(entry: SaveSlotEntry): string {
  if (entry.status !== 'ready') return t(`shell.saveManager.status.${entry.status}`);
  if (entry.compatibility && entry.compatibility !== 'compatible') {
    return t(`shell.saveManager.compatibility.${entry.compatibility}`);
  }
  return t('shell.saveManager.status.ready');
}

defineExpose({
  focusHeading: () => pane.value?.focusHeading(),
  importSerialized,
  selectAndFocus,
});
</script>

<template>
  <StepAsidePane
    ref="pane"
    :title="mode === 'pause' ? t('shell.load.pauseTitle') : t('shell.load.title')"
    :animations="animations"
    :close-label="closeLabel ?? t('shell.close')"
    @close="emit('close')"
  >
    <div class="save-manager" data-test="save-load-pane">
      <p v-if="isPausedIronman" class="policy-note" data-test="ironman-reason">
        {{ t('shell.saveManager.reasons.ironmanRun') }}
      </p>

      <div v-if="entries.length" class="save-list" role="listbox" :aria-label="t('shell.saveManager.listLabel')">
        <button
          v-for="entry in entries"
          :key="entry.slot"
          :ref="(element) => setRowElement(entry.slot, element as Element | null)"
          type="button"
          class="save-row"
          :class="{ selected: entry.slot === selectedSlot, damaged: entry.status !== 'ready' }"
          role="option"
          :aria-selected="entry.slot === selectedSlot"
          :data-test="`save-row-${entry.slot}`"
          @click="selectedSlot = entry.slot"
        >
          <span class="row-heading">
            <strong>{{ rowTitle(entry) }}</strong>
            <small>{{ statusLabel(entry) }}</small>
          </span>
          <time v-if="entry.savedAt" :datetime="entry.savedAt">{{ formatDate(entry.savedAt) }}</time>
          <span v-else class="row-date">{{ formatDate() }}</span>
          <span v-if="entry.year !== undefined || entry.month !== undefined" class="row-facts">
            {{ t('shell.saveManager.gameDate', { year: entry.year ?? '—', month: entry.month ?? '—' }) }}
            <template v-if="entry.playerParty"> · {{ entry.playerParty }}</template>
            <template v-if="entry.resources !== null && entry.resources !== undefined"> · {{ t('shell.saveManager.resources', { count: entry.resources }) }}</template>
          </span>
          <code>{{ entry.slot }}</code>
        </button>
      </div>
      <p v-else class="empty-state">{{ t('shell.saveManager.empty') }}</p>

      <div class="manager-actions">
        <button type="button" :disabled="Boolean(loadDisabledReason)" :aria-describedby="loadDisabledReason ? 'save-action-reasons' : undefined" data-test="save-load" @click="loadSelected()">
          {{ t('shell.saveManager.actions.load') }}
        </button>
        <button type="button" :disabled="Boolean(exportDisabledReason)" :aria-describedby="exportDisabledReason ? 'save-action-reasons' : undefined" data-test="save-export" @click="exportSelected">
          {{ t('shell.saveManager.actions.export') }}
        </button>
        <button type="button" :disabled="Boolean(deleteDisabledReason)" :aria-describedby="deleteDisabledReason ? 'save-action-reasons' : undefined" class="danger" data-test="save-delete" @click="deleteSelected">
          {{ t('shell.saveManager.actions.delete') }}
        </button>
        <button type="button" :disabled="isPausedIronman" :aria-describedby="isPausedIronman ? 'save-action-reasons' : undefined" data-test="save-import" @click="fileInput?.click()">
          {{ t('shell.saveManager.actions.import') }}
        </button>
        <template v-if="mode === 'pause'">
          <button type="button" :disabled="isPausedIronman" :aria-describedby="isPausedIronman ? 'save-action-reasons' : undefined" class="primary" data-test="save-new" @click="createSave">
            {{ t('shell.saveManager.actions.new') }}
          </button>
          <button type="button" :disabled="Boolean(overwriteDisabledReason)" :aria-describedby="overwriteDisabledReason ? 'save-action-reasons' : undefined" data-test="save-overwrite" @click="requestOverwrite">
            {{ t('shell.saveManager.actions.overwrite') }}
          </button>
        </template>
      </div>
      <p id="save-action-reasons" class="action-reason" aria-live="polite">
        {{ actionReasons }}
      </p>
      <p v-if="errorCode" class="operation-error" role="alert">
        {{ t('shell.saveManager.error', { code: errorCode }) }}
      </p>
      <input ref="fileInput" class="file-input" type="file" accept="application/json,.json" data-test="save-file-input" @change="onFileSelected">

      <div v-if="confirmation" class="confirmation" role="alertdialog" aria-modal="true" :aria-labelledby="'save-confirm-title'">
        <h2 id="save-confirm-title">
          {{ confirmation.kind === 'overwrite' ? t('shell.saveManager.confirm.overwriteTitle') : t('shell.saveManager.confirm.loadTitle') }}
        </h2>
        <p>
          {{ confirmation.kind === 'overwrite' ? t('shell.saveManager.confirm.overwriteBody') : t('shell.saveManager.confirm.loadBody') }}
        </p>
        <div class="confirmation-actions">
          <button v-if="confirmation.kind === 'overwrite'" type="button" class="danger" data-test="confirm-overwrite" @click="confirmOverwrite">
            {{ t('shell.saveManager.actions.overwrite') }}
          </button>
          <button v-else type="button" class="danger" data-test="confirm-load" @click="confirmLoad">
            {{ t('shell.saveManager.confirm.loadAnyway') }}
          </button>
          <button type="button" @click="confirmation = null">{{ t('shell.cancel') }}</button>
        </div>
      </div>
    </div>
  </StepAsidePane>
</template>

<style scoped>
.save-manager { position: relative; display: flex; flex-direction: column; min-height: 100%; gap: 18px; }
.policy-note, .operation-error { margin: 0; padding: 11px 14px; border-left: 4px solid var(--accent-red); background: color-mix(in srgb, var(--accent-red) 9%, transparent); }
.save-list { display: grid; gap: 9px; }
.save-row { display: grid; grid-template-columns: minmax(140px, 1fr) auto; gap: 5px 18px; width: 100%; padding: 13px 15px; border: 1px solid color-mix(in srgb, var(--ink-0) 32%, transparent); background: color-mix(in srgb, var(--paper-1) 72%, transparent); color: inherit; font: inherit; text-align: left; cursor: pointer; }
.save-row:hover { background: var(--paper-1); }
.save-row.selected { border-color: var(--accent-gold); box-shadow: inset 5px 0 0 var(--accent-gold); }
.save-row.damaged { border-color: color-mix(in srgb, var(--accent-red) 55%, transparent); }
.save-row:focus-visible, .manager-actions button:focus-visible, .confirmation button:focus-visible { outline: 3px solid var(--accent-red); outline-offset: 3px; }
.row-heading { display: flex; align-items: baseline; gap: 10px; min-width: 0; }
.row-heading strong { font-family: var(--font-title); font-size: 1rem; text-transform: uppercase; }
.row-heading small { color: var(--accent-red); font-weight: 700; }
.save-row time, .row-date { font-size: .85rem; text-align: right; }
.row-facts { color: color-mix(in srgb, var(--ink-0) 72%, transparent); font-size: .82rem; }
.save-row code { color: color-mix(in srgb, var(--ink-0) 55%, transparent); font-size: .72rem; text-align: right; }
.empty-state { margin: auto 0; padding: 48px 20px; border-block: 1px solid color-mix(in srgb, var(--ink-0) 20%, transparent); text-align: center; }
.manager-actions { display: flex; flex-wrap: wrap; gap: 8px; padding-top: 4px; border-top: 2px solid var(--ink-0); }
.manager-actions button, .confirmation-actions button { min-height: 40px; padding: 8px 13px; border: 1px solid var(--ink-0); background: var(--paper-0); color: inherit; font: 700 .78rem/1 var(--font-body); letter-spacing: .05em; text-transform: uppercase; cursor: pointer; }
.manager-actions button:hover:not(:disabled), .confirmation-actions button:hover { background: var(--paper-1); }
.manager-actions button:disabled { cursor: not-allowed; opacity: .38; }
.manager-actions .primary { margin-left: auto; background: var(--accent-gold); }
.manager-actions .danger, .confirmation-actions .danger { color: var(--accent-red); border-color: var(--accent-red); }
.action-reason { min-height: 1.2em; margin: -9px 0 0; color: color-mix(in srgb, var(--ink-0) 67%, transparent); font-size: .8rem; }
.file-input { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0, 0, 0, 0); }
.confirmation { position: sticky; bottom: 0; z-index: 2; margin-top: auto; padding: 18px; border: 2px solid var(--accent-red); background: var(--paper-0); box-shadow: 0 -12px 30px rgba(46, 42, 34, .18); }
.confirmation h2 { margin: 0; font-family: var(--font-title); font-size: 1.15rem; text-transform: uppercase; }
.confirmation p { margin: 8px 0 14px; }
.confirmation-actions { display: flex; flex-wrap: wrap; gap: 8px; }
@media (max-width: 560px) {
  .save-row { grid-template-columns: 1fr; }
  .save-row time, .row-date, .save-row code { text-align: left; }
  .row-heading { align-items: flex-start; flex-direction: column; gap: 2px; }
  .manager-actions { display: grid; grid-template-columns: 1fr 1fr; }
  .manager-actions .primary { margin-left: 0; }
}
</style>
