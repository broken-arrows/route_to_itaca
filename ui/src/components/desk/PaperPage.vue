<script setup lang="ts">
// PaperPage — the neutral full-page paper surface for the Desk UI's
// non-desk phases (Desk UI Phase 2, Task 8: .superpowers/sdd/p2-task-8-brief.md).
// One paper sheet on the stage: prose (v-html of the live frame's HTML — a
// Frame carries no separate title field, so the scene's own content stands
// as its own opener, same convention DebugPage/OpenDossier already use) +
// a simple option list. Three variants:
//   - 'page'   plain neutral paper (the "misfiled folder" default)
//   - 'event'  adds the ONE red top band this phase permits — world/
//              Parlament signal, per desk_ui_plan.md's "red is reserved"
//              rule. GameView selects this for effectiveRole 'event'.
//   - 'ending' neutral paper + a typed-ink "closed file" stamp treatment,
//              no invented flavour copy. GameView selects this when the
//              destination scene's effectiveRole is 'ending'.
//
// Options reuse PaperOption (the same option-slip the dossier uses) for
// visual/behavioural parity, wrapped the same way OpenDossier wraps it — a
// plain click listener around each slot rather than PaperOption's own
// `pick` emit. Unlike OpenDossier, a locked click here is simply ignored:
// PaperPage has no dossier/shake choreography (that lives on the desk
// store, scoped to 'dossierOpen'), so there is nothing to animate or
// report — just call game.choose(i) directly for a choosable option.
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { useGameStore } from '../../stores/game';
import type { ChoiceView } from '../../engine/types';
import PaperOption from './PaperOption.vue';
import Prose from '../Prose.vue';
import { useSettingsStore } from '../../stores/settings';

const props = defineProps<{ variant: 'page' | 'event' | 'ending' }>();

const { t } = useI18n();
const game = useGameStore();
const settings = useSettingsStore();
const imageBroken = ref(false);
const assetBase = import.meta.env.BASE_URL;
watch(() => game.frame?.sceneId, () => { imageBroken.value = false; });

const choices = computed<ChoiceView[]>(() => game.frame?.choices ?? []);

function onPick(i: number, choice: ChoiceView): void {
  if (!choice.canChoose) return;
  game.choose(i);
}
</script>

<template>
  <div class="paper-page" :class="`variant-${props.variant}`" data-test="paper-page">
    <div v-if="props.variant === 'event'" class="event-band" data-test="event-band" aria-hidden="true"></div>
    <div class="paper-sheet">
      <span v-if="props.variant === 'ending'" class="ended-stamp" data-test="ended-stamp">
        {{ t('desk.page.ended') }}
      </span>
      <img
        v-if="props.variant === 'event' && settings.eventImages && game.frame?.faceImage && !imageBroken"
        class="event-image"
        :src="`${assetBase}${game.frame.faceImage}`"
        alt=""
        data-test="event-face-image"
        @error="imageBroken = true"
      />
      <!-- Engine-authored prose; trusted content from our own game.json
           (same trust boundary as DebugPage/OpenDossier's Prose). -->
      <Prose class="prose" :html="game.frame?.html ?? ''" />
      <div v-if="choices.length" class="options">
        <div v-for="(choice, i) in choices" :key="choice.id" class="paper-slot" @click="onPick(i, choice)">
          <PaperOption :choice="choice" :index="i" :shaking="false" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.paper-page {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  overflow-y: auto;
  padding-bottom: 60px;
}
.event-band {
  flex: none;
  width: 100%;
  height: 10px;
  background: var(--accent-red);
}
.paper-sheet {
  position: relative;
  width: clamp(760px, 62vw, 1100px);
  max-width: calc(100% - clamp(32px, 5vw, 96px));
  margin-top: 40px;
  background: var(--paper-0);
  border: 1px solid var(--ink-0);
  border-radius: 3px;
  padding: 32px 40px;
  box-shadow: 0 10px 24px rgba(46, 42, 34, 0.18);
}
.ended-stamp {
  display: inline-block;
  float: right;
  font-family: var(--font-typed);
  font-size: 13px;
  letter-spacing: 0.12em;
  color: var(--ink-0);
  opacity: 0.6;
  border: 2px solid var(--ink-0);
  border-radius: 2px;
  padding: 2px 10px;
  transform: rotate(4deg);
}
.prose {
  font-family: var(--font-body);
  font-size: 15px;
  line-height: 1.6;
  color: var(--ink-0);
  clear: both;
}
.event-image { float: right; width: min(36%, 340px); max-height: 250px; margin: 0 0 20px 28px; object-fit: contain; }
.options {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 24px;
}
.paper-slot {
  cursor: pointer;
}
</style>
