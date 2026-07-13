<script setup lang="ts">
// In-tray: a face-down draw folder. Spec: prototype-draw-to-dossier-NOTES.md
// "Trays" (172x196, label above, face-down top folder, DRAW chip, blind —
// no counts). The FIXED chrome caption ("GOVERNMENT"/"PARTY"/"PARLAMENT")
// is NOT rendered here: this component is reused for all three tray kinds,
// and the deck scene's own `role` is mechanically always 'deck' (see
// compiler.role-derivation.test.ts), so it carries no gov/party/parlament
// distinction. DeskView (which knows the concrete deck ids) renders that
// label above the tray and re-tags the `deck` prop's `role` to the visual
// card-* role so skinFor resolves the right paper here. What IS rendered
// here is `deck.title` — the deck's own (game-content) name.
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import type { DeckView } from '../../engine/types';
import { skinFor } from './skins';

const props = defineProps<{
  deck: DeckView;
  disabled?: boolean;
}>();
const emit = defineEmits<{ draw: [deckId: string] }>();

const { t } = useI18n();

const skin = computed(() => skinFor(props.deck.role));
// The engine peeks deck drawability into canChoose (engine.js `_drawFromDeck`
// check at choice-compile time) without exposing a count — this is the one
// bit of "is this tray dry" signal that respects the blind design.
const empty = computed(() => props.deck.canChoose === false);

function onActivate(): void {
  if (props.disabled) return;
  emit('draw', props.deck.id);
}
</script>

<template>
  <div
    class="in-tray"
    :class="[`skin-${skin.key}`, { 'is-disabled': disabled, 'is-empty': empty }]"
    :style="{ '--tray-bg': skin.bg, '--tray-bd': skin.bd }"
    :data-test="`in-tray-${deck.id}`"
    role="button"
    :aria-disabled="disabled ? 'true' : 'false'"
    :tabindex="disabled ? -1 : 0"
    @click="onActivate"
    @keydown.enter.prevent="onActivate"
    @keydown.space.prevent="onActivate"
  >
    <p class="tray-label">{{ deck.title }}</p>
    <div class="tray-well">
      <div class="folder">
        <span v-if="skin.key === 'gov'" class="folder-seal" aria-hidden="true"></span>
        <!-- Diegetic stationery text, not UI chrome — intentionally not i18n:
             a Generalitat folder says CONFIDENCIAL whatever the UI language. -->
        <span v-if="skin.key === 'gov'" class="folder-stamp" aria-hidden="true">CONFIDENCIAL</span>
        <span v-if="skin.key === 'party'" class="folder-tie" aria-hidden="true"></span>
        <span v-if="skin.key === 'parlament'" class="folder-accent" aria-hidden="true"></span>
      </div>
      <p v-if="empty" class="tray-note">{{ t('desk.tray.empty') }}</p>
    </div>
    <span v-if="!empty" class="draw-chip">{{ t('desk.tray.draw') }} &#9656;</span>
  </div>
</template>

<style scoped>
.in-tray {
  width: 172px;
  height: 196px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  cursor: pointer;
  user-select: none;
}
.in-tray.is-disabled {
  cursor: not-allowed;
  opacity: 0.6;
}
.tray-label {
  margin: 0;
  font-family: var(--font-typed);
  font-size: 11px;
  letter-spacing: 0.04em;
  color: var(--ink-0);
  opacity: 0.75;
}
.tray-well {
  position: relative;
  flex: 1;
  border-radius: 4px;
  background: var(--paper-3);
  box-shadow: inset 0 3px 8px rgba(46, 42, 34, 0.28);
  padding: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.folder {
  position: relative;
  width: 128px;
  height: 148px;
  background: var(--tray-bg);
  border: 1px solid var(--tray-bd);
  border-radius: 3px;
  box-shadow: 0 3px 6px rgba(46, 42, 34, 0.25);
  transition: transform 0.15s ease;
}
.in-tray:not(.is-disabled):hover .folder {
  transform: translateY(-2px);
}
.folder-seal {
  position: absolute;
  top: 18px;
  left: 50%;
  transform: translateX(-50%);
  width: 46px;
  height: 46px;
  border-radius: 50%;
  border: 3px solid var(--ink-0);
  opacity: 0.13;
}
.folder-stamp {
  position: absolute;
  bottom: 22px;
  left: 50%;
  transform: translateX(-50%) rotate(-8deg);
  width: 84px;
  height: 24px;
  border: 1.5px solid var(--ink-0);
  border-radius: 3px;
  opacity: 0.22;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-typed);
  font-size: 10px;
  letter-spacing: 0.08em;
  color: var(--ink-0);
}
.folder-tie {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--ink-0);
  opacity: 0.35;
}
.folder-tie::after {
  content: '';
  position: absolute;
  top: 9px;
  left: 4px;
  width: 2px;
  height: 30px;
  background: var(--ink-0);
  opacity: 0.35;
}
.folder-accent {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: var(--accent-red);
  border-radius: 3px 3px 0 0;
}
.tray-note {
  position: absolute;
  inset: 0;
  margin: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 12px;
  font-family: var(--font-typed);
  font-size: 11px;
  color: var(--ink-0);
  opacity: 0.55;
}
.draw-chip {
  align-self: flex-end;
  font-family: var(--font-typed);
  font-size: 11px;
  letter-spacing: 0.06em;
  color: var(--paper-0);
  background: var(--ink-0);
  padding: 3px 8px;
  border-radius: 2px;
  transition: background-color 0.15s ease;
}
.in-tray:not(.is-disabled):hover .draw-chip {
  background: var(--accent-slate);
}
/* Red stays reserved for Parlament/world surfaces (binding style rule) —
   every other tray's hover uses --accent-slate above; only the Parlament
   tray itself gets the red hover the prototype used for all of them. */
.in-tray.skin-parlament:not(.is-disabled):hover .draw-chip {
  background: var(--accent-red);
}
</style>
