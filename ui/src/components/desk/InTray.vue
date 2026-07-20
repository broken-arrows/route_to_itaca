<script setup lang="ts">
// In-tray: a face-down draw folder. Geometry/colours: docs/design/reference/
// desk-frames.md §3 "In-trays" (172×212 well, 2px #c3b893 border, blind —
// no counts, top dossier with a peek sheet behind, DRAW chip overhanging
// bottom-right). The FIXED chrome caption ("GOVERNMENT"/"PARTY"/"PARLAMENT")
// is NOT rendered here: this component is reused for all three tray kinds.
// The deck scene's `role` (deck-gov, deck-party, or deck-parliament; plain
// 'deck' = neutral fallback) flows from the adapter; skinFor maps it to the
// paper skin. What IS rendered here is `deck.title` — the deck's own
// (game-content) name.
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
      <template v-if="!empty">
        <!-- Sliver of a second sheet peeking behind the top dossier — the
             "there is a stack in here" tell that replaces any count. -->
        <span class="peek" aria-hidden="true"></span>
        <div class="folder">
          <span v-if="skin.key === 'gov'" class="folder-clip" aria-hidden="true"></span>
          <span v-if="skin.key === 'gov'" class="folder-seal" aria-hidden="true"></span>
          <!-- Diegetic stationery text, not UI chrome — intentionally not i18n:
               a Generalitat folder says CONFIDENCIAL whatever the UI language.
               Ink-toned, NOT the canvas's red — red stays reserved for
               world/Parlament signals (tokens.css --paper-rule-ink comment). -->
          <span v-if="skin.key === 'gov'" class="folder-stamp" aria-hidden="true">CONFIDENCIAL</span>
          <span v-if="skin.key === 'party'" class="folder-tie" aria-hidden="true"></span>
          <span v-if="skin.key === 'parliament'" class="folder-accent" aria-hidden="true"></span>
        </div>
      </template>
      <p v-if="empty" class="tray-note">{{ t('desk.tray.empty') }}</p>
    </div>
    <span v-if="!empty" class="draw-chip">{{ t('desk.tray.draw') }} &#9656;</span>
  </div>
</template>

<style scoped>
/* Tray hexes are design-canvas literals (desk-frames.md §3), kept literal
   like skins.ts does; token vars used where an exact token exists. */
.in-tray {
  position: relative;
  width: 172px;
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
/* Caption above-left — desk-frames §3: 800 8.5px letter-spacing .12em
   #6f5f3e. The face behind --font-title is swappable (user, 2026-07-20);
   weight/tracking/case carry to whatever face lands there. */
.tray-label {
  margin: 0;
  font-family: var(--font-title);
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #6f5f3e;
}
/* Tray base: border 2px #c3b893, radius 10, translucent paper fill, soft
   inset — the canvas's tray-base recipe, shared with OutTray. */
.tray-well {
  position: relative;
  height: 212px;
  border: 2px solid #c3b893;
  border-radius: 10px;
  background: rgba(250, 249, 245, 0.4);
  box-shadow: inset 0 3px 8px rgba(60, 45, 20, 0.14);
  display: flex;
  align-items: center;
  justify-content: center;
}
.peek {
  position: absolute;
  width: 132px;
  height: 164px;
  background: #efe9da;
  border: 1px solid #ddd5c2;
  border-radius: 3px;
  transform: rotate(1.6deg) translate(4px, 3px);
}
.in-tray.skin-party .peek {
  transform: rotate(-1.8deg) translate(-3px, 3px);
}
.folder {
  position: relative;
  width: 136px;
  height: 168px;
  background: var(--tray-bg);
  border: 1px solid var(--tray-bd);
  border-radius: 3px;
  box-shadow: 0 5px 11px rgba(60, 45, 20, 0.2);
  transform: rotate(var(--folder-rot, 0deg));
  transition: transform 0.15s ease;
}
.in-tray.skin-gov .folder { --folder-rot: -1deg; }
.in-tray.skin-party .folder { --folder-rot: 1deg; }
.in-tray:not(.is-disabled):hover .folder {
  transform: translateY(-2px) rotate(var(--folder-rot, 0deg));
}
/* Paperclip detail, top edge — 13×28 outline, #b0a488. */
.folder-clip {
  position: absolute;
  top: -9px;
  left: 16px;
  width: 13px;
  height: 28px;
  border: 2px solid #b0a488;
  border-radius: 6px;
  background: transparent;
}
/* Generalitat seal watermark 44×52 at opacity .13. */
.folder-seal {
  position: absolute;
  top: 22px;
  left: 50%;
  transform: translateX(-50%);
  width: 44px;
  height: 52px;
  border-radius: 50%;
  border: 3px solid var(--ink-0);
  opacity: 0.13;
}
.folder-stamp {
  position: absolute;
  bottom: 18px;
  left: 10px;
  transform: rotate(-4deg);
  border: 2px solid var(--ink-0);
  border-radius: 3px;
  opacity: 0.22;
  padding: 2px 6px;
  font-family: var(--font-typed);
  font-size: 9.5px;
  letter-spacing: 0.08em;
  color: var(--ink-0);
}
/* Elastic tie: 14px disc + 54px string at 28°, #a58f56 (desk-frames §3). */
.folder-tie {
  position: absolute;
  top: 12px;
  right: 16px;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  border: 2px solid #a58f56;
  background: rgba(165, 143, 86, 0.25);
}
.folder-tie::after {
  content: '';
  position: absolute;
  top: 11px;
  left: 4px;
  width: 2px;
  height: 54px;
  background: #a58f56;
  transform: rotate(28deg);
  transform-origin: top center;
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
/* DRAW chip: dark, overhanging the tray's bottom-right corner, slight
   rotation (∓2° gov/party) — desk-frames §3. */
.draw-chip {
  position: absolute;
  right: -8px;
  bottom: -6px;
  background: #2e2a22;
  color: var(--paper-1);
  border-radius: 4px;
  padding: 3px 9px;
  font-family: var(--font-title);
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.08em;
  transform: rotate(-2deg);
  box-shadow: 0 3px 6px rgba(46, 42, 34, 0.35);
  transition: background-color 0.15s ease;
}
.in-tray.skin-party .draw-chip {
  transform: rotate(2deg);
}
.in-tray:not(.is-disabled):hover .draw-chip {
  background: var(--accent-slate);
}
/* Red stays reserved for Parlament/world surfaces (binding style rule) —
   every other tray's hover uses --accent-slate above; only the Parlament
   tray itself gets the red hover the prototype used for all of them. */
.in-tray.skin-parliament:not(.is-disabled):hover .draw-chip {
  background: var(--accent-red);
}
</style>
