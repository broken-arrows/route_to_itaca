<script setup lang="ts">
// A drawn, face-up hand card — full stationery anatomy per
// docs/design/reference/desk_ui.png (the two cards "Pressure the Cabinet" /
// "Inter-Party Outreach") and desk-frames.md §3 "Card anatomy":
//   gov  = WHITE ministry letterhead: Generalitat crest + department line +
//          red double rule, subject box "ASSUMPTE:", registre footer.
//   party= manila dossier: typed header row, framed poster art, subject box,
//          footer.
// Both carry a typed file-ref TAB overhanging the top and a peek sheet behind
// (the "there is a folder here" tell). The file-ref / letterhead / registre
// strings are DIEGETIC document furniture that is CONSTANT per skin (a
// Generalitat card is always the Generalitat; a pre-printed form code, like
// CONFIDENCIAL/ASSUMPTE already are) — not per-card data, so no numbering
// scheme is invented and the filler rule (plan §2) holds. Positioning (the
// two-rows-of-three grid) is DeskView's job; this component owns its own
// jitter rotation and the hover lift.
import { computed, ref } from 'vue';
import type { CardView } from '../../engine/types';
import { skinFor } from './skins';

const props = defineProps<{
  card: CardView;
  index: number;
  dimmed?: boolean;
}>();
interface CardScreenOrigin {
  x: number;
  y: number;
}

const emit = defineEmits<{ play: [card: CardView, origin: CardScreenOrigin] }>();

// Deterministic +/-2deg jitter table (NOT random — keeps renders stable and
// the component pure/testable). Six entries cover the two-rows-of-three cap.
const JITTER_DEG = [-2, 1.4, -1.6, 2, -1.2, 1.8];

const skin = computed(() => skinFor(props.card.role));
const jitter = computed(() => `${JITTER_DEG[props.index % JITTER_DEG.length]}deg`);

const imgBroken = ref(false);
const imgSrc = computed(() =>
  props.card.image ? `${import.meta.env.BASE_URL}${props.card.image}` : null,
);

function onClick(event: Event): void {
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
  emit('play', props.card, {
    x: rect.left + rect.width / 2,
    y: rect.top + rect.height / 2,
  });
}
</script>

<template>
  <div
    class="hand-card"
    :class="[`skin-${skin.key}`, { dimmed }]"
    :style="{ '--card-bd': skin.bd, '--jitter': jitter }"
    :data-test="`hand-card-${card.id}`"
    role="button"
    tabindex="0"
    @click="onClick"
    @keydown.enter.prevent="onClick"
    @keydown.space.prevent="onClick"
  >
    <!-- Peek sheet behind: a second document offset down-right, the blind
         "there is a stack in this folder" tell (desk-frames §3). -->
    <span class="peek" aria-hidden="true"></span>

    <!-- Typed file-ref tab overhanging the top edge. Diegetic constant per
         skin — not UI chrome, intentionally not i18n. -->
    <span v-if="skin.key !== 'neutral'" class="file-tab" aria-hidden="true">{{
      skin.key === 'gov' ? 'PRESIDÈNCIA · GOV-04' : 'PTY · MANIOBRA'
    }}</span>

    <div class="sheet">
      <!-- GOV: ministry letterhead + red double rule -->
      <div v-if="skin.key === 'gov'" class="letterhead" aria-hidden="true">
        <span class="crest"></span>
        <span class="ministry">GENERALITAT<br />DE CATALUNYA</span>
        <span class="dept">DEPT. DE LA PRESIDÈNCIA</span>
      </div>
      <span v-if="skin.key === 'gov'" class="gov-rule" aria-hidden="true"></span>

      <!-- PARTY: typed archive header row -->
      <div v-if="skin.key === 'party'" class="pty-header" aria-hidden="true">
        <span>DOSSIER DE PARTIT</span><span>EXP.</span>
      </div>

      <!-- Art region: framed poster for party, seal watermark for a gov card
           with no art (ministry stationery is text-first), striped
           placeholder otherwise (plan §9 — never a broken image). -->
      <div class="card-art" :class="{ framed: skin.key === 'party' }">
        <img v-if="imgSrc && !imgBroken" :src="imgSrc" alt="" @error="imgBroken = true" />
        <div v-else-if="skin.key === 'gov'" class="gov-blank" aria-hidden="true">
          <span class="crest crest-lg"></span>
        </div>
        <div v-else class="art-placeholder" aria-hidden="true"></div>
      </div>

      <div class="card-body">
        <!-- "ASSUMPTE:" is diegetic stationery text, not UI chrome —
             intentionally not i18n. -->
        <p class="card-title"><span class="assumpte">ASSUMPTE:</span> {{ card.title }}</p>
      </div>

      <p v-if="skin.key !== 'neutral'" class="footer" aria-hidden="true">{{
        skin.key === 'gov' ? 'REGISTRE OFICIAL · CONFIDENCIAL' : 'ARXIU DEL PARTIT · MANIOBRA'
      }}</p>
    </div>
  </div>
</template>

<style scoped>
.hand-card {
  /* The reference dossier keeps its 31:44 proportion while its physical
     geometry is independently bounded. The .sheet is the document. */
  width: clamp(150px, 11vw, 210px);
  height: auto;
  aspect-ratio: 31 / 44;
  border-radius: 4px;
  box-shadow: 0 12px 22px rgba(60, 45, 20, 0.28);
  cursor: pointer;
  position: relative;
  transform: rotate(var(--slot-rot, var(--jitter)));
  transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.2s ease;
}
.hand-card:hover {
  transform: translateY(-12px) rotate(0deg);
  box-shadow: 0 18px 30px rgba(46, 42, 34, 0.35);
  z-index: 5;
}
.hand-card.dimmed {
  opacity: 0.45;
}

/* The document sheet. Gov = white ministry paper; party = manila; neutral =
   plain cream. The folder edge/peek behind reads as the wallet holding it. */
.sheet {
  position: relative;
  z-index: 1;
  width: 100%;
  height: 100%;
  border: 1px solid var(--card-bd);
  border-radius: 4px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.skin-gov .sheet { background: #fdfcfb; }
.skin-party .sheet { background: #e3d3a8; }
.skin-neutral .sheet { background: #fdfcf8; }

/* Peek: a second sheet offset behind, in the folder's own tone. */
.peek {
  position: absolute;
  inset: 0;
  border-radius: 4px;
  border: 1px solid var(--card-bd);
  transform: rotate(1.6deg) translate(5px, 4px);
}
.skin-gov .peek { background: #efe9da; }
.skin-party .peek { background: #d8c68f; transform: rotate(-1.8deg) translate(-5px, 4px); }
.skin-neutral .peek { background: #efe9da; }

.file-tab {
  position: absolute;
  z-index: 2;
  top: -11px;
  left: 14px;
  padding: 3px 8px 4px;
  background: #f0ebdd;
  border: 1px solid var(--card-bd);
  border-bottom: none;
  border-radius: 3px 3px 0 0;
  font-family: var(--font-typed);
  font-size: 7.5px;
  letter-spacing: 0.1em;
  color: #6b655a;
}

/* Ministry letterhead */
.letterhead {
  display: grid;
  grid-template-columns: auto 1fr;
  grid-template-rows: auto auto;
  column-gap: 8px;
  align-items: center;
  padding: 12px 12px 6px;
}
.crest {
  grid-row: 1 / 3;
  width: 18px;
  height: 22px;
  background: repeating-linear-gradient(180deg, #f4c318 0 2.5px, #d4232a 2.5px 5px);
  border: 1px solid #b8901a;
  border-radius: 3px 3px 8px 8px / 3px 3px 12px 12px;
}
.ministry {
  font-family: var(--font-title);
  font-weight: 800;
  font-size: 9px;
  line-height: 1.1;
  letter-spacing: 0.04em;
  color: var(--ink-0);
}
.dept {
  grid-column: 2;
  font-family: var(--font-typed);
  font-size: 6.5px;
  letter-spacing: 0.06em;
  color: #8a7a58;
}
/* Ministry red double rule — paper anatomy, the ONE ratified red exception
   (tokens.css --paper-rule-ink). Never inline an off-token red here. */
.gov-rule {
  margin: 0 12px;
  border-bottom: 3px double var(--paper-rule-ink);
}

.pty-header {
  display: flex;
  justify-content: space-between;
  padding: 10px 12px 6px;
  font-family: var(--font-typed);
  font-size: 6.5px;
  letter-spacing: 0.08em;
  color: #8a7a58;
}

.card-art {
  flex: 1;
  margin: 8px 12px 0;
  border-radius: 2px;
  overflow: hidden;
  min-height: 0;
  background: transparent;
}
/* Party art is a pasted photo: white border + shadow (desk_ui.png). */
.card-art.framed {
  margin: 6px 12px 0;
  border: 4px solid #fdfcf8;
  box-shadow: 0 2px 5px rgba(46, 42, 34, 0.28);
  background: var(--paper-3);
}
.card-art img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.gov-blank {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.crest-lg {
  width: 44px;
  height: 54px;
  opacity: 0.16;
  border-radius: 5px 5px 18px 18px / 5px 5px 26px 26px;
}
.art-placeholder {
  width: 100%;
  height: 100%;
  background-image: repeating-linear-gradient(
    45deg,
    var(--paper-3),
    var(--paper-3) 8px,
    var(--paper-1) 8px,
    var(--paper-1) 16px
  );
}

/* Subject box — printed field on every skin (white inset). */
.card-body {
  margin: 8px 12px;
  padding: 6px 9px;
  background: #fdfcf8;
  border: 1px solid rgba(90, 70, 40, 0.18);
  border-radius: 2px;
}
.skin-party .card-body,
.skin-neutral .card-body {
  box-shadow: 0 1px 2px rgba(60, 45, 20, 0.1);
}
.card-title {
  margin: 0;
  font-family: var(--font-title);
  font-size: 12.5px;
  font-weight: 700;
  line-height: 1.2;
  color: var(--ink-0);
}
.assumpte {
  display: block;
  font-family: var(--font-typed);
  font-weight: 400;
  font-size: 8px;
  letter-spacing: 0.08em;
  opacity: 0.55;
}
.footer {
  margin: 0;
  padding: 0 12px 9px;
  font-family: var(--font-typed);
  font-size: 6.5px;
  letter-spacing: 0.08em;
  color: #8a7a58;
}
</style>
