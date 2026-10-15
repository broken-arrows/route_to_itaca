<script setup lang="ts">
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
const sealSrc = `${import.meta.env.BASE_URL}img/cat_seal.svg`;

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
        <img class="crest" :src="sealSrc" alt="" />
        <span class="ministry">GENERALITAT<br />DE CATALUNYA</span>
        <span class="dept">DEPT. DE LA PRESIDÈNCIA</span>
      </div>
      <span v-if="skin.key === 'gov'" class="gov-rule" aria-hidden="true"></span>

      <!-- PARTY: typed archive header row -->
      <div v-if="skin.key === 'party'" class="pty-header" aria-hidden="true">
        <span>DOSSIER DE PARTIT</span><span>EXP.</span>
      </div>

      <!-- Art region: clipped photograph for party, printed cover art for a gov card
           with no art (ministry stationery is text-first), striped
           placeholder otherwise (plan §9 — never a broken image). -->
      <div class="card-art" :class="{ framed: skin.key === 'party' }">
        <img v-if="imgSrc && !imgBroken" :src="imgSrc" alt="" @error="imgBroken = true" />
        <div v-else-if="skin.key === 'gov'" class="gov-blank" aria-hidden="true">
          <img class="crest crest-lg" :src="sealSrc" alt="" />
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
  min-height: 0;
  isolation: isolate;
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
  font: var(--type-detail);
  letter-spacing: var(--type-tracking);
  position: absolute;
  z-index: 0;
  bottom: calc(100% - 2px);
  left: 14px;
  max-width: calc(100% - 28px);
  padding: 3px 8px 4px;
  background: #f0ebdd;
  border: 1px solid var(--card-bd);
  border-bottom: none;
  border-radius: 3px 3px 0 0;
  color: #6b655a;
  white-space: nowrap;
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
  object-fit: contain;
}
.ministry {
  font: var(--type-detail);
  letter-spacing: var(--type-tracking);
  color: var(--ink-0);
}
.dept {
  font: var(--type-detail);
  letter-spacing: var(--type-tracking);
  grid-column: 2;
  color: #655b48;
}
/* Ministry red double rule — paper anatomy, the ONE ratified red exception
   (tokens.css --paper-rule-ink). Never inline an off-token red here. */
.gov-rule {
  margin: 0 12px;
  border-bottom: 3px double var(--paper-rule-ink);
}

.pty-header {
  font: var(--type-detail);
  letter-spacing: var(--type-tracking);
  display: flex;
  justify-content: space-between;
  padding: 10px 12px 6px;
  color: #655b48;
}

.card-art {
  flex: 1;
  margin: 8px 12px 0;
  border-radius: 2px;
  overflow: hidden;
  min-height: 0;
  background: transparent;
}
/* A separate photograph clipped to the party folder. */
.card-art.framed {
  position: relative;
  overflow: visible;
  transform: rotate(-2deg);
  margin: 12px 14px 4px;
  border: 4px solid #fdfcf8;
  box-shadow: 0 2px 5px rgba(46, 42, 34, 0.28);
  background: var(--paper-3);
}
.card-art.framed::after {
  content: '';
  position: absolute;
  top: -14px;
  right: 16px;
  width: 10px;
  height: 29px;
  border: 2px solid #8c8e89;
  border-radius: 7px;
  box-shadow: inset 1px 0 #e9e9e3, 1px 1px 1px #504b4155;
  transform: rotate(12deg);
}
.skin-gov .card-art { border-radius: 0; }
.skin-gov .card-art > img { mix-blend-mode: multiply; }
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
.gov-blank .crest-lg {
  width: 44px;
  height: 54px;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  opacity: 0.16;
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
  font: var(--type-heading-compact);
  letter-spacing: var(--type-tracking);
  margin: 0;

  color: var(--ink-0);
}
.assumpte {
  font: var(--type-detail);
  letter-spacing: var(--type-tracking);
  display: block;
  color: #6b655a;
}
.footer {
  font: var(--type-detail);
  letter-spacing: var(--type-tracking);
  margin: 0;
  padding: 0 12px 9px;
  color: #655b48;
}
/* Preserve subject and poster space when six dossiers share a short desk.
   Only decorative stationery is reduced; playable titles retain their size. */
@media (max-height: 780px) {
  .footer, .dept { display: none; }
  .letterhead { padding: 7px 9px 4px; column-gap: 5px; }
  .ministry { font-size: .5rem; }
  .pty-header { padding: 7px 9px 4px; }
  .card-body { margin: 6px 9px 8px; padding: 4px 6px; }
  .card-art, .card-art.framed { margin-left: 9px; margin-right: 9px; }
}
</style>
