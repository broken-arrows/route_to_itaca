<script setup lang="ts">
// The draw-flip overlay. Spec: prototype-draw-to-dossier-NOTES.md motion
// sequence #1 (tray -> hand: .8s cubic-bezier(.3,.7,.3,1), starts face-down
// scale(.74) rotateY(180deg), ends face-up scale(1) rotateY(360deg);
// backface-visibility:hidden front/back layers). DeskView mounts this only
// during the 'drawing' phase, which already lasts exactly animMs('draw')
// via the desk store's own timer (see stores/desk.ts drawFrom) — this
// component just has to make its CSS duration match that window so the
// flip completes exactly as the phase clears (animations off => 0ms =>
// the keyframe resolves instantly, same "reduced motion" contract as the
// rest of the desk components).
//
// The back face is deliberately skin-agnostic (plain, no gov/party tell):
// consistent with the trays' "blind, no counts" rule — what's drawn stays
// unknown until the reveal, not just before it visually.
//
// Deliberately no from/to position props (brief's contract: FlyingCard
// {card: CardView} only). The reveal is centered in the live desk region,
// avoiding both design-canvas coordinates and coupling to a landing slot.
import { computed } from 'vue';
import type { CardView } from '../../engine/types';
import { useDeskStore } from '../../stores/desk';
import { skinFor } from './skins';

const props = defineProps<{ card: CardView }>();
const desk = useDeskStore();

const skin = computed(() => skinFor(props.card.role));
const drawMs = computed(() => `${desk.animMs('draw')}ms`);
</script>

<template>
  <div class="flying-card" data-test="flying-card" :style="{ '--draw-ms': drawMs }">
    <div class="flip-inner">
      <div class="face back" aria-hidden="true"></div>
      <div
        class="face front"
        :class="`skin-${skin.key}`"
        :style="{ '--card-bg': skin.bg, '--card-bd': skin.bd }"
      >
        <p class="card-title">{{ card.title }}</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.flying-card {
  position: absolute;
  left: 50%;
  top: 50%;
  width: clamp(150px, 11vw, 210px);
  height: auto;
  aspect-ratio: 31 / 44;
  transform: translate(-50%, -50%);
  perspective: 1200px;
  pointer-events: none;
  z-index: 60;
}
.flip-inner {
  position: relative;
  width: 100%;
  height: 100%;
  transform-style: preserve-3d;
  animation: draw-flip var(--draw-ms) cubic-bezier(0.3, 0.7, 0.3, 1) both;
}
@keyframes draw-flip {
  from {
    transform: scale(0.74) rotateY(180deg);
  }
  to {
    transform: scale(1) rotateY(360deg);
  }
}
.face {
  position: absolute;
  inset: 0;
  backface-visibility: hidden;
  border-radius: 4px;
  box-shadow: 0 10px 22px rgba(46, 42, 34, 0.35);
}
.face.back {
  background: var(--ink-0);
  transform: rotateY(180deg);
}
.face.front {
  background: var(--card-bg);
  border: 1px solid var(--card-bd);
  transform: rotateY(0deg);
  display: flex;
  align-items: flex-end;
  padding: 10px;
}
.card-title {
  margin: 0;
  font-family: var(--font-title);
  font-size: 13px;
  color: var(--ink-0);
}
</style>
