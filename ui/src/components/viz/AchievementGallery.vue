<script setup lang="ts">
import { computed, reactive } from 'vue';
import { useGameStore } from '../../stores/game';

defineOptions({ name: 'AchievementGallery' });

const props = withDefaults(
  defineProps<{ scope?: 'ever' | 'playthrough'; q?: Record<string, unknown> }>(),
  { scope: 'ever', q: undefined },
);

const game = useGameStore();

interface Row {
  id: string;
  name: string;
  description: string;
  stars: number;
  image: string;
  unlocked: boolean;
}

const rows = computed<Row[]>(() => {
  const q = props.q ?? {};
  const prefix = props.scope === 'playthrough' ? 'game_achievement_' : 'achievement_';
  return game.achievements
    .filter((a) => props.scope !== 'playthrough' || !!q[prefix + a.id])
    .map((a) => ({
      ...a,
      unlocked: !!q[prefix + a.id],
    }));
});

// Registry image paths are web-root-relative (`img/...`), same as every other
// art path in compiled content — resolve against BASE_URL and fall back to the
// striped placeholder on 404, matching HandCard/ActionsTray/GlossaryTerm
// (spec §9: never a broken image).
const broken = reactive<Record<string, boolean>>({});
function imgSrc(row: Row): string | null {
  return row.image ? `${import.meta.env.BASE_URL}${row.image}` : null;
}
</script>

<template>
  <div class="achievement-gallery" data-test="achievement-gallery">
    <div
      v-for="row in rows"
      :key="row.id"
      class="achievement-row"
      :class="row.unlocked ? 'achievement-row--unlocked' : 'achievement-row--locked'"
      :data-test="`achievement-row-${row.id}`"
    >
      <div class="achievement-row-image">
        <img
          v-if="imgSrc(row) && !broken[row.id]"
          :src="imgSrc(row)!"
          :alt="row.name"
          @error="broken[row.id] = true"
        />
        <div v-else class="art-placeholder" aria-hidden="true"></div>
      </div>
      <div class="achievement-row-body">
        <div class="achievement-row-name">{{ row.name }}</div>
        <div class="achievement-row-stars" :aria-label="`${row.stars}/5`">
          <span v-for="i in 5" :key="i" :class="i <= row.stars ? 'star--filled' : 'star--empty'"
            >★</span
          >
        </div>
        <div class="achievement-row-description">{{ row.description }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.achievement-gallery {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.achievement-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border: 1px solid var(--ink-0);
  border-radius: 3px;
  background: var(--paper-0);
}
.achievement-row--locked {
  opacity: 0.6;
}
.achievement-row--locked .achievement-row-image img {
  filter: grayscale(1);
}
.achievement-row-image {
  flex: 0 0 auto;
  width: 72px;
  aspect-ratio: 3 / 2;
  overflow: hidden;
  border-radius: 2px;
  border: 1px solid var(--ink-0);
}
.achievement-row-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.art-placeholder {
  width: 100%;
  height: 100%;
  background-image: repeating-linear-gradient(
    45deg,
    var(--paper-3),
    var(--paper-3) 8px,
    var(--paper-2) 8px,
    var(--paper-2) 16px
  );
}
.achievement-row-body {
  flex: 1 1 auto;
  min-width: 0;
}
.achievement-row-name {
  font-family: var(--font-title);
  font-size: 14px;
  font-weight: 600;
  color: var(--ink-0);
}
.achievement-row-stars {
  margin: 2px 0 4px;
}
.achievement-row-stars .star--filled {
  color: var(--accent-gold);
}
.achievement-row-stars .star--empty {
  color: var(--paper-3);
}
.achievement-row-description {
  font-family: var(--font-body);
  font-size: 12px;
  color: var(--ink-0);
  opacity: 0.85;
}
</style>
