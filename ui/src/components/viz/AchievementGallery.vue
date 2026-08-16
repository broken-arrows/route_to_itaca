<script setup lang="ts">
import { computed, reactive } from 'vue';
import { useGameStore } from '../../stores/game';
import { i18n } from '../../i18n';

defineOptions({ name: 'AchievementGallery' });

const props = withDefaults(
  defineProps<{ scope?: 'ever' | 'playthrough'; q?: Record<string, unknown> }>(),
  { scope: 'ever', q: undefined },
);

const game = useGameStore();
const locale = computed(() => i18n.global.locale.value);
const t = i18n.global.t;

// Snapshot the engine-owned ledger when the authored gallery mounts. Title and
// pause navigation remount it on each opening, refreshing relative labels
// without a permanent clock or a Vue-owned persistence domain.
const openedAt = Date.now();
const ledger = { ...game.achievementLedger };

function unlockedLabel(id: string): string {
  const value = ledger[id];
  if (!value || typeof value !== 'object') return t('shell.achievements.unknownDate');
  const raw = value.unlockedAt;
  if (typeof raw !== 'string') return t('shell.achievements.unknownDate');
  const timestamp = Date.parse(raw);
  if (!Number.isFinite(timestamp)) return t('shell.achievements.unknownDate');

  const elapsed = Math.max(0, openedAt - timestamp);
  if (elapsed < 24 * 60 * 60 * 1000) {
    const minutes = Math.floor(elapsed / (60 * 1000));
    if (minutes < 60) return t('shell.achievements.minutesAgo', { count: minutes });
    return t('shell.achievements.hoursAgo', { count: Math.floor(minutes / 60) });
  }
  return new Intl.DateTimeFormat(locale.value, {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  }).format(timestamp);
}

interface Row {
  id: string;
  name: string;
  description: string;
  stars: number;
  image: string;
  unlocked: boolean;
  unlockedLabel?: string;
}

const rows = computed<Row[]>(() => {
  const q = props.q ?? {};
  const prefix = props.scope === 'playthrough' ? 'game_achievement_' : 'achievement_';
  return game.achievements
    .filter((a) => props.scope !== 'playthrough' || !!q[prefix + a.id])
    .map((a) => ({
      ...a,
      unlocked: !!q[prefix + a.id],
      unlockedLabel: q[prefix + a.id] ? unlockedLabel(a.id) : undefined,
    }));
});

const unlockedCount = computed(() => rows.value.filter((row) => row.unlocked).length);
const totalCount = computed(() =>
  props.scope === 'playthrough' ? game.achievements.length : rows.value.length,
);

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
    <div class="achievement-gallery-count" data-test="achievement-count">
      {{ t('shell.achievements.count', { unlocked: unlockedCount, total: totalCount }) }}
    </div>
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
        <div class="achievement-row-description">{{ row.description }}</div>
      </div>
      <div class="achievement-row-meta">
        <div class="achievement-row-stars" :aria-label="`${row.stars}/5`">
          <span v-for="i in 5" :key="i" :class="i <= row.stars ? 'star--filled' : 'star--empty'"
            >★</span
          >
        </div>
        <div
          v-if="row.unlockedLabel"
          class="achievement-row-date"
          :data-test="`achievement-date-${row.id}`"
        >
          {{ row.unlockedLabel }}
        </div>
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
.achievement-gallery-count {
  align-self: flex-end;
  font-family: var(--font-body);
  font-size: 12px;
  color: var(--ink-1, var(--ink-0));
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
  white-space: nowrap;
}
.achievement-row-stars .star--filled {
  color: var(--accent-gold);
}
.achievement-row-stars .star--empty {
  color: var(--paper-3);
}
.achievement-row-description {
  margin-top: 4px;
  font-family: var(--font-body);
  font-size: 12px;
  color: var(--ink-0);
  opacity: 0.85;
}
.achievement-row-meta {
  flex: 0 0 112px;
  align-self: stretch;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  justify-content: center;
  text-align: right;
}
.achievement-row-date {
  margin-top: 5px;
  font-family: var(--font-body);
  font-size: 11px;
  font-style: italic;
  color: var(--ink-1, var(--ink-0));
}

@media (max-width: 560px) {
  .achievement-row {
    align-items: flex-start;
    flex-wrap: wrap;
  }
  .achievement-row-body {
    flex-basis: calc(100% - 86px);
  }
  .achievement-row-meta {
    flex-basis: 100%;
    align-items: flex-start;
    padding-left: 86px;
    text-align: left;
  }
}
</style>
