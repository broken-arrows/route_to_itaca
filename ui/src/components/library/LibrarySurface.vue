<script setup lang="ts">
import { computed } from 'vue';
import { useGameStore } from '../../stores/game';
import { useBriefStore } from '../../stores/brief';
import Clipboard from '../brief/Clipboard.vue';
import Prose from '../Prose.vue';
import { splitAuthoredPane } from '../menu/authoredPane';

defineProps<{ underlyingHtml: string }>();

const game = useGameStore();
const brief = useBriefStore();
const articleChoices = computed(() => game.frame?.choices ?? []);
const articlePane = computed(() => splitAuthoredPane(game.frame?.html ?? ''));
</script>

<template>
  <div class="library-surface" data-test="library-surface">
    <div
      class="library-underlay"
      data-test="library-underlay"
      inert
      aria-hidden="true"
      v-html="underlyingHtml"
    />
    <div class="library-brief"><Clipboard /></div>
    <article v-if="!brief.libraryAtIndex" class="library-article" data-test="library-article">
      <header><h1><Prose tag="span" :html="articlePane.titleHtml ?? game.frame?.title ?? ''" /></h1></header>
      <div class="library-article-body">
        <Prose :html="articlePane.bodyHtml" />
        <div class="library-article-options">
          <button
            v-for="(choice, index) in articleChoices"
            :key="choice.id"
            type="button"
            :disabled="!choice.canChoose"
            data-test="library-article-choice"
            @click="brief.chooseLibraryArticle(index)"
          ><Prose tag="span" :html="choice.title" /></button>
        </div>
      </div>
    </article>
  </div>
</template>

<style scoped>
.library-surface { position: absolute; inset: 0; overflow: hidden; }
.library-underlay { position: absolute; z-index: 0; isolation: isolate; inset: 0; }
.library-brief { position: absolute; inset: 0 auto 0 0; z-index: 3; width: clamp(340px, 36.5vw, 560px); }
.library-article { position: absolute; z-index: 2; inset: 0 0 0 clamp(340px, 36.5vw, 560px); display: flex; flex-direction: column; min-width: 0; background: var(--paper-0); border-left: 1px solid #c6bda8; box-shadow: -8px 0 24px rgba(46, 42, 34, .2); }
.library-article header { flex: none; border-bottom: 3px double #c6bda8; padding: 28px 40px 16px; }
.library-article h1 { margin: 0; color: var(--ink-0); font: 800 clamp(24px, 2.2vw, 38px)/1.15 var(--font-news); }
.library-article-body { flex: 1; min-height: 0; overflow: auto; padding: 26px clamp(32px, 5vw, 80px) 64px; color: var(--ink-0); font: 15px/1.65 var(--font-body); }
.library-article-options { display: flex; flex-direction: column; gap: 10px; margin-top: 28px; }
.library-article-options button { align-self: flex-start; border: 0; border-bottom: 1px solid #a89e8c; background: transparent; padding: 8px 2px; color: var(--ink-0); font: 700 13px/1.3 var(--font-title); cursor: pointer; }
.library-article-options button:focus-visible { outline: 3px solid var(--accent-red); outline-offset: 3px; }
@media (max-width: 760px) {
  .library-brief { width: 44%; }
  .library-article { left: 44%; }
  .library-article header { padding-inline: 22px; }
  .library-article-body { padding-inline: 22px; }
}
</style>
