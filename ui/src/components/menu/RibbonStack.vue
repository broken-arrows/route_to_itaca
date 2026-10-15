<script setup lang="ts">
import { nextTick, ref } from 'vue';

export type RibbonTone = 'gold' | 'red' | 'dark';

export interface RibbonItem {
  id: string;
  title: string;
  subtitle?: string;
  disabled?: boolean;
  disabledReason?: string;
  tone?: RibbonTone;
}

const props = withDefaults(defineProps<{
  items: readonly RibbonItem[];
  activeId?: string | null;
  animations?: boolean;
  ariaLabel?: string;
}>(), {
  activeId: null,
  animations: true,
  ariaLabel: 'Menu',
});

const emit = defineEmits<{
  select: [id: string];
}>();

const buttons = ref<Record<string, HTMLButtonElement | null>>({});
const leaving = ref(false);

function dropOut(): Promise<unknown> | undefined {
  leaving.value = true;
  if (!props.animations || window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return;
  return Promise.all(Object.values(buttons.value).filter(Boolean).map((button, index) => {
    if (!button?.animate) return;
    return button.animate([
      { translate: getComputedStyle(button).translate },
      { translate: '0 110vh' },
    ], { duration: 300, delay: index * 35, easing: 'cubic-bezier(.55, 0, .85, .45)', fill: 'forwards' }).finished.catch(() => {});
  }));
}

function rememberButton(id: string, element: unknown): void {
  buttons.value[id] = element instanceof HTMLButtonElement ? element : null;
}

function select(item: RibbonItem): void {
  if (!item.disabled && !leaving.value) emit('select', item.id);
}

async function focusItem(id?: string | null): Promise<boolean> {
  await nextTick();
  const requested = id ? buttons.value[id] : null;
  const fallback = props.items
    .filter(item => !item.disabled)
    .map(item => buttons.value[item.id])
    .find((button): button is HTMLButtonElement => Boolean(button));
  const target = requested?.disabled ? fallback : (requested ?? fallback);
  target?.focus();
  return Boolean(target);
}

function focusActive(): Promise<boolean> {
  return focusItem(props.activeId);
}

function focusFirst(): Promise<boolean> {
  return focusItem();
}

defineExpose({ focusItem, focusActive, focusFirst, dropOut });
</script>

<template>
  <nav
    class="ribbon-stack"
    :class="{ 'is-animated': animations }"
    :aria-label="ariaLabel"
    :inert="leaving || undefined"
    data-test="ribbon-stack"
  >
    <button
      v-for="item in items"
      :key="item.id"
      :ref="element => rememberButton(item.id, element)"
      type="button"
      class="ribbon"
      :class="[
        `tone-${item.tone ?? 'gold'}`,
        { active: item.id === activeId },
      ]"
      :disabled="item.disabled"
      :title="item.disabled ? item.disabledReason : undefined"
      :aria-current="item.id === activeId ? 'page' : undefined"
      :data-test="`ribbon-${item.id}`"
      @click="select(item)"
    >
      <span class="ribbon-title type-control"><slot name="title" :item="item">{{ item.title }}</slot></span>
      <span v-if="item.subtitle" class="ribbon-subtitle type-compact"><slot name="subtitle" :item="item">{{ item.subtitle }}</slot></span>
      <span v-if="item.id === activeId" class="active-mark" aria-hidden="true" />
    </button>
  </nav>
</template>

<style scoped>
.ribbon-stack { display: flex; flex-direction: column; align-items: stretch; gap: clamp(14px, 2vh, 22px); width: 100%; min-width: 0; padding: 12px 18px; }
.ribbon {
  --ribbon-rotation: -0.7deg;
  --ribbon-skew: -4deg;
  --ribbon-width: 100%;
  --ribbon-fill: #efd17e;
  position: relative; display: grid; grid-template-columns: minmax(0, 1fr); gap: 4px; align-content: center;
  min-height: clamp(56px, 7vh, 78px); width: var(--ribbon-width); padding: 12px clamp(20px, 2.4vw, 36px);
  border: 0; border-radius: 0; background: var(--ribbon-fill); color: #30271f;
  box-shadow: 3px 3px 0 rgba(65, 49, 35, .65); text-align: left;
  transform: rotate(var(--ribbon-rotation)) skewX(var(--ribbon-skew));
  transition: transform 160ms ease, box-shadow 160ms ease; cursor: pointer;
}
.ribbon:nth-child(even) { --ribbon-fill: #dc9182; }
.ribbon:nth-child(6n + 2) { --ribbon-rotation: 0.4deg; --ribbon-skew: -2deg; --ribbon-width: 87%; }
.ribbon:nth-child(6n + 3) { --ribbon-rotation: -0.3deg; --ribbon-skew: 3deg; --ribbon-width: 95%; }
.ribbon:nth-child(6n + 4) { --ribbon-rotation: 0.7deg; --ribbon-skew: 4deg; --ribbon-width: 82%; }
.ribbon:nth-child(6n + 5) { --ribbon-rotation: 0.2deg; --ribbon-skew: -3deg; --ribbon-width: 92%; }
.ribbon:nth-child(6n) { --ribbon-rotation: -0.5deg; --ribbon-skew: 2deg; --ribbon-width: 85%; }
.ribbon-title { min-width: 0; font-size: clamp(16px, 1.5vw, 23px); font-weight: 600; text-transform: uppercase; }
.ribbon-subtitle { color: inherit; }
.active-mark { position: absolute; right: 12px; top: 50%; width: 8px; height: 8px; background: currentColor; transform: translateY(-50%) rotate(45deg); }
.ribbon.active, .ribbon:hover:not(:disabled), .ribbon:focus-visible { transform: none; box-shadow: 4px 4px 0 rgba(65, 49, 35, .65); z-index: 1; }
.ribbon:focus-visible { outline: 2px solid currentColor; outline-offset: -5px; }
.ribbon:disabled { filter: grayscale(.65); opacity: .55; cursor: not-allowed; }
.is-animated .ribbon { animation: ribbon-enter 380ms cubic-bezier(.16, .85, .3, 1) backwards; }
.is-animated .ribbon:nth-child(2) { animation-delay: 35ms; }
.is-animated .ribbon:nth-child(3) { animation-delay: 70ms; }
.is-animated .ribbon:nth-child(4) { animation-delay: 105ms; }
.is-animated .ribbon:nth-child(5) { animation-delay: 140ms; }
.is-animated .ribbon:nth-child(6) { animation-delay: 175ms; }
.is-animated .ribbon:nth-child(n + 7) { animation-delay: 210ms; }
@keyframes ribbon-enter { from { translate: 0 -110vh; } to { translate: 0 0; } }
@media (prefers-reduced-motion: reduce) { .is-animated .ribbon { animation: none; } .ribbon { transition: none; } }
.ribbon-stack:not(.is-animated) .ribbon { transition: none; }
</style>
