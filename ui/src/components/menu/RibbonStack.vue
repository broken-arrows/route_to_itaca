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

function rememberButton(id: string, element: unknown): void {
  buttons.value[id] = element instanceof HTMLButtonElement ? element : null;
}

function select(item: RibbonItem): void {
  if (!item.disabled) emit('select', item.id);
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

defineExpose({ focusItem, focusActive, focusFirst });
</script>

<template>
  <nav
    class="ribbon-stack"
    :class="{ 'is-animated': animations }"
    :aria-label="ariaLabel"
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
      <span class="ribbon-title"><slot name="title" :item="item">{{ item.title }}</slot></span>
      <span v-if="item.subtitle" class="ribbon-subtitle"><slot name="subtitle" :item="item">{{ item.subtitle }}</slot></span>
      <span v-if="item.id === activeId" class="active-mark" aria-hidden="true" />
    </button>
  </nav>
</template>

<style scoped>
.ribbon-stack {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: clamp(10px, 1.25vh, 17px);
  width: 100%;
  padding: 8px 12px;
}

.ribbon {
  --ribbon-accent: var(--accent-gold);
  --ribbon-fill: color-mix(in srgb, var(--paper-0) 84%, var(--accent-gold) 16%);
  --ribbon-rotation: -0.25deg;
  --ribbon-offset: 0px;
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, auto);
  align-items: center;
  gap: 18px;
  min-height: clamp(58px, 7vh, 88px);
  width: 100%;
  padding: 14px clamp(22px, 3vw, 42px);
  border: 1px solid color-mix(in srgb, var(--ribbon-accent) 42%, transparent);
  border-left: 10px solid var(--ribbon-accent);
  border-radius: 0;
  background: var(--ribbon-fill);
  color: var(--ink-0);
  box-shadow: 0 10px 18px rgba(46, 42, 34, 0.13);
  font: inherit;
  text-align: left;
  transform: translateX(var(--ribbon-offset)) rotate(var(--ribbon-rotation));
  cursor: pointer;
}

.ribbon:nth-child(3n + 2) {
  --ribbon-rotation: 0.45deg;
  --ribbon-offset: 8px;
}

.ribbon:nth-child(3n) {
  --ribbon-rotation: -0.55deg;
  --ribbon-offset: -5px;
}

.tone-red {
  --ribbon-accent: var(--accent-red);
  --ribbon-fill: color-mix(in srgb, var(--paper-0) 86%, var(--accent-red) 14%);
}

.tone-dark {
  --ribbon-accent: var(--accent-gold);
  --ribbon-fill: var(--ink-0);
  color: var(--paper-0);
}

.ribbon-title {
  min-width: 0;
  font-family: var(--font-title);
  font-size: clamp(1rem, 1.5vw, 1.35rem);
  font-weight: 800;
  letter-spacing: 0.035em;
  line-height: 1.1;
  text-transform: uppercase;
}

.ribbon-subtitle {
  max-width: 30ch;
  color: color-mix(in srgb, currentColor 65%, transparent);
  font-size: clamp(0.72rem, 1vw, 0.9rem);
  line-height: 1.25;
  text-align: right;
}

.active-mark {
  position: absolute;
  right: 17px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--ribbon-accent);
}

.ribbon:hover:not(:disabled) {
  filter: brightness(0.98);
  box-shadow: 0 12px 22px rgba(46, 42, 34, 0.18);
}

.ribbon:focus-visible {
  outline: 3px solid var(--ink-0);
  outline-offset: 4px;
  z-index: 1;
}

.ribbon:disabled {
  filter: grayscale(0.8);
  opacity: 0.48;
  cursor: not-allowed;
}

.is-animated .ribbon {
  animation: ribbon-enter 220ms cubic-bezier(0.22, 0.72, 0.24, 1) both;
}

.is-animated .ribbon:nth-child(2) { animation-delay: 25ms; }
.is-animated .ribbon:nth-child(3) { animation-delay: 50ms; }
.is-animated .ribbon:nth-child(4) { animation-delay: 75ms; }
.is-animated .ribbon:nth-child(5) { animation-delay: 100ms; }
.is-animated .ribbon:nth-child(6) { animation-delay: 125ms; }
.is-animated .ribbon:nth-child(n + 7) { animation-delay: 150ms; }

@keyframes ribbon-enter {
  from {
    opacity: 0;
    transform: translateX(calc(var(--ribbon-offset) + 32px)) rotate(var(--ribbon-rotation));
  }
  to {
    opacity: 1;
    transform: translateX(var(--ribbon-offset)) rotate(var(--ribbon-rotation));
  }
}

@media (max-width: 700px) {
  .ribbon {
    grid-template-columns: minmax(0, 1fr);
    gap: 4px;
    padding-inline: 20px;
  }

  .ribbon-subtitle { text-align: left; }
}

@media (prefers-reduced-motion: reduce) {
  .is-animated .ribbon { animation: none; }
}
</style>
