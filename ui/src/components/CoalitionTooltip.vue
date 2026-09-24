<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { autoUpdate, flip, offset, shift, useFloating } from '@floating-ui/vue';
import { usePartyLogo } from './viz/usePartyLogo';
import '../styles/tooltips.css';

const props = defineProps<{
  institution: 'generalitat' | 'gobierno' | 'ajuntament';
  parties: string[];
  summary: string;
  anchor: Element | null;
  accent?: string;
}>();

const TITLES = {
  generalitat: 'Generalitat de Catalunya',
  gobierno: 'Gobierno de España',
  ajuntament: 'Ajuntament de Barcelona',
} as const;

const reference = ref<Element | null>(null);
const floating = ref<HTMLElement | null>(null);
watch(() => props.anchor, (el) => { reference.value = el; }, { immediate: true });

const { floatingStyles } = useFloating(reference, floating, {
  placement: 'top',
  whileElementsMounted: autoUpdate,
  middleware: [offset(10), flip(), shift({ padding: 8 })],
});

const partyLogo = usePartyLogo();
const members = computed(() => props.parties.map((id) => ({ id, logo: partyLogo(id) })));
</script>

<template>
  <Teleport to="body">
    <div
      ref="floating"
      :style="[floatingStyles, { '--coalition-tooltip-color': accent }]"
      class="glossary-popover coalition-popover"
      role="tooltip"
      data-test="coalition-popover"
    >
      <div class="coalition-logos" data-test="coalition-logos">
        <template v-for="member in members" :key="member.id">
          <img
            v-if="member.logo"
            class="coalition-logo"
            :src="member.logo"
            :alt="`${member.id} logo`"
            :data-party="member.id"
          />
          <span v-else class="coalition-logo-fallback" :data-party="member.id">{{ member.id }}</span>
        </template>
      </div>
      <div class="popover-title">{{ TITLES[institution] }}</div>
      <div v-if="summary" class="popover-subtitle">{{ summary }}</div>
    </div>
  </Teleport>
</template>
