<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue';
import { gameLib } from '../../game-bindings';
import { useGameStore } from '../../stores/game';
import { useElectionSvg } from './useElectionSvg';

const props = withDefaults(defineProps<{ q?: Record<string, unknown> }>(), { q: () => ({}) });
const root = ref<HTMLElement | null>(null);
const { markup, failed } = useElectionSvg('congreso-results.svg');
const regions = computed(() => gameLib.congresoPartyTour(props.q));
const game = useGameStore();

const SELECTORS: Record<string, string[]> = {
  rest: ['andalusia', 'castile_and_leon', 'castile_la_mancha', 'extremadura', 'galicia', 'la_rioja', 'madrid', 'murcia', 'ceuta', 'melilla'],
  catalonia: ['catalonia'], ppcc: ['valencia', 'balearic_islands'],
  eh: ['basque_country', 'navarre'], galicia: ['galicia'],
  others: ['cantabria', 'asturias', 'canary_islands'],
};
interface Anchor { x: number; y: number; cols: number; only?: string[]; tray?: boolean; line?: Array<[number, number]> }
const ANCHORS: Record<string, Anchor[]> = {
  rest: [{ x: 180, y: 120, cols: 3 }],
  catalonia: [{ x: 410, y: 50, cols: 2, tray: true }],
  ppcc: [{ x: 380, y: 235, cols: 2, only: ['compromis'], line: [[370, 180], [400, 230]] }, { x: 490, y: 180, cols: 2, only: ['mesm'] }],
  eh: [{ x: 250, y: -40, cols: 2, only: ['pnv', 'ehbildu', 'amaiur'], line: [[310, 35], [290, 5]] }, { x: 340, y: -30, cols: 3, only: ['gbai', 'nsuma', 'ehbildu', 'amaiur', 'upn'], line: [[335, 50], [395, 15]] }],
  galicia: [{ x: 60, y: 25, cols: 2, line: [[150, 40], [90, 45]] }],
  others: [{ x: 330, y: 90, cols: 1, only: ['texiste'], tray: true }, { x: 150, y: -50, cols: 1, only: ['fac'], line: [[210, 20], [180, -10]] }, { x: 120, y: 100, cols: 1, only: ['prc'], line: [[260, 30], [155, 100]] }, { x: 100, y: 310, cols: 1, only: ['cc'] }],
};
const LOGOS: Record<string, string> = {
  psoe: 'logo_psoe.png', psc: 'logo_psc.svg', pp: 'logo_pp.png', csspa: 'logo_cs.svg', vox: 'logo_vox.svg',
  up: 'logo_up.svg', podemos: 'logo_podemos.svg', iu: 'logo_iu.svg', mpais: 'logo_mpais.svg', nsuma: 'logo_nsuma.png', upyd: 'logo_upyd.svg',
  erc: 'logo_erc.png', ciu: 'logo_ciu.png', cdc: 'logo_cdc.png', dl: 'logo_dl.png', jxcat: 'logo_jxcat.svg',
  jxsi: 'logo_jxsi.png', junts: 'logo_junts.png', pdcat: 'logo_pdecat.svg', unio: 'logo_unio.png', cup: 'logo_cup.svg', fr: 'logo_fr.svg',
  compromis: 'logo_compromis.svg', mesm: 'logo_mes.png', amaiur: 'logo_amaiur.svg', pnv: 'logo_pnv.svg',
  ehbildu: 'logo_ehbildu.png', upn: 'logo_upn.svg', gbai: 'logo_gbai.svg', bng: 'logo_bng.svg',
  fac: 'logo_fac.svg', prc: 'logo_prc.png', texiste: 'logo_te.jpg', cc: 'logo_ccpnc.svg',
};
const NS = 'http://www.w3.org/2000/svg';
const XHTML_NS = 'http://www.w3.org/1999/xhtml';
function svgNode(name: string, attributes: Record<string, string>) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}
function addParties(svg: SVGSVGElement, id: string, parties: string[]) {
  for (const anchor of ANCHORS[id] ?? []) {
    const shown = anchor.only ? parties.filter(p => anchor.only?.includes(p)) : parties;
    if (!shown.length) continue;
    const cols = Math.min(anchor.cols, shown.length);
    const rows = Math.ceil(shown.length / cols);
    const group = svgNode('g', { class: 'party-tour-logos', 'data-interest-region': id });
    const logoSize = 24;
    const columnWidths = Array.from({ length: cols }, (_, column) =>
      Math.max(...shown.filter((_, index) => index % cols === column).map(party =>
        Math.max(logoSize, Math.max(30, Math.min(72, party.length * 6.2 + 12))))),
    );
    const stripWidth = columnWidths.reduce((sum, width) => sum + width, 0) + (cols - 1) * 6;
    const stripHeight = rows * 41 + (rows - 1) * 11;
    if (anchor.line) {
      group.appendChild(svgNode('polyline', {
        points: anchor.line.map(point => point.join(',')).join(' '), fill: 'none',
        stroke: '#443c32', 'stroke-width': '1.25', 'stroke-linecap': 'round', 'stroke-linejoin': 'round',
      }));
      group.appendChild(svgNode('circle', {
        cx: String(anchor.line[0][0]), cy: String(anchor.line[0][1]), r: '2.5', fill: '#443c32',
      }));
    }
    if (anchor.tray) group.appendChild(svgNode('rect', {
      x: String(anchor.x - 7), y: String(anchor.y - 7), width: String(stripWidth + 14), height: String(stripHeight + 14),
      rx: '6', fill: '#eee', 'fill-opacity': '.54',
    }));
    shown.forEach((party, index) => {
      const column = index % cols;
      const x = anchor.x + columnWidths.slice(0, column).reduce((sum, width) => sum + width, 0)
        + column * 6 + (columnWidths[column] - logoSize) / 2;
      const y = anchor.y + Math.floor(index / cols) * 52;
      const image = svgNode('image', {
        href: `${import.meta.env.BASE_URL}img/parties/${LOGOS[party] ?? ''}`,
        x: String(x), y: String(y), width: '24', height: '24',
      });
      image.appendChild(svgNode('title', {})).textContent = party.toUpperCase();
      group.appendChild(image);
      const term = game.glossary.find(entry =>
        entry.match.some(match => match.toLowerCase() === party.toLowerCase()));
      const label = svgNode('foreignObject', {
        x: String(x - 24), y: String(y + 27), width: '72', height: '18',
      });
      const text = document.createElementNS(XHTML_NS, 'div');
      text.style.cssText = 'width:100%;text-align:center;white-space:nowrap;font:10px/1.2 Arial,sans-serif;color:#444;';
      if (term) {
        const trigger = document.createElementNS(XHTML_NS, 'span');
        trigger.className = `term${term.tooltip ? ' term-hoverable' : ''}`;
        trigger.setAttribute('data-term', term.id);
        trigger.textContent = term.display ?? party.toUpperCase();
        if (term.colour) trigger.style.color = term.colour.startsWith('#') ? term.colour : `var(--${term.colour})`;
        if (term.bold) trigger.style.fontWeight = 'bold';
        if (term.tooltip) trigger.style.cursor = 'help';
        text.appendChild(trigger);
      } else {
        text.textContent = party.toUpperCase();
      }
      label.appendChild(text);
      group.appendChild(label);
    });
    svg.appendChild(group);
  }
}
function paint() {
  const svg = root.value?.querySelector<SVGSVGElement>('svg');
  if (!svg) return;
  // `paint` also runs when Q changes while the marker stays mounted. Always
  // rebuild from the original geometry instead of expanding the expanded box.
  const original = svg.dataset.originalViewBox || svg.getAttribute('viewBox') || '';
  svg.dataset.originalViewBox = original;
  const box = original.trim().split(/[\s,]+/).map(Number);
  if (box.length === 4 && box.every(Number.isFinite) && box[2] > 0 && box[3] > 0) {
    svg.setAttribute('viewBox', `${box[0]} ${box[1] - 50} ${box[2]} ${box[3] + 50}`);
  }
  svg.querySelectorAll('.party-tour-logos').forEach(group => group.remove());
  for (const region of regions.value) {
    const ids = [...(SELECTORS[region.id] ?? [])];
    if (region.id === 'rest' && !region.aragon) ids.push('aragon');
    if (region.id === 'others' && region.aragon) ids.push('aragon');
    for (const id of ids) {
      const node = svg.querySelector<SVGElement>(`#${id}`);
      if (!node) continue;
      node.style.fill = region.highlighted ? 'var(--paper-2, #f3efe4)' : region.viewed || region.revealedByRest ? 'var(--paper-3, #eee9db)' : 'var(--paper-4, #ded8c6)';
      node.style.stroke = region.highlighted ? 'var(--ink, #2e2a22)' : '#b9b3a7';
      node.style.strokeWidth = region.highlighted ? '1.5' : '.5';
    }
    if (region.viewed) addParties(svg, region.id, region.parties);
  }
}
watch([markup, regions], async () => { await nextTick(); paint(); }, { immediate: true });
</script>

<template>
  <div ref="root" class="congreso-party-tour" data-test="congreso-party-tour">
    <div v-if="markup" class="map-svg" v-html="markup" />
    <p v-else-if="failed">Map unavailable</p>
  </div>
</template>

<style scoped>
.congreso-party-tour { width: 100%; }
.map-svg :deep(svg) { width: 100%; height: auto; display: block; }
</style>
