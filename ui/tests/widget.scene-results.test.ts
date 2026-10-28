import { beforeEach, describe, expect, it, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { i18n } from '../src/i18n';
import { useGameStore } from '../src/stores/game';
import { gameLib } from '../src/game-bindings';
import ElectionResults from '../src/components/viz/ElectionResults.vue';
import ResultBars from '../src/components/viz/ResultBars.vue';
import StackedBreakdown from '../src/components/viz/StackedBreakdown.vue';
import CabinetProposal from '../src/components/viz/CabinetProposal.vue';
import TensionRows from '../src/components/viz/TensionRows.vue';
import type { SceneBarSpec } from '../../source/lib';

const read = (path: string) => readFileSync(resolve(__dirname, '../..', path), 'utf8');
const shell = window as unknown as { RTI_GAME_LIB: typeof gameLib; mountWidgets: (root: Element, q: object) => void; dendryUI: unknown };
function classic(name: string, props: object, q: object) {
  const root = document.createElement('main');
  const marker = document.createElement('div');
  marker.setAttribute('data-widget', name);
  marker.setAttribute('data-props', JSON.stringify(props));
  root.append(marker);
  shell.mountWidgets(root, q);
  return root;
}
function sceneProps(path: string, name: string): { rows: SceneBarSpec[] }[] {
  return [...read(path).matchAll(new RegExp(`data-widget="${name}" data-props='([^']+)'`, 'g'))].map(m => JSON.parse(m[1]));
}
beforeEach(() => {
  setActivePinia(createPinia());
  const game = useGameStore();
  game.initFromText(read('out/game.json'));
  game.newGame();
  i18n.global.locale.value = 'en';
  vi.stubGlobal('RTI_GAME_LIB', gameLib);
  vi.stubGlobal('ClassicI18n', { t: (key: string, fallback: string) => fallback || key, content: (value: string) => value });
  vi.stubGlobal('dendryUI', { dendryEngine: { qdisplay: (v: unknown, id: string) => game.adapter!.qdisplay(v, id) } });
  window.eval(read('out/html/widgets.js'));
});

describe('scene result widgets in both interfaces', () => {
  it('sorts visible rows by seats, then votes; retains zero-seat rows and numeric/new changes without mutating state', () => {
    const q = { erc_parlament_showviz: 1, cup_parlament_showviz: 1, psc_parlament_showviz: 1, ciu_parlament_showviz: 0,
      erc_parlament_s: 20, erc_parlament_pv: 20, erc_parlament_s_old: 19, erc_parlament_pv_old: 21.5,
      psc_parlament_s: 20, psc_parlament_pv: 25, psc_parlament_s_old: 21, psc_parlament_pv_old: 22,
      cup_parlament_s: 0, cup_parlament_pv: 2, cup_parlament_s_old: 0, ciu_parlament_s: 90 };
    const before = JSON.stringify(q);
    const props = { layout: 'parlament', q };
    const vue = mount(ElectionResults, { props, global: { plugins: [i18n] } });
    const old = classic('election-results', props, q);
    for (const root of [vue.element, old]) {
      expect([...root.querySelectorAll('tbody tr')].map(r => r.getAttribute('data-party'))).toEqual(['psc_parlament', 'erc_parlament', 'cup_parlament']);
      expect(root.textContent).toContain('New!');
      expect(root.querySelector('tbody i')?.getAttribute('style')).toContain('var(--psc)');
    }
    expect(vue.text()).toContain('-1.50%');
    const loss = old.querySelector('[data-party="erc_parlament"] td:nth-child(3)');
    expect(loss?.textContent).toContain('1.50%');
    expect(loss?.querySelector('img')?.getAttribute('src')).toBe('img/arrowdown.png');
    expect(old.querySelector('[data-party="cup_parlament"] em')?.textContent).toBe('New!');
    expect(JSON.stringify(q)).toBe(before);
  });

  it('preserves the authored Catalonia subcolumn and Barcelona visibility key', () => {
    const rows = gameLib.electionResultRows({ erc_congreso_showviz: 1, pp_congreso_showviz: 1, erc_congreso_pv_cat: 17 }, 'congreso');
    expect(rows.find(r => r.id === 'erc_congreso')?.catalonia).toBe(17);
    const lowShare = { psoe_congreso_showviz: 1, psoe_congreso_pv_cat: 0.1 };
    expect(gameLib.electionResultRows(lowShare, 'congreso')[0].catalonia).toBeNull();
    expect(gameLib.electionResultRows(lowShare, 'simulation_congreso')[0].catalonia).toBe(0.1);
    expect(gameLib.electionResultRows({ ...lowShare, psoe_congreso_pv_cat: 0.2 }, 'congreso')[0].catalonia).toBe(0.2);
    const props = { layout: 'congreso', voteHeading: 'Popular Vote (Cat.)', q: lowShare };
    for (const root of [mount(ElectionResults, { props, global: { plugins: [i18n] } }).element, classic('election-results', props, lowShare)]) {
      expect(root.querySelector('thead')?.textContent).toContain('Popular Vote (Cat.)');
    }
    // This misspelt key is pre-existing authoring, not a replacement display policy.
    expect(gameLib.electionResultRows({ jxsi_local_barcelona_showviz: 1 }, 'local_barcelona')).toEqual([]);
    expect(gameLib.electionResultRows({ jxsi_bacelona_local_showviz: 1 }, 'local_barcelona')).toHaveLength(1);
  });

  it('renders the inverted assembly question independently of consultation data, including small shares and ties', () => {
    const props = sceneProps('source/scenes/events/elections/parlament_coalition_sovereignty_assembly.scene.dry', 'result-bars')[0];
    for (const [count, share] of [[3000, 99], [1515, 50], [1676, 55.3]]) {
      const q = { parlament_coalition_sovereignty_assembly_yes: count, parlament_coalition_sovereignty_assembly_yes_pc: share, consultation_yes_pc: 0 };
      expect(gameLib.sceneResultBars(q, props.rows).map(r => [r.label, r.value, r.share])).toEqual([['YES', 3030 - count, Number((100 - share).toFixed(2))], ['NO', count, share]]);
      const vue = mount(ResultBars, { props: { ...props, q }, global: { plugins: [i18n] } });
      const old = classic('result-bars', props, q);
      for (const root of [vue.element, old]) {
        expect(root.textContent).toContain(String(3030 - count));
        expect(root.textContent).toContain(String(count));
      }
    }
  });

  it('keeps all three consultation answers, formatted counts, turnout and small positive holdings readable', () => {
    const props = sceneProps('source/scenes/events/referendum/nonbinding_consultation.scene.dry', 'result-bars')[1];
    const q = { consultation_yes: '1.234.567', consultation_yes_pc: 80, consultation_yesno: '100', consultation_yesno_pc: 1, consultation_no: '200', consultation_no_pc: 19, consultation_participation_pc: 35 };
    expect(gameLib.sceneResultBars(q, props.rows)).toHaveLength(4);
    for (const root of [mount(ResultBars, { props: { ...props, q }, global: { plugins: [i18n] } }).element, classic('result-bars', props, q)]) {
      expect(root.textContent).toContain('1.234.567');
      expect(root.textContent).toContain('YES/NO');
      expect(root.textContent).toContain('35%');
    }
    const holdings = sceneProps('source/scenes/events/local_elections/local_elections.scene.dry', 'stacked-breakdown')[0];
    for (const party of ['icv', 'csqp', 'cecp', 'ecp']) {
      const row = gameLib.sceneResultBars({ comuns_local_rb_pct: 20, parlament_current_icv: party }, holdings.rows)[0];
      expect(row.label).toBe(party);
      expect(row.colour).toBe(`var(--${party})`);
    }
    const stocks = { psc_local_rb_pct: 95, erc_local_rb_pct: 1, cs_local_rb_pct: 2, others_local_rb_pct: 2 };
    expect(gameLib.sceneResultBars(stocks, holdings.rows, holdings.groupBelow).map(row => [row.label, row.share])).toEqual([['psc', 95], ['cs', 2], ['others', 3]]);
    for (const root of [mount(StackedBreakdown, { props: { ...holdings, q: stocks }, global: { plugins: [i18n] } }).element, classic('stacked-breakdown', holdings, stocks)]) {
      expect(root.textContent).toContain('3%');
      expect(root.textContent).toContain('95%');
    }
  });

  it('uses proposed cabinet people and gendered offices, not the incumbent cabinet', () => {
    const q = { president: 'Incumbent', parlament_coalition_proposal_president_name: 'Neus Munté', parlament_coalition_proposal_president_gender: 'f', parlament_coalition_proposal_president: 'ciu', parlament_coalition_proposal_vp_name: 'Marta Rovira', parlament_coalition_proposal_vp_gender: 'f', parlament_coalition_proposal_vp: 'erc' };
    for (const root of [mount(CabinetProposal, { props: { q }, global: { plugins: [i18n] } }).element, classic('cabinet-proposal', {}, q)]) {
      expect(root.textContent).toContain('M.H. Presidenta');
      expect(root.textContent).toContain('H. Vicepresidenta');
      expect(root.textContent).toContain('Neus Munté');
      expect(root.textContent).not.toContain('Incumbent');
      expect(root.textContent).not.toContain('widgets.scene.');
      expect(root.querySelectorAll('img')).toHaveLength(8);
      expect(root.querySelectorAll('img[src$="cat_seal.svg"]')).toHaveLength(1);
    }
  });

  it('reuses faction qdisplay classifications in both shells', () => {
    const q = { player_party: 'erc', erc_left_strength: 60, erc_left_dissent: 2 };
    const rows = (gameLib as any).brief.factions(q);
    for (const root of [mount(TensionRows, { props: { rows, q }, global: { plugins: [i18n] } }).element, classic('tension-rows', { deriveFrom: 'factions' }, q)]) {
      expect(root.textContent).toContain('very high');
      expect(root.textContent).toContain('very low');
    }
  });
});
