import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { flushPromises, mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import { nextTick } from 'vue';
import { i18n, setLocale } from '../src/i18n';
import { useGameStore } from '../src/stores/game';
import ElectionHistory from '../src/components/viz/ElectionHistory.vue';
import Prose from '../src/components/Prose.vue';
import caUi from '../../source/locales/ca/ui.json';

const root = resolve(import.meta.dirname, '../..');
const read = (file: string) => readFileSync(resolve(root, file), 'utf8');
const gameText = read('out/game.json');
const records = [
  { date: '2012-08-07T00:00:00.000Z', ciu: 25, erc: 8, jxsi: 0 },
  { date: new Date('2012-09-07T00:00:00.000Z'), ciu: 24, erc: 10, jxsi: 0 },
  { date: '2015-09-07T00:00:00.000Z', ciu: 0, erc: null, jxsi: 30 },
  { date: '2016-09-07T00:00:00.000Z', ciu: 12, erc: 15, jxsi: 0 },
];
const parties = ['ciu', 'erc', 'jxsi', 'absent'];
const chartProps = {
  recordsFrom: 'history', partiesFrom: 'parties',
  partyNames: { ciu: 'CiU', erc: 'ERC', jxsi: 'JxSí' },
  partyColours: { ciu: 'var(--ciu)', erc: 'var(--erc)', jxsi: 'var(--jxsi)' },
  eventsFrom: 'events',
};
beforeEach(() => {
  setActivePinia(createPinia());
  useGameStore().initFromText(gameText);
  setLocale('en');
});
afterEach(() => { setLocale('en'); vi.restoreAllMocks(); vi.unstubAllGlobals(); document.body.innerHTML = ''; });
function render(q: Record<string, unknown> = { history: records, parties }) {
  return mount(ElectionHistory, { props: { ...chartProps, q }, global: { plugins: [i18n] } });
}

describe('election history chart', () => {
  it('uses saved electorate percentages and real elapsed dates, with gaps between party lifetimes', () => {
    const w = render();
    expect(w.findAll('.history-line'), w.html()).toHaveLength(3);
    const path = w.get('g[data-party="ciu"] .history-line');
    expect(path.attributes('d')?.match(/M/g)).toHaveLength(2);
    expect(w.get('g[data-party="ciu"] .history-line-hit').attributes('d')).toBe(path.attributes('d'));
    expect(path.attributes('stroke')).toBe('var(--ciu)');
    const dots = w.findAll('g[data-party="ciu"] circle');
    expect(dots).toHaveLength(1);
    expect(dots[0].attributes('cx')).toBe('480');
    const positions = [...path.attributes('d')!.matchAll(/[ML]([\d.]+),/g)].map(match => Number(match[1]));
    expect(positions[1] - positions[0]).toBeLessThan((positions[2] - positions[0]) / 20);
    expect(w.findAll('.history-point')).toHaveLength(3);
    expect(w.findAll('.history-axis line')).toHaveLength(1);
    expect(w.find('.history-label-leader').exists()).toBe(false);
    expect(w.get('button[data-party="ciu"]').attributes('aria-label')).toContain('12%');
    expect(w.get('.history-unit').text()).toContain('electorate');
    expect(w.find('[data-term="jxs_"]').exists()).toBe(true);
    expect(w.find('.history-legend').exists()).toBe(false);
    expect(w.findAll('.history-plot > .history-line-label')).toHaveLength(3);
    expect(w.find('g[data-party="jxsi"] circle').exists()).toBe(true);
    const baseline = Number(w.get('svg').attributes('viewBox')!.split(' ')[3]) - 45;
    expect(w.findAll('.history-point').every(point => Number(point.attributes('cy')) < baseline)).toBe(true);
    expect(w.html()).not.toMatch(/NaN|Infinity|\[object Object\]/);
    w.unmount();
  });

  it('supports keyboard highlighting without mutating data and keeps two charts independent', async () => {
    const q = { history: records, parties };
    const before = JSON.stringify(q);
    const a = render(q), b = render(q);
    await a.get('button[data-party="erc"]').trigger('focus');
    expect(a.get('g[data-party="ciu"]').attributes('opacity')).toBe('0.15');
    expect(b.get('g[data-party="ciu"]').attributes('opacity')).toBe('1');
    await a.get('button[data-party="erc"]').trigger('blur');
    expect(a.get('g[data-party="ciu"]').attributes('opacity')).toBe('1');
    expect(JSON.stringify(q)).toBe(before);
    a.unmount(); b.unmount();
  });

  it('uses glossary colours and tooltips on legend names while line hover only highlights', async () => {
    const data = JSON.parse(gameText);
    const term = data.data.glossary.terms.find((entry: { id: string }) => entry.id === 'erc');
    term.colour = '#126b4a';
    useGameStore().initFromText(JSON.stringify(data));
    const w = render();
    expect(w.get('.history-line-label[data-party="erc"]').attributes('style')).toContain('rgb(18, 107, 74)');
    await w.get('g[data-party="erc"]').trigger('mouseenter', { clientX: 100, clientY: 100 });
    await flushPromises();
    expect(document.querySelector('[data-test="glossary-popover"]')).toBeNull();
    expect(w.get('g[data-party="erc"] .history-line').attributes('stroke-width')).toBe('3.5');
    expect(w.find('g[data-party="erc"] title').exists()).toBe(false);
    await w.get('g[data-party="erc"]').trigger('mouseleave');
    const label = w.get('.history-line-label[data-party="erc"] [data-term="erc"]');
    await label.trigger('mouseover');
    await flushPromises();
    const tips = document.querySelectorAll('[data-test="glossary-popover"]');
    expect(tips).toHaveLength(1);
    expect(tips[0].textContent).toContain(term.tooltip.title);
    await label.trigger('mouseout');
    expect(document.querySelector('[data-test="glossary-popover"]')).toBeNull();
    w.unmount();
  });

  it('shows explicit Rubicon weeks and month-only captions without metadata', async () => {
    const history = [
      { date: '2017-09-07', week: 1, erc: 12 },
      { date: '2017-09-14', week: 2, erc: 13 },
      { date: '2017-10-07', week: null, erc: 14 },
    ];
    const w = render({ history, parties: ['erc'] });
    const svg = w.get('svg');
    const height = Number(svg.attributes('viewBox')!.split(' ')[3]);
    vi.spyOn(svg.element, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 600, height));
    const secondX = Number([...w.get('.history-line').attributes('d')!.matchAll(/[ML]([\d.]+),/g)][1][1]);
    await w.get('.history-hover-surface').trigger('mousemove', { clientX: secondX, clientY: 50 });
    expect(w.findAll('.history-date-guide tspan').map(line => line.text())).toEqual(['Sep. 2017', 'Week 2']);
    setLocale('ca'); await nextTick();
    expect(w.get('.history-date-guide text').text()).toContain('Setmana 2');
    expect(w.findAll('.history-date-guide tspan').map(line => line.text())).toEqual(['Set. 2017', 'Setmana 2']);
    await w.get('.history-hover-surface').trigger('mousemove', { clientX: 480, clientY: 50 });
    expect(w.get('.history-date-guide text').text()).not.toContain('Setmana');
    await w.setProps({ q: { parties: ['erc'], history: history.map(({ week: _week, ...record }) => record) } });
    await w.get('.history-hover-surface').trigger('mousemove', { clientX: secondX, clientY: 50 });
    expect(w.findAll('.history-date-guide tspan').map(line => line.text())).toEqual(['Set. 2017']);
    w.unmount();
  });

  it('shows a dated guide in blank space and highlights a line through its wider hover target', async () => {
    const q = { history: records, parties };
    const before = JSON.stringify(q);
    const w = render(q);
    const svg = w.get('svg');
    const height = Number(svg.attributes('viewBox')!.split(' ')[3]);
    vi.spyOn(svg.element, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 600, height));
    await w.get('.history-hover-surface').trigger('mousemove', { clientX: 261, clientY: 60 });
    expect(w.get('.history-date-guide line').attributes('x1')).toBe(w.get('g[data-party="jxsi"] circle').attributes('cx'));
    expect(w.get('.history-date-guide text').text()).toBe('Sep. 2015');
    expect(w.find('.history-date-guide rect').exists()).toBe(false);
    expect(Number(w.get('.history-date-guide text').attributes('y'))).toBeLessThan(Number(w.get('.history-date-guide line').attributes('y1')));
    expect(w.findAll('.history-guide-point')).toHaveLength(1);
    expect(w.get('.history-guide-point').attributes('data-party')).toBe('jxsi');
    await w.get('.history-hover-surface').trigger('mousemove', { clientX: 470, clientY: 60 });
    expect(w.get('.history-date-guide line').attributes('x1')).toBe('480');
    expect(w.get('.history-date-guide text').text()).toBe('Sep. 2016');
    expect(w.findAll('.history-guide-point').map(point => point.attributes('data-party'))).toEqual(['ciu', 'erc']);
    await w.get('g[data-party="erc"]').trigger('mouseenter');
    await w.get('g[data-party="erc"] .history-line-hit').trigger('mousemove', { clientX: 261, clientY: 60 });
    expect(w.find('.history-date-guide').exists()).toBe(false);
    expect(w.get('g[data-party="erc"] .history-line').attributes('stroke-width')).toBe('3.5');
    expect(w.get('g[data-party="ciu"]').attributes('opacity')).toBe('0.15');
    await w.get('g[data-party="erc"]').trigger('mouseleave');
    await w.get('.history-hover-surface').trigger('mousemove', { clientX: 261, clientY: 60 });
    expect(w.find('.history-date-guide').exists()).toBe(true);
    await svg.trigger('mouseleave');
    expect(w.find('.history-date-guide').exists()).toBe(false);
    expect(JSON.stringify(q)).toBe(before);
    w.unmount();
  });

  it('marks actual event dates, groups simultaneous events and includes an election after the latest sample', async () => {
    const w = render({ history: records, parties, events: [
      { date: '2011-11-20', label: 'Before this history' },
      { date: 'bad-date', label: 'Invalid' },
      { date: '2014-11-07', label: 'Non-binding consultation' },
      { date: '2014-11-07', label: 'Non-binding consultation' },
      { date: '2014-11-07', label: 'Local Election' },
      { date: '2014-12-07', label: 'Congreso Election' },
      { date: '2017-12-21', label: 'Parlament Election' },
    ] });
    expect(w.findAll('.history-event')).toHaveLength(3);
    expect(w.find('.history-events').exists()).toBe(false);
    expect(w.findAll('.history-event text').map(label => label.text())).toEqual([
      'Non-binding consultation Local Election', 'Congreso Election', 'Parlament Election',
    ]);
    const labels = w.findAll('.history-event text');
    expect(w.find('.history-event-leader').exists()).toBe(false);
    for (const marker of w.findAll('.history-event')) {
      expect(marker.findAll('line')).toHaveLength(1);
      expect(marker.get('line').attributes('x1')).toBe(marker.get('line').attributes('x2'));
      const label = marker.get('text');
      expect(Number(label.attributes('x')) + (label.findAll('tspan').length - 1) * 14).toBeLessThan(Number(marker.get('line').attributes('x1')));
    }
    expect(labels.every(label => label.attributes('transform')?.startsWith('rotate(-90 '))).toBe(true);
    expect(labels.map(label => label.findAll('tspan').length)).toEqual([2, 1, 1]);
    expect(labels.every(label => label.attributes('text-anchor') === 'end')).toBe(true);
    expect(labels.every(label => label.attributes('y') === '36')).toBe(true);
    expect(w.get('svg').attributes('viewBox')).toBe('0 0 600 320');
    expect(w.get('.history-event-date').attributes('y1')).toBe('32');
    expect(w.findAll('.history-event-date').at(-1)!.attributes('x1')).toBe('480');
    expect(w.findAll('.history-event-date').at(-1)!.attributes('x2')).toBe('480');
    expect(w.text()).not.toContain('Before this history');
    useGameStore().adapter!.setLocale('ca', JSON.parse(read('source/locales/ca/content.json')));
    setLocale('ca'); await nextTick();
    expect(w.findAll('.history-event text').map(label => label.text()).join(' ')).toContain('Consulta no vinculant');
    expect(w.findAll('.history-event text').map(label => label.text()).join(' ')).toContain('Eleccions al Parlament');
    expect(w.findAll('.history-event text').map(label => label.text()).join(' ')).not.toMatch(/2014|2017/);
    w.unmount();
  });

  it('handles empty, invalid, zero and isolated readings and removes stale series on replacement', async () => {
    const w = render({ parties, history: [
      null, { date: null, ciu: 50 }, { date: 'invalid', ciu: 40 },
      { date: '2012-08-07', ciu: NaN, erc: Infinity, jxsi: 0 },
    ] });
    expect(w.find('svg').exists()).toBe(false);
    expect(w.text()).toContain('No support history');
    await w.setProps({ q: { parties, history: [{ date: '2012-08-07', erc: 8.5 }] } });
    expect(w.findAll('circle')).toHaveLength(1);
    expect(w.html()).not.toMatch(/NaN|Infinity/);
    expect(w.get('circle').attributes('cx')).toBe('261');
    setLocale('ca'); await nextTick();
    expect(w.get('svg').attributes('aria-label')).toBe(caUi.widgets.history.label);
    expect(w.findAll('.history-axis text').map(label => label.text()).join(' ')).not.toContain(' del ');
    expect(w.get('button').attributes('aria-label')).toContain('8,5%');
    await w.setProps({ q: {} });
    expect(w.find('svg').exists()).toBe(false);
    expect(w.text()).toContain('Encara no');
    w.unmount();
  });
});

describe.each([
  ['parlament_election.post_election', 'party_support_records', 'parties'],
  ['congreso_election.post_election', 'party_spa_support_records', 'congreso_parties'],
])('compiled result %s', (sceneId, recordsKey, partiesKey) => {
  it('mounts its authored widget and restores its history without consuming RNG on interaction or locale refresh', async () => {
    const errors = vi.spyOn(console, 'error');
    const game = useGameStore();
    game.newGame();
    Object.assign(game.adapter!.qualities, { [recordsKey]: records, [partiesKey]: parties });
    game.goToScene(sceneId);
    expect(game.adapter!.qualities.election_history_events).toEqual([
      expect.objectContaining({ type: sceneId.startsWith('parlament') ? 'parlament-election' : 'congreso-election' }),
    ]);
    const saved = game.captureState();
    game.restoreState(saved);
    const rootElement = document.createElement('main');
    rootElement.innerHTML = game.frame!.html;
    const marker = rootElement.querySelector('[data-widget="election-history"]')!;
    expect(marker).not.toBeNull();
    expect(JSON.parse(marker.getAttribute('data-props')!)).not.toHaveProperty('eventLabels');
    const baseline = JSON.parse(game.captureState());
    const rng = game.adapter!.engine.random.getState();
    const w = mount(Prose, { props: { html: marker.outerHTML }, global: { plugins: [i18n] } });
    await flushPromises();
    expect(w.findAll('.history-line'), w.html()).toHaveLength(3);
    await w.get('button[data-party="erc"]').trigger('focus');
    setLocale('ca'); await nextTick();
    expect(w.get('svg').attributes('aria-label')).toBe(caUi.widgets.history.label);
    expect(JSON.parse(game.captureState())).toEqual(baseline);
    expect(game.adapter!.engine.random.getState()).toEqual(rng);
    expect(errors).not.toHaveBeenCalled();
    w.unmount();
  });
});

describe.each([
  ['parlament_election.post_election', 'parlament-election'],
  ['congreso_election.post_election', 'congreso-election'],
  ['local_elections.post_election', 'local-election'],
  ['nonbinding_consultation', 'consultation'],
])('recording %s', (sceneId, type) => {
  it('records the playthrough date once and survives save/load and display without replaying it', () => {
    const errors = vi.spyOn(console, 'error');
    const game = useGameStore();
    game.newGame();
    Object.assign(game.adapter!.qualities, { year: 2016, month: 2, week: 2 });
    game.goToScene(sceneId);
    const events = game.adapter!.qualities.election_history_events as Array<{ date: string; type: string }>;
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe(type);
    const date = new Date(events[0].date);
    expect([date.getFullYear(), date.getMonth(), date.getDate()]).toEqual([2016, 1, 14]);
    const saved = game.captureState();
    game.restoreState(saved);
    game.goToScene(sceneId);
    expect(game.adapter!.qualities.election_history_events).toEqual(events);
    expect(errors).not.toHaveBeenCalled();
  });
});

describe('Classic election history', () => {
  it('runs the actual D3 renderer from an authored marker, renders once on refresh and clears empty history', () => {
    // Execute the supported shell's real dependencies, not a D3 mock.
    window.eval(read('out/html/d3.v7.min.js'));
    window.eval(read('out/html/d3-linegraph.js'));
    window.eval(read('out/html/widgets.js'));
    vi.stubGlobal('dendryUI', { game: { data: { glossary: { terms: [{ id: 'erc', match: ['ERC'], display: 'ERC',
      colour: '#126b4a', tooltip: { title: 'Republican Left' } }] } } } });
    new Function(read('out/html/game.js'))();
    document.body.innerHTML = '<main></main>';
    const el = document.createElement('div');
    el.setAttribute('data-widget', 'election-history');
    el.setAttribute('data-props', JSON.stringify(chartProps));
    document.querySelector('main')!.append(el);
    const shell = window as unknown as { mountWidgets: (root: Element, q: object) => void };
    const q = { history: JSON.parse(JSON.stringify(records)), parties, events: [
      { date: '2014-11-07', label: 'Non-binding consultation' },
      { date: '2014-11-07', label: 'Local Election' },
      { date: '2014-11-14', label: 'Parlament Election' },
    ] };
    shell.mountWidgets(document.body, q);
    expect(el.querySelectorAll('svg')).toHaveLength(1);
    expect(el.querySelectorAll('.party-line')).toHaveLength(3);
    expect(el.querySelectorAll('.party-node')).toHaveLength(3);
    expect(Number(el.querySelector('.party-node[series="erc"]')?.getAttribute('cx'))).toBeCloseTo(Number(el.querySelector('.party-label[series="erc"]')?.getAttribute('x')) - 15);
    expect(el.querySelectorAll('.history-event')).toHaveLength(2);
    const partyLabel = el.querySelector<SVGTextElement>('.party-label[series="erc"]')!;
    expect(partyLabel.style.fill).toBe('#126b4a');
    expect(partyLabel.getAttribute('data-term')).toBe('erc');
    expect(partyLabel.classList.contains('mytooltip')).toBe(true);
    expect(el.querySelector('.party-line[series="erc"]')?.hasAttribute('data-term')).toBe(false);
    expect(el.querySelector('.party-line[series="erc"]')?.classList.contains('mytooltip')).toBe(false);
    expect(el.querySelector('.history-events')).toBeNull();
    expect([...el.querySelectorAll('.history-event text')].map(label => label.textContent?.trim())).toEqual([
      'Non-binding consultation Local Election', 'Parlament Election',
    ]);
    const labels = [...el.querySelectorAll('.history-event text')];
    expect(el.querySelector('.history-event-leader')).toBeNull();
    expect([...el.querySelectorAll('.history-event line')].every(line => line.getAttribute('x1') === line.getAttribute('x2'))).toBe(true);
    for (const marker of el.querySelectorAll('.history-event')) {
      const line = marker.querySelector('line')!;
      const label = marker.querySelector('text')!;
      expect(Number(label.getAttribute('x')) + (label.querySelectorAll('tspan').length - 1) * 14).toBeLessThan(Number(line.getAttribute('x1')));
      expect(line.getAttribute('stroke')).toBe('#b8b8b8');
      expect(line.hasAttribute('stroke-dasharray')).toBe(false);
    }
    expect(labels.every(label => label.getAttribute('transform')?.startsWith('rotate(-90 '))).toBe(true);
    expect(labels.map(label => label.querySelectorAll('tspan').length)).toEqual([2, 1]);
    expect(el.querySelector('.history-axis-baseline')?.getAttribute('d')).toContain('H');
    expect(labels.every(label => label.getAttribute('text-anchor') === 'end')).toBe(true);
    expect(labels.every(label => label.getAttribute('y') === '36')).toBe(true);
    expect(el.querySelector('svg')?.getAttribute('viewBox')).toBe('0 0 500 400');
    expect(el.querySelector('.history-event-date')?.getAttribute('y1')).toBe('32');
    expect(el.querySelector('svg')?.namespaceURI).toBe('http://www.w3.org/2000/svg');
    const svg = el.querySelector('svg')!;
    svg.dispatchEvent(new MouseEvent('mousemove', { clientX: 180, clientY: 100, bubbles: true }));
    const guide = el.querySelector<SVGGElement>('.history-date-guide')!;
    expect(guide.style.display).not.toBe('none');
    expect(guide.querySelector('line')?.getAttribute('x1')).not.toBe('180');
    expect(guide.querySelector('text')?.textContent).toBe('Sep. 2012');
    expect(guide.querySelector('rect')).toBeNull();
    expect(Number(guide.querySelector('text')?.getAttribute('y'))).toBeLessThan(Number(guide.querySelector('line')?.getAttribute('y1')));
    expect(guide.querySelectorAll('.history-guide-point')).toHaveLength(2);
    svg.dispatchEvent(new MouseEvent('mousemove', { clientX: 350, clientY: 100, bubbles: true }));
    expect(guide.querySelector('text')?.textContent).toBe('Sep. 2015');
    expect(guide.querySelectorAll('.history-guide-point')).toHaveLength(1);
    expect(guide.querySelector('.history-guide-point')?.getAttribute('data-party')).toBe('jxsi');
    svg.dispatchEvent(new MouseEvent('mouseleave'));
    expect(guide.style.display).toBe('none');
    shell.mountWidgets(document.body, q);
    expect(el.querySelectorAll('.party-line')).toHaveLength(3);
    expect(el.querySelectorAll('svg')).toHaveLength(1);
    expect(el.querySelectorAll('.history-event')).toHaveLength(2);
    expect(el.querySelectorAll('.history-date-guide')).toHaveLength(1);
    shell.mountWidgets(document.body, { ...q, history: [
      { date: '2017-09-07', week: 1, erc: 12 }, { date: '2017-09-14', week: 2, erc: 13 },
    ] });
    el.querySelector('svg')!.dispatchEvent(new MouseEvent('mousemove', { clientX: 479, clientY: 50, bubbles: true }));
    expect([...el.querySelectorAll('.history-date-guide tspan')].map(line => line.textContent)).toEqual(['Sep. 2017', 'Week 2']);
    shell.mountWidgets(document.body, { ...q, history: [
      { date: '2017-09-07', erc: 12 }, { date: '2017-09-14', erc: 13 },
    ] });
    el.querySelector('svg')!.dispatchEvent(new MouseEvent('mousemove', { clientX: 479, clientY: 50, bubbles: true }));
    expect([...el.querySelectorAll('.history-date-guide tspan')].map(line => line.textContent)).toEqual(['Sep. 2017']);
    shell.mountWidgets(document.body, { history: [], parties });
    expect(el.children).toHaveLength(0);
  });
});

it('records weekly and monthly sampling metadata through the real post-event action', () => {
  const errors = vi.spyOn(console, 'error');
  const game = useGameStore();
  game.newGame();
  for (const id of ['root.start', 'root.normal_mode', 'root.esquerra', 'root.esquerra_2']) game.goToScene(id);
  for (let step = 0; game.frame!.effectiveRole !== 'desk' && step < 20; step++) {
    const choice = game.frame!.choices.findIndex(entry => entry.canChoose);
    expect(choice).toBeGreaterThanOrEqual(0);
    game.choose(choice);
  }
  expect(game.frame!.effectiveRole).toBe('desk');
  const q = game.adapter!.qualities;
  Object.assign(q, { rubicon: true, week: 2, month_actions: 1, month_actions_last: 2, historical_mode: false });
  game.goToScene('post_event');
  expect(q.week).toBe(3);
  for (const key of ['party_support_records', 'party_spa_support_records']) {
    expect((q[key] as Array<{ week: number | null }>).at(-1)?.week).toBe(3);
  }
  Object.assign(q, { rubicon: false, month_actions: 1 });
  game.goToScene('post_event');
  for (const key of ['party_support_records', 'party_spa_support_records']) {
    expect((q[key] as Array<{ week: number | null }>).at(-1)?.week).toBeNull();
  }
  expect(errors).not.toHaveBeenCalled();
});
