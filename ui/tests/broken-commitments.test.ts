import { describe, it, expect, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';

const game = readFileSync('../out/game.json', 'utf8');
function setup() {
  const a = DendryAdapter.fromJSONText(game);
  a.beginGame([1, 2, 3, 4]);
  a.goToScene('root.start');
  return { a, q: a.qualities as any };
}
const lib = gameLib as any;

describe('Broken commitments', () => {
  it('settles the authored burden rate without blanket recipient scaling', () => {
    const { q } = setup();
    Object.assign(q, { burdens_broken: true, broken_roadmap: false, independence_trust: 80,
      cat_coalition: ['erc'], cat_coalition_support: [], cat_coalition_abstain: [], cat_caretaker_gov: false,
      parlament_vote_trace_enabled: true, parlament_vote_trace: { ticks: 0, mechanisms: {} } });
    const opening = JSON.parse(JSON.stringify(q));
    lib.engineTick(q);
    let stock = 0, population = 0;
    for (const c of q.parlament_constituencies) for (const d of q.parlament_demographics) {
      const n = q[`parlament_${c}_${d}_pop`];
      stock += n * opening[`erc_parlament_${c}_${d}_support`]; population += n;
    }
    const flows = q.parlament_vote_trace.mechanisms['nonlinear.broken_burdens'];
    expect(flows.il).toBeCloseTo(-0.006 * stock / population, 10);
    expect(Object.values(flows).reduce((s: number, v) => s + Number(v), 0)).toBeCloseTo(0, 10);
  });
  it.each([['jxsi', false], ['jxcat', false], ['jxsi', true], ['jxcat', true]] as const)('roadmap failure reaches the live %s Parlament ballot (CUP inside: %s)', (list, cupInside) => {
    const { a, q } = setup();
    Object.assign(q, { player_party: 'erc', parlament_current_ciu: list === 'jxsi' ? 'cdc' : 'jxcat',
      erc_in_jxsi: list === 'jxsi', erc_in_jxcat: list === 'jxcat',
      cup_in_jxsi: list === 'jxsi' && cupInside, cup_in_jxcat: list === 'jxcat' && cupInside });
    const c = q.parlament_constituencies[0], d = q.parlament_demographics[0];
    const key = (p: string) => `${p}_parlament_${c}_${d}_support`;
    for (const p of ['erc', 'cdc', 'jxsi', 'jxcat', 'cup', 'fnc']) q[key(p)] = 0;
    q[key(list)] = 40;
    q[key('abstain')] = 30;
    a.goToScene('broken_roadmap');
    expect(q[key(list)]).toBeCloseTo(35.2, 10);
    expect(q[key('erc')]).toBe(0);
    expect(q[key('cup')]).toBeCloseTo(cupInside ? 0 : 0.72, 10);
    expect(q[key('fnc')]).toBeCloseTo(cupInside ? 0.48 : 0.24, 10);
    expect(q[key('abstain')]).toBeCloseTo(cupInside ? 34.32 : 33.84, 10);
    expect(['erc', 'cdc', 'jxsi', 'jxcat', 'cup', 'fnc', 'abstain'].reduce((s,p)=>s+q[key(p)],0)).toBeCloseTo(70, 10);
  });

  it('preserves standalone roadmap rates despite obsolete joint-list membership flags', () => {
    const { a, q } = setup();
    Object.assign(q, { player_party: 'erc', parlament_current_ciu: 'cdc',
      erc_in_jxsi: true, cup_in_jxsi: true, erc_in_jxcat: false, cup_in_jxcat: false });
    const c = q.parlament_constituencies[0], d = q.parlament_demographics[0];
    const key = (p: string) => `${p}_parlament_${c}_${d}_support`;
    for (const p of ['jxsi', 'jxcat', 'cup', 'fnc', 'abstain']) q[key(p)] = 0;
    q[key('erc')] = 20; q[key('cdc')] = 30;
    a.goToScene('broken_roadmap');
    expect(q[key('erc')]).toBeCloseTo(17.6);
    expect(q[key('cdc')]).toBeCloseTo(27.3);
    expect(q[key('cup')]).toBeCloseTo(0.765);
    expect(['erc', 'cdc', 'cup', 'fnc', 'abstain'].reduce((s,p)=>s+q[key(p)],0)).toBeCloseTo(50);
  });

  it('charges the current Congreso successor rather than a stale joint-list residue', () => {
    const { q } = setup();
    Object.assign(q, { cat_coalition: ['dl'], cat_coalition_support: [], cat_coalition_abstain: [],
      parlament_current_ciu: 'dl', dl_congreso_catalunya_support: 20,
      ciu_congreso_catalunya_support: 0, jxsi_congreso_catalunya_support: 0.4,
      game_random_state: 123 });
    q.congreso_parties_catalunya = [...new Set([...q.congreso_parties_catalunya, 'dl', 'jxsi'])];
    const good = JSON.parse(JSON.stringify(q)), bad = JSON.parse(JSON.stringify(q));
    good.burdens_broken = false; bad.burdens_broken = true;
    lib.engineTick(good); lib.engineTick(bad);
    expect(bad.dl_congreso_catalunya_support).toBeLessThan(good.dl_congreso_catalunya_support);
    expect(bad.jxsi_congreso_catalunya_support).toBeCloseTo(good.jxsi_congreso_catalunya_support, 10);
  });

  it('counts recovery down, exposes its event, and starts a fresh cycle after recovery', () => {
    const { a, q } = setup();
    Object.assign(q, { player_party: 'erc', year: 2013, month: 3, week: 1,
      burdens_goal: 4, burdens_done: 0, burdens_time: 0, burdens_broken: true,
      burdens_restart_countdown: 3, roadmap_year: 0 });
    if (!q.countdowns.includes('burdens_restart')) q.countdowns.push('burdens_restart');
    const advance = () => {
      const before = q.time;
      let frame = a.goToScene('debug_card.pass_time');
      for (let guard = 0; q.time === before && guard < 300; guard++) {
        const choices = frame.choices.map((c,i)=>({...c,index:i})).filter(c=>c.canChoose);
        const choice = choices.find(c=>c.id==='post_event') || choices.find(c=>c.id==='root') || choices[0];
        if (!choice) throw Error(`No continuation at ${frame.sceneId}`);
        frame = a.choose(choice.index);
      }
      expect(q.time).toBe(before + 1);
    };
    advance(); expect(q.burdens_restart_countdown).toBe(2);
    advance(); expect(q.burdens_restart_countdown).toBe(1);
    advance(); expect(q.burdens_restart_trigger).toBe(1);
    const frame = a.goToScene('post_event.events_choice');
    const index = frame.choices.findIndex(c=>c.id==='burdens_start' && c.canChoose);
    expect(index).toBeGreaterThanOrEqual(0);
    a.choose(index);
    expect(q.burdens_broken).toBe(false);
    expect(q.burdens_restart_countdown).toBeUndefined();
    // A later failure, including a burden flag set before the deadline, gets
    // a new countdown rather than inheriting the previous expired timer.
    Object.assign(q, { burdens_broken: true, burdens_time: 0, burdens_done: 0 });
    advance(); expect(q.burdens_restart_countdown).toBe(23);
    advance(); expect(q.burdens_restart_countdown).toBe(22);
  });

  it('the actual roadmap event removes Congreso support from ERC and the live CiU successor', () => {
    const { a, q } = setup();
    Object.assign(q, { player_party: 'erc', parlament_current_ciu: 'dl', erc_congreso_catalunya_support: 15,
      ciu_congreso_catalunya_support: 0, cdc_congreso_catalunya_support: 0,
      jxsi_congreso_catalunya_support: 0.4,
      dl_congreso_catalunya_support: 20, abstain_congreso_catalunya_support: 25 });
    q.congreso_parties_catalunya = [...new Set([...q.congreso_parties_catalunya, 'dl'])];
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      a.goToScene('broken_roadmap');
      expect(q.erc_congreso_catalunya_support).toBe(12);
      expect(q.dl_congreso_catalunya_support).toBe(17);
      expect(q.jxsi_congreso_catalunya_support).toBe(0.4);
      expect(q.abstain_congreso_catalunya_support).toBe(31);
      expect(errors.mock.calls).toEqual([]);
    } finally { errors.mockRestore(); }
  });

  it('bounds compounding, excludes responsible recipients, and stops when burdens clear', () => {
    const { q } = setup();
    Object.assign(q, { burdens_broken: true, cat_caretaker_gov: false,
      cat_coalition: ['erc'], cat_coalition_support: [], cat_coalition_abstain: [],
      independence_trust: 40, broken_roadmap: false });
    const context = { carriers: { il: 'erc', icr: 'ciu', cup: 'cup', fl: 'icv', psc: 'psc', abs: 'abstain' },
      support: { erc: 20 } };
    const loss = () => lib.buildBrokenBurdensTransfers(q, context).reduce((s: number, t: any) => s + t.amount, 0);
    expect(loss()).toBeCloseTo(0.12);
    q.broken_roadmap = true; q.independence_trust = 0;
    expect(loss()).toBeCloseTo(0.24);
    q.cat_coalition = ['ciu']; q.cat_coalition_support = ['erc'];
    expect(lib.buildBrokenBurdensTransfers(q, context).every((t: any) => !['icr','il'].includes(t.to))).toBe(true);
    expect(loss()).toBeCloseTo(0.24 * 0.38);
    q.burdens_broken = false;
    expect(loss()).toBe(0);
  });

  it('does not double-charge a joint list', () => {
    const { q } = setup();
    Object.assign(q, { burdens_broken: true, cat_caretaker_gov: false, independence_trust: 40, broken_roadmap: false });
    const routes = lib.buildBrokenBurdensTransfers(q, {
      carriers: { icr: 'jxsi', il: 'jxsi', cup: 'cup', abs: 'abstain' },
      support: { jxsi: 40 }, responsibility: { icr: 0.5, il: 0.5 },
    });
    expect(routes.reduce((s: number, t: any) => s + t.amount, 0)).toBeCloseTo(0.24);
    expect(routes.every((t: any) => ['cup', 'abs'].includes(t.to))).toBe(true);
  });

  it('reduces responsible support in both contests during a real tick and conserves electorate totals', () => {
    const { q } = setup();
    Object.assign(q, { cat_coalition: ['erc'], cat_coalition_support: [], cat_coalition_abstain: [],
      cat_caretaker_gov: false, game_random_state: 123, broken_roadmap: true, independence_trust: 20 });
    const good = JSON.parse(JSON.stringify(q)), bad = JSON.parse(JSON.stringify(q));
    good.burdens_broken = false; bad.burdens_broken = true;
    lib.engineTick(good); lib.engineTick(bad);
    expect(bad.erc_congreso_catalunya_support).toBeLessThan(good.erc_congreso_catalunya_support);
    for (const c of q.parlament_constituencies) for (const d of q.parlament_demographics) {
      const key = `erc_parlament_${c}_${d}_support`;
      expect(bad[key]).toBeLessThan(good[key]);
      const keys = [...q.parties, 'abstain'].map(p => `${p}_parlament_${c}_${d}_support`);
      expect(keys.reduce((s,k)=>s+bad[k],0)).toBeCloseTo(keys.reduce((s,k)=>s+good[k],0), 8);
      expect(keys.every(k=>bad[k]>=0)).toBe(true);
    }
  });
});
