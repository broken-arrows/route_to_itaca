// DIAGNOSTIC (temporary): no-action runs from Nov 2012 to Dec 2019, recording
// Vox emergence, rest-of-Spain PP/Cs/Vox shares and every Congreso election's
// right-bloc seats and resulting government. Opt-in: RB_RUNS=<n>.
import { it } from 'vitest';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';

const GAME = process.env.RB_GAME || path.join(__dirname, '..', '..', 'out', 'game.json');
const RUNS = Number(process.env.RB_RUNS ?? 0);
const SEED = Number(process.env.RB_SEED ?? 20121125);
const RIGHT = ['pp', 'cs', 'vox', 'nsuma', 'upn'];

function seeded(seed: number) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}
type Frame = ReturnType<DendryAdapter['goToScene']>;
const choosable = (f: Frame) => f.choices.map((c, index) => ({ index, id: c.id, ok: c.canChoose })).filter(c => c.ok);

function restShares(q: Record<string, any>) {
  const parties: string[] = q.congreso_parties_rest || [];
  const total = parties.reduce((s, p) => s + Number(q[p + '_congreso_rest_support'] || 0), 0);
  return Object.fromEntries(['pp', 'cs', 'vox', 'psoe', 'podemos', 'up'].map(p =>
    [p, total > 0 ? Math.round(Number(q[p + '_congreso_rest_support'] || 0) / total * 1000) / 10 : 0]));
}

function run(seed: number) {
  const control = seeded(seed ^ 0xa5a5a5a5);
  Math.random = seeded(seed);
  const adapter = DendryAdapter.fromJSONText(readFileSync(GAME, 'utf8'));
  adapter.beginGame([seed >>> 0, (seed ^ 0x9e3779b9) >>> 0, (seed ^ 0x243f6a88) >>> 0, (seed ^ 0xb7e15162) >>> 0]);
  adapter.goToScene('root.start');
  adapter.goToScene('root.normal_mode');
  adapter.goToScene('root.esquerra');
  let frame = adapter.goToScene('root.esquerra_2');
  for (let g = 0; frame.effectiveRole !== 'desk' && g < 20; g++) frame = adapter.choose(choosable(frame)[0].index);
  frame = adapter.goToScene('election_simulation.2012_11');
  const q = adapter.qualities as Record<string, any>;
  Object.assign(q, { year: 2012, month: 11, week: 1, time: 4, month_actions: 0, month_actions_last: 0,
    next_election_year: 2015, next_election_month: 9, next_election_week: 1, next_election_time: 38 });
  const date = () => `${q.year}-${String(q.month).padStart(2, '0')}`;
  const elections: any[] = [];
  const governments: any[] = [];
  const monthly: any[] = [];
  let voxFrom: string | null = null;
  let ending: string | null = null;
  const path: string[] = [];
  const leftOffer: any[] = [];
  const observe = () => {
    if (q.vox_active && !voxFrom) voxFrom = date();
    const gov = { coalition: [...(q.spanish_coalition || [])], pm: q.spanish_pm_party };
    if (JSON.stringify(governments.at(-1)?.gov) !== JSON.stringify(gov)) governments.push({ date: date(), gov });
    const n = (q.election_history_events || []).filter((e: any) => e.type === 'congreso-election').length;
    if (n > elections.length) {
      const seats = Object.fromEntries((q.congreso_parties as string[]).filter(p => p !== 'abstain')
        .map(p => [p, Number(q[p + '_congreso_s'] || 0)]).filter(([, s]) => (s as number) > 0));
      elections.push({ date: date(), voxActive: !!q.vox_active, seats, right: RIGHT.reduce((s, p) => s + (seats[p] || 0), 0),
        ppCsVox: (seats.pp || 0) + (seats.cs || 0) + (seats.vox || 0), rest: restShares(q),
        rightPv: Math.round(['pp', 'cs', 'vox', 'nsuma', 'upn'].reduce((s, p) => s + Number(q[p + '_congreso_pv'] || 0), 0) * 10) / 10,
        pv: Object.fromEntries(['pp', 'cs', 'vox', 'psoe'].map(p => [p, Math.round(Number(q[p + '_congreso_pv'] || 0) * 10) / 10])) });
    }
  };
  const drain = (f: Frame) => {
    for (let g = 0; g < 400; g++) {
      observe();
      if (f.gameOver) { ending = `${date()} ${f.sceneId}`; return f; }
      if (f.effectiveRole === 'desk') return f;
      if (f.sceneId === 'congreso_coalition_left_offer' && !leftOffer.some(o => o.date === date())) leftOffer.push({ date: date(), erc_s: q.erc_congreso_s, erc_in_jxsi: q.erc_in_jxsi, erc_in_jxcat: q.erc_in_jxcat, erc_roadmap: q.erc_roadmap, erc_civic: q.erc_civic_strength, psc_rel: q.psc_relations, ally: q.parlament_current_icv, ally_rel: q[q.parlament_current_icv + '_relations'], res: q.party_resources, icv_leader: q[q.parlament_current_icv + '_leader'], pod_ideology: q.podemos_ideology, compromis_ever: q.compromis_reached_out_ever, erc_arith: q.congreso_left_offer_erc_arith, comp_arith: q.congreso_left_offer_compromis_arith, erc_possible: q.congreso_left_offer_erc_possible, comp_possible: q.congreso_left_offer_compromis_possible });
      const c = choosable(f);
      if (!c.length) throw new Error(`dead end ${f.sceneId}`);
      let pick = process.env.RB_MADINA ? c.find(x => x.id === 'bye_rubalcaba.madina') : undefined;
      if (!pick) pick = c.find(x => x.id === 'post_event' || x.id === 'root');
      if (!pick) {
        const queue = f.sceneId === 'post_event.events_choice' || f.sceneId === 'post_event.elections_choice';
        pick = queue ? c[0] : c[Math.floor(control() * c.length)];
      }
      if (/^(congreso_|spanish_censure|psoe_primaries)/.test(f.sceneId)) path.push(`${date()} ${f.sceneId} -> ${pick.id} [${c.map(x => x.id.split(".").pop()).join(",")}]`);
      f = adapter.choose(pick.index);
    }
    throw new Error(`guard at ${f.sceneId}`);
  };
  const done: string[] = [];
  for (let m = 0; m < 100 && !ending && (q.year < 2020); m++) {
    if (q.year === 2015 && q.month === 1 && !done.includes('pleb')) {
      const key = `${q.parlament_current_ciu || 'ciu'}_roadmap`;
      if (!Number(q[key])) q[key] = 1;
      frame = drain(adapter.goToScene('plebiscite_election'));
      done.push('pleb');
    }
    const toElection = Number(q.next_election_time) - Number(q.time);
    if (q.unio_split && !q.jxsi_formed && toElection >= 3 && toElection < 8 && !done.includes('jxsi')) {
      frame = drain(adapter.goToScene('jxsi_formation'));
      done.push('jxsi');
    }
    if (ending) break;
    // Emulate an ERC player able to back Madina in the July 2014 primaries.
    if (process.env.RB_MADINA && date() === '2014-06') { q.psc_relations = Math.max(40, Number(q.psc_relations) || 0); q.party_resources = Math.max(1, Number(q.party_resources) || 0); }
    // Early dissolution after the 2016 election: reschedule the next Congreso
    // election to RB_CONGRESO_AT (YYYY-MM), as a dissolution event would.
    const target = process.env.RB_CONGRESO_AT;
    if (target && !done.includes('dissolve') && date() >= '2016-08' && elections.some(e => e.date >= '2016-01')) {
      const [ty, tm] = target.split('-').map(Number);
      const months = (ty - q.year) * 12 + (tm - q.month);
      Object.assign(q, { next_spanish_election_year: ty, next_spanish_election_month: tm, next_spanish_election_week: 1,
        next_spanish_election_time: Number(q.time) + months });
      done.push('dissolve');
    }
    // Confrontation variant: hold Catalan-Spanish relations at RB_RELATIONS after the dissolution.
    if (process.env.RB_RELATIONS && (done.includes('dissolve') || (process.env.RB_RELATIONS_FROM && date() >= process.env.RB_RELATIONS_FROM))) q.cat_spa_relations = Number(process.env.RB_RELATIONS);
    if (process.env.RB_VOX_AT === date() && !q.vox_active) {
      monthly.push({ date: date(), label: 'before-vox', ...restShares(q) });
      frame = drain(adapter.goToScene('vox_emergence'));
      monthly.push({ date: date(), label: 'after-vox', ...restShares(q) });
    }
    frame = drain(adapter.goToScene('debug_card.pass_time'));
    if (q.month === 6 || q.month === 12) monthly.push({ date: date(), vox: !!q.vox_active, ...restShares(q) });
  }
  return { seed, voxFrom, ending, end: date(), elections, governments, monthly, path, leftOffer };
}

it.skipIf(!(RUNS > 0))('right-bloc diagnostic', async () => {
  const origRandom = Math.random, origLog = console.log, origInfo = console.info, origErr = console.error;
  console.log = console.info = () => {};
  console.error = (...a: unknown[]) => { const m = a.map(String).join(' '); if (/Scene (action|predicate|expression) error/.test(m)) { if (process.env.RB_TOLERATE) { swallowed.push(m.slice(0, 80)); return; } throw new Error(m); } };
  const results: any[] = [];
  const swallowed: string[] = [];
  try {
    for (let i = 0; i < RUNS; i++) {
      try { results.push(run(SEED + i)); } catch (e) { results.push({ seed: SEED + i, error: String(e).slice(0, 300) }); }
      if (i % 5 === 4) await new Promise(r => setTimeout(r, 0));
    }
  } finally { Math.random = origRandom; console.log = origLog; console.info = origInfo; console.error = origErr; }
  writeFileSync(path.resolve(process.env.RB_OUTPUT || 'rb-diag.json'), JSON.stringify(results.map(r => ({ ...r, swallowedErrors: swallowed.length })), null, 1));
}, 3_600_000);
