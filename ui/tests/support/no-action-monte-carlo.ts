import { readFileSync } from 'node:fs';
import { DendryAdapter } from '../../src/engine/adapter';
import type { Frame } from '../../src/engine/types';

export type Difficulty = 'easy' | 'normal' | 'hard';

export interface NoActionRunOptions {
  gamePath: string;
  seed: number;
  difficulty: Difficulty;
  quiet?: boolean;
  localBarcelonaSupportOverrides?: Record<string, number>;
  localBarcelonaSensitivityOverrides?: Record<string, number[]>;
}

export interface ElectionSnapshot {
  year: number;
  month: number;
  broadSovereigntySeats: number;
  seats: Record<string, number>;
  support: Record<string, number>;
  /** Share of valid votes, as calculated by the real election algorithm. */
  validVoteShare: Record<string, number>;
  families: Record<string, number>;
  familySeats: Record<string, number>;
  familyValidVoteShare: Record<string, number>;
  totalSeats: number;
  totalValidVoteShare: number;
  /** Raw fixture rounding is retained; conserved evolution preserves each cell. */
  cellSupportTotals: Record<string, number>;
}

export interface LocalElectionSnapshot {
  year: number;
  month: number;
  currentCiu: string;
  currentIcv: string;
  barcelonaSeats: Record<string, number>;
  barcelonaValidVoteShare: Record<string, number>;
  barcelonaSupport: Record<string, number>;
  totalBarcelonaSeats: number;
  redBeltScores: Record<string, number>;
  redBeltHoldings: Record<string, number>;
  redBeltWinners: Record<string, string>;
  drivers: Record<string, number>;
  ballotParties: string[];
  unregisteredSupport: Record<string, number>;
}

export interface NoActionRunResult {
  seed: number;
  difficulty: Difficulty;
  baseline: ElectionSnapshot;
  final: ElectionSnapshot;
  majorityRetained: boolean;
  seatDelta: number;
  monthsAdvanced: number;
  forcedEventChoices: number;
  historicalStructuralEvents: string[];
  structuralState: {
    unioSplit: boolean;
    dlFormed: boolean;
    jxsiFormed: boolean;
    cupInJxsi: boolean;
    consultationPending: boolean;
    consultationHappened: boolean;
    referendumPending: boolean;
    referendumHappened: boolean;
  };
  structuralValues: Record<string, number>;
  targetElectionKeys: string[];
  voteTrace: Record<string, Record<string, number>>;
}

export interface NoActionCohort {
  runs: number;
  retentionRate: number;
  meanBroadSovereigntySeats: number;
  meanAbstentionSupport: number;
  meanFamilySupport: Record<string, number>;
  meanFamilySeats: Record<string, number>;
  meanFamilyValidVoteShare: Record<string, number>;
}

export interface NoActionAggregate {
  runs: number;
  retained: number;
  retentionRate: number;
  meanSeatDelta: number;
  meanSupportDelta: Record<string, number>;
  meanValidVoteShareDelta: Record<string, number>;
  meanFinalSeats: Record<string, number>;
  meanFinalValidVoteShare: Record<string, number>;
  meanFamilyDelta: Record<string, number>;
  meanFinalFamilySeats: Record<string, number>;
  meanFinalFamilyValidVoteShare: Record<string, number>;
  meanTotalSeats: number;
  minTotalSeats: number;
  maxTotalSeats: number;
  invalidSeatTotalRuns: number;
  meanTotalValidVoteShare: number;
  minTotalValidVoteShare: number;
  maxTotalValidVoteShare: number;
  invalidValidVoteTotalRuns: number;
  electionDates: Record<string, number>;
  electionDateCohorts: Record<string, NoActionCohort>;
  organizationCohorts: Record<string, NoActionCohort>;
  jxsiCarrierRuns: number;
  jxsiFlagRuns: number;
  structuralStateCounts: Record<string, number>;
  meanStructuralValues: Record<string, number>;
  meanVoteTrace: Record<string, Record<string, number>>;
}

const PARTY_KEYS = [
  'ciu', 'erc', 'cup', 'si', 'cdc', 'unio', 'dl', 'jxsi', 'jxcat', 'junts',
  'pdcat', 'psc', 'icv', 'csqp', 'cecp', 'ecp', 'cs', 'ppc', 'vox', 'fnc', 'pxc',
] as const;

const BCN_PARTY_KEYS = [
  'bcomu', 'cup', 'psc', 'erc', 'ciu', 'cdc', 'dl', 'pdcat', 'jxcat', 'junts', 'jxsi',
  'primaries', 'cs', 'pp', 'icv',
] as const;

const RED_BELT_PARTY_KEYS = [
  'psc', 'cs', 'erc', 'cup', 'comuns', 'pp', 'ciu', 'cdc', 'dl', 'jxcat',
  'jxsi', 'pdcat', 'junts',
] as const;

const RED_BELT_LOCATIONS = [
  'lhospitalet-de-llobregat', 'sant-boi-de-llobregat', 'el-prat-de-llobregat',
  'viladecans', 'esplugues-de-llobregat', 'ripollet', 'sant-adria-de-besos',
  'terrassa', 'sabadell', 'cornella-de-llobregat', 'rubi', 'granollers',
  'mollet-del-valles', 'mataro', 'badalona', 'santacolomadegramenet', 'balaguer',
  'sant-vicenc-dels-horts', 'martorell', 'vilafranca-del-penedes',
] as const;

const BROAD_SOVEREIGNTY = [
  'ciu', 'erc', 'cup', 'si', 'cdc', 'dl', 'jxsi', 'jxcat', 'junts', 'pdcat', 'fnc',
] as const;

const DIAGNOSTIC_FAMILIES: Record<string, readonly string[]> = {
  sovereignty: BROAD_SOVEREIGNTY,
  sovereignty_mainstream: ['ciu', 'erc', 'si', 'cdc', 'dl', 'jxsi', 'jxcat', 'junts', 'pdcat'],
  ciu_successors_outside_joint_lists: ['ciu', 'cdc', 'dl', 'pdcat', 'junts'],
  erc_outside_joint_lists: ['erc'],
  cup_outside_joint_lists: ['cup'],
  sovereignty_joint_lists: ['jxsi', 'jxcat'],
  federal_left: ['icv', 'csqp', 'cecp', 'ecp'],
  psc: ['psc'],
  cs: ['cs'],
  ppc: ['ppc'],
  fnc_pxc: ['fnc', 'pxc'],
  abstention: ['abstain'],
};

const START_SCENES: Record<Difficulty, string> = {
  easy: 'root.easy_mode',
  normal: 'root.normal_mode',
  hard: 'root.hard_mode',
};

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function numberQuality(q: Record<string, unknown>, key: string): number {
  const value = Number(q[key] ?? 0);
  if (!Number.isFinite(value)) throw new Error(`Non-finite quality ${key}: ${String(q[key])}`);
  return value;
}

function weightedSupport(q: Record<string, unknown>, party: string): number {
  const constituencies = q.parlament_constituencies as string[];
  const demographics = q.parlament_demographics as string[];
  let votes = 0;
  let population = 0;
  for (const constituency of constituencies) {
    for (const demographic of demographics) {
      const pop = numberQuality(q, `parlament_${constituency}_${demographic}_pop`);
      votes += numberQuality(q, `${party}_parlament_${constituency}_${demographic}_support`) * pop / 100;
      population += pop;
    }
  }
  return population === 0 ? 0 : votes / population * 100;
}

function snapshot(q: Record<string, unknown>): ElectionSnapshot {
  const support: Record<string, number> = {};
  const validVoteShare: Record<string, number> = {};
  const seats: Record<string, number> = {};
  for (const party of PARTY_KEYS) {
    support[party] = weightedSupport(q, party);
    validVoteShare[party] = numberQuality(q, `${party}_parlament_pv`);
    seats[party] = numberQuality(q, `${party}_parlament_s`);
  }
  support.abstain = weightedSupport(q, 'abstain');

  const families: Record<string, number> = {};
  const familySeats: Record<string, number> = {};
  const familyValidVoteShare: Record<string, number> = {};
  for (const [family, members] of Object.entries(DIAGNOSTIC_FAMILIES)) {
    families[family] = members.reduce((sum, party) => sum + support[party], 0);
    familySeats[family] = members.reduce((sum, party) => sum + (seats[party] ?? 0), 0);
    familyValidVoteShare[family] = members.reduce(
      (sum, party) => sum + (validVoteShare[party] ?? 0),
      0,
    );
  }
  const broadSovereigntySeats = BROAD_SOVEREIGNTY.reduce(
    (sum, party) => sum + seats[party],
    0,
  );
  return {
    year: numberQuality(q, 'year'),
    month: numberQuality(q, 'month'),
    broadSovereigntySeats,
    seats,
    support,
    validVoteShare,
    families,
    familySeats,
    familyValidVoteShare,
    totalSeats: Object.values(seats).reduce((sum, value) => sum + value, 0),
    totalValidVoteShare: Object.values(validVoteShare).reduce((sum, value) => sum + value, 0),
    cellSupportTotals: Object.fromEntries((q.parlament_constituencies as string[]).flatMap((province) =>
      (q.parlament_demographics as string[]).map((demo) => [`${province}.${demo}`,
        [...PARTY_KEYS, 'abstain'].reduce((sum, party) => sum + numberQuality(q, `${party}_parlament_${province}_${demo}_support`), 0),
      ]),
    )),
  };
}

function localSnapshot(q: Record<string, unknown>): LocalElectionSnapshot {
  const values = (suffix: string, parties: readonly string[]) => Object.fromEntries(
    parties.map((party) => [party, numberQuality(q, `${party}${suffix}`)]),
  );
  const barcelonaSeats = values('_local_barcelona_s', BCN_PARTY_KEYS);
  const ballotParties = [...(q.parties_bcn as string[])];
  const potentiallyUnregistered = ['pdcat', 'unio', 'fnc', 'pxc', 'vox'];
  return {
    year: numberQuality(q, 'year'),
    month: numberQuality(q, 'month'),
    currentCiu: String(q.parlament_current_ciu || 'ciu'),
    currentIcv: String(q.parlament_current_icv || 'icv'),
    barcelonaSeats,
    barcelonaValidVoteShare: values('_local_barcelona_pv', BCN_PARTY_KEYS),
    barcelonaSupport: values('_local_barcelona_support', BCN_PARTY_KEYS),
    totalBarcelonaSeats: Object.values(barcelonaSeats).reduce((sum, seats) => sum + seats, 0),
    redBeltScores: Object.fromEntries(['cs', 'erc', 'comuns', 'cup'].map((party) =>
      [party, numberQuality(q, `${party}_local_redbelt`)])),
    redBeltHoldings: values('_local_rb_holdings', RED_BELT_PARTY_KEYS),
    redBeltWinners: Object.fromEntries(RED_BELT_LOCATIONS.map((location) =>
      [location, String(q[`local_${location}_wp`] ?? '')])),
    drivers: Object.fromEntries([
      'independence_movement', 'independence_trust', 'social_dissent',
      'welfare_index', 'cat_spa_relations', 'podemos_channeling',
      'cs_redbelt_mod', 'erc_redbelt_mod', 'comuns_redbelt_mod', 'cup_redbelt_mod',
      'psc_redbelt_mod',
    ].map((key) => [key, numberQuality(q, key)])),
    ballotParties,
    unregisteredSupport: Object.fromEntries(
      potentiallyUnregistered.filter((party) => !ballotParties.includes(party)).map((party) =>
        [party, numberQuality(q, `${party}_local_barcelona_support`)]),
    ),
  };
}

function readVoteTrace(q: Record<string, unknown>): Record<string, Record<string, number>> {
  const trace = q.parlament_vote_trace as {
    mechanisms?: Record<string, Record<string, unknown>>;
  } | undefined;
  const result: Record<string, Record<string, number>> = {};
  for (const [mechanism, targets] of Object.entries(trace?.mechanisms ?? {})) {
    result[mechanism] = Object.fromEntries(
      Object.entries(targets)
        .map(([target, value]) => [target, Number(value)] as const)
        .filter(([, value]) => Number.isFinite(value)),
    );
  }
  return result;
}

function choosable(frame: Frame): Array<{ index: number; id: string }> {
  return frame.choices
    .map((choice, index) => ({ index, id: choice.id, canChoose: choice.canChoose }))
    .filter((choice) => choice.canChoose)
    .map(({ index, id }) => ({ index, id }));
}

/** Resolve mandatory event/election pages. The no-action policy never draws or
 * plays a desk card. If content explicitly offers an exit (`post_event` or
 * `root`), it takes that no-intervention route. Otherwise a mandatory branch
 * is selected by the seeded control RNG. */
function drainMandatoryFlow(
  adapter: DendryAdapter,
  frame: Frame,
  random: () => number,
  targetElectionKeys: Set<string>,
  stopAtParlament = true,
  enforceNoTimeAdvance = true,
): { frame: Frame; forcedChoices: number; reachedTarget: boolean } {
  let current = frame;
  let forcedChoices = 0;
  for (let guard = 0; guard < 300; guard++) {
    const q = adapter.qualities;
    // post_election has an unconditional go-to, so Dendry can execute its
    // arrival actions and redirect through coalitions before exposing a frame.
    // Its schedule assignment is therefore the reliable completion marker.
    const completedParlamentElection = Number(q.year) > 2012
      && Number(q.next_election_year) === Number(q.year) + 4;
    if (completedParlamentElection && stopAtParlament) {
      const key = `${q.year}-${q.month}`;
      if (targetElectionKeys.has(key)) {
        throw new Error(`Parlament election entry replayed at ${key}`);
      }
      targetElectionKeys.add(key);
      return { frame: current, forcedChoices, reachedTarget: true };
    }
    if (current.effectiveRole === 'desk') {
      return { frame: current, forcedChoices, reachedTarget: false };
    }

    const choices = choosable(current);
    if (choices.length === 0) {
      throw new Error(`Mandatory flow dead-ended at ${current.sceneId}`);
    }
    let selected = choices.find((choice) => choice.id === 'post_event' || choice.id === 'root');
    if (!selected) {
      const isQueue = current.sceneId === 'post_event.events_choice'
        || current.sceneId === 'post_event.elections_choice';
      selected = isQueue ? choices[0] : choices[Math.floor(random() * choices.length)];
      if (!isQueue && choices.length > 1) forcedChoices += 1;
    }
    const timeBefore = numberQuality(adapter.qualities, 'time');
    current = adapter.choose(selected.index);
    const timeAfter = numberQuality(adapter.qualities, 'time');
    if (enforceNoTimeAdvance && timeAfter !== timeBefore) {
      throw new Error(
        `Mandatory choice ${selected.id} advanced time at ${current.sceneId}; ` +
        `only debug_card.pass_time may advance this control run ` +
        `(date=${String(q.year)}-${String(q.month)}, time=${timeBefore}->${timeAfter}, ` +
        `next=${String(q.next_election_year)}-${String(q.next_election_month)}/${String(q.next_election_time)})`,
      );
    }
  }
  throw new Error(`Mandatory flow exceeded its 300-scene guard at ${current.sceneId}`);
}

export interface LocalNoActionRunResult {
  seed: number;
  difficulty: Difficulty;
  monthsAdvanced: number;
  forcedEventChoices: number;
  historicalStructuralEvents: string[];
  result: LocalElectionSnapshot;
}

export interface LocalNoActionAggregate {
  runs: number;
  meanBarcelonaSeats: Record<string, number>;
  meanBarcelonaValidVoteShare: Record<string, number>;
  meanBarcelonaSupport: Record<string, number>;
  barcelonaSeatHistograms: Record<string, Record<string, number>>;
  cupNoSeatRuns: number;
  cupBelowThresholdRuns: number;
  meanRedBeltScores: Record<string, number>;
  meanRedBeltHoldings: Record<string, number>;
  meanRedBeltTownsWon: Record<string, number>;
  redBeltWinnerCounts: Record<string, Record<string, number>>;
  csTownWinHistogram: Record<string, number>;
  meanDrivers: Record<string, number>;
  cupVoteCorrelations: Record<string, number>;
  csTownCorrelations: Record<string, number>;
  unregisteredSupportRuns: Record<string, number>;
  meanUnregisteredSupport: Record<string, number>;
}

/** Follow the same seeded no-action policy as the Parlament control, but do not
 * stop on an early Parlament election: the target is always the May 2015 local
 * election. This prevents the 2014-election cohort from disappearing. */
export function runNoActionLocalSimulation(options: NoActionRunOptions): LocalNoActionRunResult {
  const originalRandom = Math.random;
  const originalLog = console.log;
  const originalInfo = console.info;
  const controlRandom = seededRandom(options.seed ^ 0xa5a5a5a5);
  Math.random = seededRandom(options.seed);
  if (options.quiet !== false) {
    console.log = () => {};
    console.info = () => {};
  }
  const restoreErrors = installSceneErrorTrap();

  try {
    const adapter = DendryAdapter.fromJSONText(readFileSync(options.gamePath, 'utf8'));
    adapter.beginGame([
      options.seed >>> 0,
      (options.seed ^ 0x9e3779b9) >>> 0,
      (options.seed ^ 0x243f6a88) >>> 0,
      (options.seed ^ 0xb7e15162) >>> 0,
    ]);
    adapter.goToScene('root.start');
    adapter.goToScene(START_SCENES[options.difficulty]);
    adapter.goToScene('root.esquerra');
    let frame = adapter.goToScene('root.esquerra_2');
    for (let guard = 0; frame.effectiveRole !== 'desk' && guard < 20; guard++) {
      const choices = choosable(frame);
      if (choices.length === 0) throw new Error(`Introduction dead-ended at ${frame.sceneId}`);
      frame = adapter.choose(choices[0].index);
    }
    if (frame.effectiveRole !== 'desk') throw new Error('Introduction did not reach the desk');

    frame = adapter.goToScene('election_simulation.2012_11');
    const q = adapter.qualities;
    q.year = 2012;
    q.month = 11;
    q.week = 1;
    q.time = 4;
    q.month_actions = 0;
    q.month_actions_last = 0;
    q.next_election_year = 2015;
    q.next_election_month = 9;
    q.next_election_week = 1;
    q.next_election_time = 38;
    for (const [party, support] of Object.entries(options.localBarcelonaSupportOverrides ?? {})) {
      q[`${party}_local_barcelona_support`] = support;
    }
    const localMatrices = q.LOCAL_BCN_MATRICES as {
      BCN_SENSITIVITY?: Record<string, number[]>;
    } | undefined;
    for (const [party, sensitivity] of Object.entries(
      options.localBarcelonaSensitivityOverrides ?? {},
    )) {
      if (!localMatrices?.BCN_SENSITIVITY?.[party]) {
        throw new Error(`Missing Barcelona sensitivity vector for ${party}`);
      }
      localMatrices.BCN_SENSITIVITY[party] = [...sensitivity];
    }

    const historicalStructuralEvents: string[] = [];
    const ignoredParlamentKeys = new Set<string>();
    let forcedEventChoices = 0;
    for (let monthsAdvanced = 0; monthsAdvanced < 40; monthsAdvanced++) {
      const currentQ = adapter.qualities;
      if (Number(currentQ.year) === 2015 && Number(currentQ.month) === 1
          && !historicalStructuralEvents.includes('plebiscite_election')) {
        const currentCiu = String(currentQ.parlament_current_ciu || 'ciu');
        const roadmapKey = `${currentCiu}_roadmap`;
        if (!Number.isFinite(Number(currentQ[roadmapKey])) || Number(currentQ[roadmapKey]) === 0) {
          currentQ[roadmapKey] = 1;
        }
        frame = adapter.goToScene('plebiscite_election');
        historicalStructuralEvents.push('plebiscite_election');
        const structural = drainMandatoryFlow(
          adapter, frame, controlRandom, ignoredParlamentKeys, false, false,
        );
        forcedEventChoices += structural.forcedChoices;
        frame = structural.frame;
      }

      frame = adapter.goToScene('debug_card.pass_time');
      const drained = drainMandatoryFlow(
        adapter, frame, controlRandom, ignoredParlamentKeys, false, false,
      );
      forcedEventChoices += drained.forcedChoices;
      frame = drained.frame;
      const afterDrainQ = adapter.qualities;
      if (Number(afterDrainQ.year) === 2015
          && Number(afterDrainQ.next_local_election_year) === 2019) {
        return {
          seed: options.seed,
          difficulty: options.difficulty,
          monthsAdvanced: monthsAdvanced + 1,
          forcedEventChoices,
          historicalStructuralEvents,
          result: localSnapshot(afterDrainQ),
        };
      }
    }
    throw new Error(`May 2015 local election was not reached; ended at ${frame.sceneId}`);
  } finally {
    restoreErrors();
    Math.random = originalRandom;
    console.log = originalLog;
    console.info = originalInfo;
  }
}

function correlation(xs: number[], ys: number[]): number {
  const meanX = xs.reduce((sum, value) => sum + value, 0) / xs.length;
  const meanY = ys.reduce((sum, value) => sum + value, 0) / ys.length;
  let covariance = 0;
  let varianceX = 0;
  let varianceY = 0;
  for (let index = 0; index < xs.length; index++) {
    const dx = xs[index] - meanX;
    const dy = ys[index] - meanY;
    covariance += dx * dy;
    varianceX += dx * dx;
    varianceY += dy * dy;
  }
  return varianceX > 0 && varianceY > 0
    ? covariance / Math.sqrt(varianceX * varianceY)
    : 0;
}

export function aggregateNoActionLocalRuns(results: LocalNoActionRunResult[]): LocalNoActionAggregate {
  if (results.length === 0) throw new Error('Cannot aggregate zero local-election runs');
  const mean = (selector: (run: LocalNoActionRunResult, key: string) => number, keys: readonly string[]) =>
    Object.fromEntries(keys.map((key) => [key,
      results.reduce((sum, run) => sum + selector(run, key), 0) / results.length,
    ]));
  const histogram = (values: number[]) => values.reduce<Record<string, number>>((counts, value) => {
    const key = String(value);
    counts[key] = (counts[key] ?? 0) + 1;
    return counts;
  }, {});
  const redBeltWinnerCounts = Object.fromEntries(RED_BELT_LOCATIONS.map((location) => {
    const counts: Record<string, number> = {};
    for (const run of results) {
      const winner = run.result.redBeltWinners[location];
      counts[winner] = (counts[winner] ?? 0) + 1;
    }
    return [location, counts];
  }));
  const townsWon = (run: LocalNoActionRunResult, party: string) =>
    Object.values(run.result.redBeltWinners).filter((winner) => winner === party).length;
  const correlationInputs = [
    ...Object.keys(results[0].result.drivers),
    ...BCN_PARTY_KEYS.map((party) => `support.${party}`),
    ...['cs', 'erc', 'comuns', 'cup'].map((party) => `score.${party}`),
  ];
  const inputValue = (run: LocalNoActionRunResult, key: string) => {
    const [kind, name] = key.split('.');
    if (kind === 'support') return run.result.barcelonaSupport[name] ?? 0;
    if (kind === 'score') return run.result.redBeltScores[name] ?? 0;
    return run.result.drivers[key] ?? 0;
  };
  const cupVotes = results.map((run) => run.result.barcelonaValidVoteShare.cup ?? 0);
  const csTowns = results.map((run) => townsWon(run, 'cs'));
  return {
    runs: results.length,
    meanBarcelonaSeats: mean((run, key) => run.result.barcelonaSeats[key] ?? 0, BCN_PARTY_KEYS),
    meanBarcelonaValidVoteShare: mean((run, key) => run.result.barcelonaValidVoteShare[key] ?? 0, BCN_PARTY_KEYS),
    meanBarcelonaSupport: mean((run, key) => run.result.barcelonaSupport[key] ?? 0, BCN_PARTY_KEYS),
    barcelonaSeatHistograms: Object.fromEntries(BCN_PARTY_KEYS.map((party) => [party,
      histogram(results.map((run) => run.result.barcelonaSeats[party] ?? 0)),
    ])),
    cupNoSeatRuns: results.filter((run) => (run.result.barcelonaSeats.cup ?? 0) === 0).length,
    cupBelowThresholdRuns: results.filter((run) => (run.result.barcelonaValidVoteShare.cup ?? 0) < 5).length,
    meanRedBeltScores: mean((run, key) => run.result.redBeltScores[key] ?? 0, ['cs', 'erc', 'comuns', 'cup']),
    meanRedBeltHoldings: mean((run, key) => run.result.redBeltHoldings[key] ?? 0, RED_BELT_PARTY_KEYS),
    meanRedBeltTownsWon: mean(townsWon, RED_BELT_PARTY_KEYS),
    redBeltWinnerCounts,
    csTownWinHistogram: histogram(csTowns),
    meanDrivers: mean((run, key) => run.result.drivers[key] ?? 0, Object.keys(results[0].result.drivers)),
    cupVoteCorrelations: Object.fromEntries(correlationInputs.map((key) => [key,
      correlation(results.map((run) => inputValue(run, key)), cupVotes),
    ])),
    csTownCorrelations: Object.fromEntries(correlationInputs.map((key) => [key,
      correlation(results.map((run) => inputValue(run, key)), csTowns),
    ])),
    unregisteredSupportRuns: Object.fromEntries(
      Object.keys(results[0].result.unregisteredSupport).map((party) => [party,
        results.filter((run) => (run.result.unregisteredSupport[party] ?? 0) > 0).length,
      ]),
    ),
    meanUnregisteredSupport: mean(
      (run, key) => run.result.unregisteredSupport[key] ?? 0,
      Object.keys(results[0].result.unregisteredSupport),
    ),
  };
}

export function formatNoActionLocalAggregate(aggregate: LocalNoActionAggregate): string {
  const parties = BCN_PARTY_KEYS.filter((party) =>
    Math.abs(aggregate.meanBarcelonaValidVoteShare[party] ?? 0) >= 0.005);
  const strongest = (values: Record<string, number>) => Object.entries(values)
    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
    .slice(0, 8)
    .map(([key, value]) => `${key}=${value.toFixed(3)}`)
    .join(' ');
  return [
    `No-action 2015 local-election Monte Carlo: ${aggregate.runs} runs`,
    `Barcelona mean seats: ${parties.map((party) => `${party}=${aggregate.meanBarcelonaSeats[party].toFixed(2)}`).join(' ')}`,
    `Barcelona mean vote shares (%): ${parties.map((party) => `${party}=${aggregate.meanBarcelonaValidVoteShare[party].toFixed(2)}`).join(' ')}`,
    `Barcelona mean pre-allocation support: ${parties.map((party) => `${party}=${aggregate.meanBarcelonaSupport[party].toFixed(2)}`).join(' ')}`,
    `CUP zero-seat / below-5%: ${aggregate.cupNoSeatRuns}/${aggregate.runs} / ${aggregate.cupBelowThresholdRuns}/${aggregate.runs}`,
    `CUP seat histogram: ${Object.entries(aggregate.barcelonaSeatHistograms.cup).sort(([a], [b]) => Number(a) - Number(b)).map(([seats, count]) => `${seats}:${count}`).join(' ')}`,
    `Red-belt mean scores: ${Object.entries(aggregate.meanRedBeltScores).map(([party, value]) => `${party}=${value.toFixed(2)}`).join(' ')}`,
    `Red-belt mean towns won (of ${RED_BELT_LOCATIONS.length}): ${Object.entries(aggregate.meanRedBeltTownsWon).filter(([, value]) => value >= 0.005).map(([party, value]) => `${party}=${value.toFixed(2)}`).join(' ')}`,
    `Red-belt mean weighted holdings: ${Object.entries(aggregate.meanRedBeltHoldings).filter(([, value]) => value >= 0.005).map(([party, value]) => `${party}=${value.toFixed(2)}`).join(' ')}`,
    `Cs town-win histogram: ${Object.entries(aggregate.csTownWinHistogram).sort(([a], [b]) => Number(a) - Number(b)).map(([towns, count]) => `${towns}:${count}`).join(' ')}`,
    `Mean local-election drivers: ${Object.entries(aggregate.meanDrivers).map(([key, value]) => `${key}=${value.toFixed(2)}`).join(' ')}`,
    `Strongest CUP-vote correlations: ${strongest(aggregate.cupVoteCorrelations)}`,
    `Strongest Cs-town correlations: ${strongest(aggregate.csTownCorrelations)}`,
    `Positive support outside Barcelona ballot registry: ${Object.entries(aggregate.unregisteredSupportRuns).map(([party, count]) => `${party}=${count}/${aggregate.runs}`).join(' ')}`,
    `Mean support outside Barcelona ballot registry: ${Object.entries(aggregate.meanUnregisteredSupport).map(([party, value]) => `${party}=${value.toFixed(3)}`).join(' ')}`,
  ].join('\n');
}

function installSceneErrorTrap(): () => void {
  const original = console.error;
  console.error = (...args: unknown[]) => {
    const message = args.map(String).join(' ');
    // root.start_menu deliberately warns under jsdom's localhost:3000 URL.
    if (message.startsWith('Page loaded from unexpected origin:')) return;
    if (/Scene (action|predicate|expression) error/.test(message)) throw new Error(message);
    original(...args);
  };
  return () => { console.error = original; };
}

export function runNoActionSimulation(options: NoActionRunOptions): NoActionRunResult {
  const originalRandom = Math.random;
  const originalLog = console.log;
  const originalInfo = console.info;
  const controlRandom = seededRandom(options.seed ^ 0xa5a5a5a5);
  Math.random = seededRandom(options.seed);
  if (options.quiet !== false) {
    console.log = () => {};
    console.info = () => {};
  }
  const restoreErrors = installSceneErrorTrap();

  try {
    const adapter = DendryAdapter.fromJSONText(readFileSync(options.gamePath, 'utf8'));
    adapter.beginGame([
      options.seed >>> 0,
      (options.seed ^ 0x9e3779b9) >>> 0,
      (options.seed ^ 0x243f6a88) >>> 0,
      (options.seed ^ 0xb7e15162) >>> 0,
    ]);
    adapter.goToScene('root.start');
    adapter.goToScene(START_SCENES[options.difficulty]);
    adapter.goToScene('root.esquerra');
    let frame = adapter.goToScene('root.esquerra_2');
    for (let guard = 0; frame.effectiveRole !== 'desk' && guard < 20; guard++) {
      const choices = choosable(frame);
      if (choices.length === 0) throw new Error(`Introduction dead-ended at ${frame.sceneId}`);
      frame = adapter.choose(choices[0].index);
    }
    if (frame.effectiveRole !== 'desk') throw new Error('Introduction did not reach the desk');

    const historicalStructuralEvents: string[] = [];
    // The compiled 2012 calibration scene is the canonical 2012 electorate
    // fixture. Start the control period immediately after that election, then
    // schedule the historical September 2015 comparison election. No support,
    // macro variable, lifecycle flag, or government composition is overridden.
    frame = adapter.goToScene('election_simulation.2012_11');
    const q = adapter.qualities;
    q.year = 2012;
    q.month = 11;
    q.week = 1;
    q.time = 4;
    q.month_actions = 0;
    q.month_actions_last = 0;
    q.next_election_year = 2015;
    q.next_election_month = 9;
    q.next_election_week = 1;
    q.next_election_time = 38;
    q.parlament_vote_trace_enabled = true;
    q.parlament_vote_trace = { ticks: 0, mechanisms: {} };
    const baseline = snapshot(q);

    const targetElectionKeys = new Set<string>();
    let forcedEventChoices = 0;
    let monthsAdvanced = 0;
    for (; monthsAdvanced < 60; monthsAdvanced++) {
      const currentQ = adapter.qualities;
      // The real September 2015 election was called by President Mas as a
      // plebiscitary election. The runner used to write that election date
      // directly, bypassing the authored event that raises the CiU-family
      // roadmap and schedules the CiU split. Without those structural effects
      // the later JxSí formation event is unreachable. This is a world event,
      // not a discretionary ERC desk action, so fire its real scene at the
      // historical January 2015 decision point and let post_event resolve all
      // consequences normally.
      if (Number(currentQ.year) === 2015 && Number(currentQ.month) === 1
          && !historicalStructuralEvents.includes('plebiscite_election')) {
        // Loading the electorate fixture bypasses the September 2012 vote
        // which normally initializes this structural field to roadmap 1.
        // Restore only that omitted precondition immediately before executing
        // the authored January event; doing it at boot would make the dynamic
        // plebiscitary-election event eligible in 2014 for some seeds.
        const currentCiu = String(currentQ.parlament_current_ciu || 'ciu');
        const roadmapKey = `${currentCiu}_roadmap`;
        if (!Number.isFinite(Number(currentQ[roadmapKey])) || Number(currentQ[roadmapKey]) === 0) {
          currentQ[roadmapKey] = 1;
        }
        frame = adapter.goToScene('plebiscite_election');
        const eventQ = adapter.qualities;
        if (Number(eventQ[roadmapKey]) !== 2) {
          throw new Error(
            `Plebiscitary election did not promote ${roadmapKey}: ${String(eventQ[roadmapKey])}`,
          );
        }
        historicalStructuralEvents.push('plebiscite_election');
        const structural = drainMandatoryFlow(
          adapter,
          frame,
          controlRandom,
          targetElectionKeys,
        );
        forcedEventChoices += structural.forcedChoices;
        frame = structural.frame;
      }
      const afterStructuralQ = adapter.qualities;
      const monthsToElection = Number(afterStructuralQ.next_election_time)
        - Number(afterStructuralQ.time);
      if (Boolean(afterStructuralQ.unio_split)
          && !Boolean(afterStructuralQ.jxsi_formed)
          && monthsToElection >= 3 && monthsToElection < 8
          && !historicalStructuralEvents.includes('jxsi_formation_offered')) {
        // The game exposes this as a world-event paper even though its inner
        // branch is a player decision. Opening that paper is part of resolving
        // mandatory events; the seeded policy still decides whether ERC
        // accepts or refuses the list.
        frame = adapter.goToScene('jxsi_formation');
        historicalStructuralEvents.push('jxsi_formation_offered');
        const formation = drainMandatoryFlow(
          adapter,
          frame,
          controlRandom,
          targetElectionKeys,
        );
        forcedEventChoices += formation.forcedChoices;
        frame = formation.frame;
      }
      frame = adapter.goToScene('debug_card.pass_time');
      const drained = drainMandatoryFlow(adapter, frame, controlRandom, targetElectionKeys);
      forcedEventChoices += drained.forcedChoices;
      frame = drained.frame;
      if (drained.reachedTarget) {
        const finalQ = adapter.qualities;
        const final = snapshot(finalQ);
        return {
          seed: options.seed,
          difficulty: options.difficulty,
          baseline,
          final,
          majorityRetained: final.broadSovereigntySeats >= numberQuality(finalQ, 'parlament_s_majority'),
          seatDelta: final.broadSovereigntySeats - baseline.broadSovereigntySeats,
          monthsAdvanced: monthsAdvanced + 1,
          forcedEventChoices,
          historicalStructuralEvents,
          structuralState: {
            unioSplit: Boolean(finalQ.unio_split),
            dlFormed: Boolean(finalQ.dl_formed),
            jxsiFormed: Boolean(finalQ.jxsi_formed),
            cupInJxsi: Boolean(finalQ.cup_in_jxsi),
            consultationPending: Boolean(finalQ.consultation_pending),
            consultationHappened: Boolean(finalQ.consultation_happened),
            referendumPending: Boolean(finalQ.referendum_pending),
            referendumHappened: Boolean(finalQ.referendum_happened),
          },
          structuralValues: {
            ciuRoadmap: numberQuality(finalQ, 'ciu_roadmap'),
            cdcRoadmap: numberQuality(finalQ, 'cdc_roadmap'),
            dlRoadmap: numberQuality(finalQ, 'dl_roadmap'),
            cdcRelations: numberQuality(finalQ, 'cdc_relations'),
            dlRelations: numberQuality(finalQ, 'dl_relations'),
          },
          targetElectionKeys: [...targetElectionKeys],
          voteTrace: readVoteTrace(finalQ),
        };
      }
    }
    throw new Error(
      `No post-2012 Parlament election after ${monthsAdvanced} months; ended at ${frame.sceneId} ` +
      `on ${adapter.qualities.year}-${adapter.qualities.month}, ` +
      `next=${adapter.qualities.next_election_year}-${adapter.qualities.next_election_month}, ` +
      `trigger=${String(adapter.qualities.next_election_event_trigger)}`,
    );
  } finally {
    restoreErrors();
    Math.random = originalRandom;
    console.log = originalLog;
    console.info = originalInfo;
  }
}

export function aggregateNoActionRuns(results: NoActionRunResult[]): NoActionAggregate {
  if (results.length === 0) throw new Error('Cannot aggregate zero Monte Carlo runs');
  const meanDelta = (selector: (run: NoActionRunResult, key: string) => number, keys: string[]) =>
    Object.fromEntries(keys.map((key) => [
      key,
      results.reduce((sum, run) => sum + selector(run, key), 0) / results.length,
    ]));
  const summarizeCohorts = (groups: Record<string, NoActionRunResult[]>) =>
    Object.fromEntries(Object.entries(groups).map(([key, runs]) => {
      const cohortMean = (selector: (run: NoActionRunResult) => number) =>
        runs.reduce((sum, run) => sum + selector(run), 0) / runs.length;
      return [key, {
        runs: runs.length,
        retentionRate: runs.filter((run) => run.majorityRetained).length / runs.length,
        meanBroadSovereigntySeats: cohortMean((run) => run.final.broadSovereigntySeats),
        meanAbstentionSupport: cohortMean((run) => run.final.support.abstain),
        meanFamilySupport: Object.fromEntries(Object.keys(DIAGNOSTIC_FAMILIES).map((family) => [
          family,
          cohortMean((run) => run.final.families[family] ?? 0),
        ])),
        meanFamilySeats: Object.fromEntries(Object.keys(DIAGNOSTIC_FAMILIES).map((family) => [
          family,
          cohortMean((run) => run.final.familySeats[family] ?? 0),
        ])),
        meanFamilyValidVoteShare: Object.fromEntries(Object.keys(DIAGNOSTIC_FAMILIES).map((family) => [
          family,
          cohortMean((run) => run.final.familyValidVoteShare[family] ?? 0),
        ])),
      } satisfies NoActionCohort];
    }));
  const byElectionDate = results.reduce<Record<string, NoActionRunResult[]>>((groups, run) => {
    const key = `${run.final.year}-${String(run.final.month).padStart(2, '0')}`;
    (groups[key] ??= []).push(run);
    return groups;
  }, {});
  const byOrganization = results.reduce<Record<string, NoActionRunResult[]>>((groups, run) => {
    const date = `${run.final.year}-${String(run.final.month).padStart(2, '0')}`;
    const organization = !run.structuralState.jxsiFormed
      ? 'jxsi-not-formed'
      : run.structuralState.cupInJxsi ? 'jxsi-formed-cup-in' : 'jxsi-formed-cup-out';
    const key = `${date}/${organization}`;
    (groups[key] ??= []).push(run);
    return groups;
  }, {});
  const traceTargets = new Map<string, Set<string>>();
  for (const run of results) {
    for (const [mechanism, targets] of Object.entries(run.voteTrace)) {
      if (!traceTargets.has(mechanism)) traceTargets.set(mechanism, new Set());
      for (const target of Object.keys(targets)) traceTargets.get(mechanism)!.add(target);
    }
  }
  const meanVoteTrace = Object.fromEntries(
    [...traceTargets.entries()].map(([mechanism, targets]) => [
      mechanism,
      Object.fromEntries([...targets].map((target) => [
        target,
        results.reduce(
          (sum, run) => sum + (run.voteTrace[mechanism]?.[target] ?? 0),
          0,
        ) / results.length,
      ])),
    ]),
  );
  return {
    runs: results.length,
    retained: results.filter((run) => run.majorityRetained).length,
    retentionRate: results.filter((run) => run.majorityRetained).length / results.length,
    meanSeatDelta: results.reduce((sum, run) => sum + run.seatDelta, 0) / results.length,
    meanSupportDelta: meanDelta(
      (run, key) => run.final.support[key] - run.baseline.support[key],
      [...PARTY_KEYS, 'abstain'],
    ),
    meanValidVoteShareDelta: meanDelta(
      (run, key) => (run.final.validVoteShare[key] ?? 0) - (run.baseline.validVoteShare[key] ?? 0),
      [...PARTY_KEYS],
    ),
    meanFinalSeats: meanDelta(
      (run, key) => run.final.seats[key] ?? 0,
      [...PARTY_KEYS],
    ),
    meanFinalValidVoteShare: meanDelta(
      (run, key) => run.final.validVoteShare[key] ?? 0,
      [...PARTY_KEYS],
    ),
    meanFamilyDelta: meanDelta(
      (run, key) => run.final.families[key] - run.baseline.families[key],
      Object.keys(DIAGNOSTIC_FAMILIES),
    ),
    meanFinalFamilySeats: meanDelta(
      (run, key) => run.final.familySeats[key] ?? 0,
      Object.keys(DIAGNOSTIC_FAMILIES),
    ),
    meanFinalFamilyValidVoteShare: meanDelta(
      (run, key) => run.final.familyValidVoteShare[key] ?? 0,
      Object.keys(DIAGNOSTIC_FAMILIES),
    ),
    meanTotalSeats: results.reduce((sum, run) => sum + run.final.totalSeats, 0) / results.length,
    minTotalSeats: Math.min(...results.map((run) => run.final.totalSeats)),
    maxTotalSeats: Math.max(...results.map((run) => run.final.totalSeats)),
    invalidSeatTotalRuns: results.filter((run) => run.final.totalSeats !== 135).length,
    meanTotalValidVoteShare: results.reduce((sum, run) => sum + run.final.totalValidVoteShare, 0) / results.length,
    minTotalValidVoteShare: Math.min(...results.map((run) => run.final.totalValidVoteShare)),
    maxTotalValidVoteShare: Math.max(...results.map((run) => run.final.totalValidVoteShare)),
    // Each party share is rounded independently to 0.01pp by the election
    // algorithm, so a small aggregate deviation is expected.
    invalidValidVoteTotalRuns: results.filter((run) => Math.abs(run.final.totalValidVoteShare - 100) > 0.05).length,
    electionDates: results.reduce<Record<string, number>>((counts, run) => {
      const key = `${run.final.year}-${String(run.final.month).padStart(2, '0')}`;
      counts[key] = (counts[key] ?? 0) + 1;
      return counts;
    }, {}),
    electionDateCohorts: summarizeCohorts(byElectionDate),
    organizationCohorts: summarizeCohorts(byOrganization),
    jxsiCarrierRuns: results.filter((run) => (run.final.support.jxsi ?? 0) > 0).length,
    jxsiFlagRuns: results.filter((run) => run.structuralState.jxsiFormed).length,
    structuralStateCounts: Object.fromEntries(
      Object.keys(results[0].structuralState).map((key) => [
        key,
        results.filter((run) => run.structuralState[key as keyof typeof run.structuralState]).length,
      ]),
    ),
    meanStructuralValues: meanDelta(
      (run, key) => run.structuralValues[key] ?? 0,
      Object.keys(results[0].structuralValues),
    ),
    meanVoteTrace,
  };
}

export function formatNoActionAggregate(aggregate: NoActionAggregate): string {
  const support = Object.entries(aggregate.meanSupportDelta)
    .filter(([key]) => [
      'ciu', 'cdc', 'dl', 'pdcat', 'junts', 'jxsi', 'jxcat', 'erc', 'cup',
      'psc', 'cs', 'ppc', 'fnc', 'pxc', 'abstain',
    ].includes(key))
    .map(([key, value]) => `${key}=${value >= 0 ? '+' : ''}${value.toFixed(3)}`)
    .join(' ');
  const families = Object.entries(aggregate.meanFamilyDelta)
    .map(([key, value]) => `${key}=${value >= 0 ? '+' : ''}${value.toFixed(3)}`)
    .join(' ');
  const displayedParties = [
    'ciu', 'cdc', 'dl', 'pdcat', 'junts', 'jxsi', 'jxcat', 'erc', 'cup',
    'psc', 'icv', 'csqp', 'cecp', 'ecp', 'cs', 'ppc', 'unio', 'fnc', 'pxc',
  ];
  const finalSeats = displayedParties
    .filter((key) => Math.abs(aggregate.meanFinalSeats[key] ?? 0) >= 0.005)
    .map((key) => `${key}=${aggregate.meanFinalSeats[key].toFixed(2)}`)
    .join(' ');
  const finalValidVoteShare = displayedParties
    .filter((key) => Math.abs(aggregate.meanFinalValidVoteShare[key] ?? 0) >= 0.005)
    .map((key) => `${key}=${aggregate.meanFinalValidVoteShare[key].toFixed(2)}`)
    .join(' ');
  const voteShareDelta = displayedParties
    .filter((key) => Math.abs(aggregate.meanValidVoteShareDelta[key] ?? 0) >= 0.0005)
    .map((key) => {
      const value = aggregate.meanValidVoteShareDelta[key];
      return `${key}=${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
    })
    .join(' ');
  const finalFamilySeats = Object.entries(aggregate.meanFinalFamilySeats)
    .filter(([key]) => key !== 'abstention')
    .map(([key, value]) => `${key}=${value.toFixed(2)}`)
    .join(' ');
  const finalFamilyVoteShare = Object.entries(aggregate.meanFinalFamilyValidVoteShare)
    .filter(([key]) => key !== 'abstention')
    .map(([key, value]) => `${key}=${value.toFixed(2)}`)
    .join(' ');
  const cohortLines = Object.entries(aggregate.electionDateCohorts)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, cohort]) => {
      const s = cohort.meanFamilySeats;
      const v = cohort.meanFamilyValidVoteShare;
      const e = cohort.meanFamilySupport;
      return `  ${date} n=${cohort.runs} majority=${(cohort.retentionRate * 100).toFixed(1)}% `
        + `sovereignty=${cohort.meanBroadSovereigntySeats.toFixed(2)}s/${v.sovereignty.toFixed(2)}%v/${e.sovereignty.toFixed(2)}%e `
        + `mainstream=${s.sovereignty_mainstream.toFixed(2)}s/${v.sovereignty_mainstream.toFixed(2)}%v/${e.sovereignty_mainstream.toFixed(2)}%e `
        + `joint=${s.sovereignty_joint_lists.toFixed(2)}s/${v.sovereignty_joint_lists.toFixed(2)}%v/${e.sovereignty_joint_lists.toFixed(2)}%e `
        + `CUP=${s.cup_outside_joint_lists.toFixed(2)}s/${v.cup_outside_joint_lists.toFixed(2)}%v/${e.cup_outside_joint_lists.toFixed(2)}%e `
        + `Cs=${s.cs.toFixed(2)}s/${v.cs.toFixed(2)}%v/${e.cs.toFixed(2)}%e `
        + `PPC=${s.ppc.toFixed(2)}s/${v.ppc.toFixed(2)}%v/${e.ppc.toFixed(2)}%e `
        + `PSC=${s.psc.toFixed(2)}s/${v.psc.toFixed(2)}%v/${e.psc.toFixed(2)}%e `
        + `federal-left=${s.federal_left.toFixed(2)}s/${v.federal_left.toFixed(2)}%v/${e.federal_left.toFixed(2)}%e `
        + `abstention=${cohort.meanAbstentionSupport.toFixed(2)}%`;
    });
  const organizationLines = Object.entries(aggregate.organizationCohorts)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, cohort]) => {
      const s = cohort.meanFamilySeats;
      const v = cohort.meanFamilyValidVoteShare;
      const e = cohort.meanFamilySupport;
      return `  ${key} n=${cohort.runs} majority=${(cohort.retentionRate * 100).toFixed(1)}% `
        + `sovereignty=${cohort.meanBroadSovereigntySeats.toFixed(2)}s/${v.sovereignty.toFixed(2)}%v/${e.sovereignty.toFixed(2)}%e `
        + `mainstream=${s.sovereignty_mainstream.toFixed(2)}s/${v.sovereignty_mainstream.toFixed(2)}%v/${e.sovereignty_mainstream.toFixed(2)}%e `
        + `CUP=${s.cup_outside_joint_lists.toFixed(2)}s/${v.cup_outside_joint_lists.toFixed(2)}%v/${e.cup_outside_joint_lists.toFixed(2)}%e`;
    });
  const voteTraceLines = Object.entries(aggregate.meanVoteTrace)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([mechanism, targets]) => {
      const entries = Object.entries(targets)
        .filter(([, value]) => Math.abs(value) >= 0.0005)
        .sort(([a], [b]) => a.localeCompare(b));
      const net = entries.reduce((sum, [, value]) => sum + value, 0);
      return `  ${mechanism}: ${entries.map(([target, value]) =>
        `${target}=${value >= 0 ? '+' : ''}${value.toFixed(3)}`,
      ).join(' ')} net=${net >= 0 ? '+' : ''}${net.toFixed(3)}`;
    });
  return [
    `No-action Dendry Monte Carlo: ${aggregate.runs} runs`,
    `Majority retained: ${aggregate.retained}/${aggregate.runs} (${(aggregate.retentionRate * 100).toFixed(1)}%)`,
    `JxSi carrier / formation flag: ${aggregate.jxsiCarrierRuns}/${aggregate.runs} / ${aggregate.jxsiFlagRuns}/${aggregate.runs}`,
    `First post-2012 election dates: ${Object.entries(aggregate.electionDates).sort(([a], [b]) => a.localeCompare(b)).map(([key, value]) => `${key}=${value}`).join(' ')}`,
    'Election-date cohorts (mean seats / valid-vote share / electorate support):',
    ...cohortLines,
    'Election-date / JxSi-organization cohorts:',
    ...organizationLines,
    `Final structural-state counts: ${Object.entries(aggregate.structuralStateCounts).map(([key, value]) => `${key}=${value}`).join(' ')}`,
    `Mean final structural values: ${Object.entries(aggregate.meanStructuralValues).map(([key, value]) => `${key}=${value.toFixed(2)}`).join(' ')}`,
    `Mean broader-sovereignty seat delta: ${aggregate.meanSeatDelta.toFixed(2)}`,
    `Seat-total invariant: mean=${aggregate.meanTotalSeats.toFixed(2)} min=${aggregate.minTotalSeats} max=${aggregate.maxTotalSeats} invalid=${aggregate.invalidSeatTotalRuns}/${aggregate.runs}`,
    `Valid-vote-total invariant: mean=${aggregate.meanTotalValidVoteShare.toFixed(4)} min=${aggregate.minTotalValidVoteShare.toFixed(4)} max=${aggregate.maxTotalValidVoteShare.toFixed(4)} invalid=${aggregate.invalidValidVoteTotalRuns}/${aggregate.runs}`,
    `Mean final seats: ${finalSeats}`,
    `Mean final family seats: ${finalFamilySeats}`,
    `Mean final valid-vote shares (%): ${finalValidVoteShare}`,
    `Mean final family valid-vote shares (%): ${finalFamilyVoteShare}`,
    `Mean party valid-vote-share deltas (pp): ${voteShareDelta}`,
    `Mean party/electorate support deltas (pp): ${support}`,
    `Mean family support deltas (pp): ${families}`,
    'Vote-flow trace (cumulative population-weighted electorate pp, mean/run):',
    ...voteTraceLines,
  ].join('\n');
}
