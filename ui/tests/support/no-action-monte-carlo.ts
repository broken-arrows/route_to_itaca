import { readFileSync } from 'node:fs';
import { DendryAdapter } from '../../src/engine/adapter';
import type { Frame } from '../../src/engine/types';

export type Difficulty = 'easy' | 'normal' | 'hard';

export interface NoActionRunOptions {
  gamePath: string;
  seed: number;
  difficulty: Difficulty;
  quiet?: boolean;
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
}

const PARTY_KEYS = [
  'ciu', 'erc', 'cup', 'si', 'cdc', 'unio', 'dl', 'jxsi', 'jxcat', 'junts',
  'pdcat', 'psc', 'icv', 'csqp', 'cecp', 'ecp', 'cs', 'ppc', 'vox', 'fnc', 'pxc',
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
  };
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
    if (completedParlamentElection) {
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
    if (timeAfter !== timeBefore) {
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
  ].join('\n');
}
