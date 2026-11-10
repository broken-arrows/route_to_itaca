import { DendryAdapter } from '../../src/engine/adapter';
import { gameLib } from '../../src/game-bindings';

export type Strategy = 'historical' | 'services' | 'campaign';
export type HistoricalHorizon = '2015-09' | '2017-08';
const macroKeys = ['gdp_growth', 'unemployment', 'generalitat_surplus', 'public_debt', 'welfare_index',
  'social_dissent', 'independence_movement', 'independence_trust', 'education_spending',
  'healthcare_spending', 'public_salary_level', 'tax_policy_level', 'social_spending_cap', 'budget_year', 'party_resources'];
export const HISTORICAL_BALANCE_ASSUMPTIONS = [
  'Normal difficulty, normal August 2012 electorate; no historical result fixture or forced vote totals.',
  'NPC policy/event dates are scripted; unscheduled government initiatives and budget proposals are disabled.',
  'ERC externally supports CiU, accepts its proposal and gives it the Speaker; JxSi 40% ERC, CUP outside.',
  '2013 pay cut; 2014 and 2015 budgets; pay restored in March 2015. Inherited high regional taxes retained.',
  'Lleida-La Pobla prepared by the CiU ministry with one NPC action allowance, funded January 2014; completion scheduled July 2016.',
  'Sovereignty January 2013; Congreso April 2014; consultation law/question September 2014; double-question vote November 2014.',
  'LGBTI enforcement October 2014; energy disconnection ban and compulsory housing relief July 2015 use existing law approximations.',
  'January 2015 election announcement separated from August caretaker status; Unio split June; JxSi July.',
  'Historical/services campaigning: October 2012 and August 2015. Campaign variant: each affordable month, reserving one resource until the double consultation question is secured; no free resources.',
  'Campaign card access is guaranteed to isolate action efficacy from draw luck; parent time and option resource costs retained.',
  'Campaigns rotate workers/retirees/rural/unemployed. Middle-class action excluded: missing raul_romeva_advisor causes NaN.',
  'Services variant prescribes NPC agreement to restore education May 2014 and healthcare June 2014, subject to normal execution gates.',
  'Other choices use explicit preferences, then exit, then first available. All choices and deviations are recorded.',
  'Not a complete historical reconstruction or proof these actions are attainable by an ERC player from random card draws.',
];
export const HISTORICAL_2017_ASSUMPTIONS = [
  'Continue through the first August 2017 checkpoint, before the September referendum law and Rubicon.',
  'November/December 2015 bargaining, ERC private then public pressure for replacing Mas; select Puigdemont in January 2016. Normal assembly and investiture rules apply.',
  'The authored investiture social commitments restore education/health to level 0; this is a coarse social-plan approximation.',
  'June 2016 budget: cap 0, tax 1, ERC yes, no further concessions; actual CUP refusal leaves the 2015 budget extended.',
  'March 2017 budget: cap 1, tax 2, normal parliamentary support. These are coarse game policy settings, not a euro reconstruction.',
  'Spanish elections remain December 2015/June 2016; the authored PP toleration resolves in June rather than October 2016. PP remains the macro governing category throughout.',
  'September 2016 confidence vote and June 2017 referendum announcement are recorded milestones only; no bespoke scene effects are invented.',
  'No additional discretionary projects or laws are prescribed after 2015; existing policies and automatic world events continue.',
];

export function runHistoricalBalance(game: string, seed: number, strategy: Strategy, engineOverrides: object = {},
  observer?: (qualities: Record<string, any>, scene: string) => void, horizon: HistoricalHorizon = '2015-09') {
  const extended = horizon === '2017-08';
  const original = { random: Math.random, log: console.log, info: console.info, error: console.error };
  let rng = seed >>> 0;
  Math.random = () => { rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0; return rng / 4294967296; };
  console.log = console.info = () => {};
  const diagnostics: string[] = [];
  console.error = (...args) => {
    const message = args.join(' ');
    if (/Scene (action|predicate|expression) error/.test(message)) throw new Error(message);
    if (!message.startsWith('Page loaded from unexpected origin:')) diagnostics.push(message);
  };
  const log: Array<{date: string; scene: string; choice: string}> = [];
  const monthly: Array<Record<string, unknown>> = [];
  const elections: Array<Record<string, any>> = [];
  const congresoElections: Array<Record<string, any>> = [];
  let consultation: Record<string, unknown> | null = null;
  const interventions: Array<Record<string, unknown>> = [];
  const checkpoints: Array<Record<string, any>> = [];
  const governments: Array<Record<string, any>> = [];
  let rejectBudget = false;
  try {
    // A controlled world-policy scenario, not an autonomous player bot. Only
    // event discovery is replaced: actions, costs, votes and monthly effects
    // remain the compiled game's. The manifest and interventions are exported.
    const compiled = JSON.parse(game);
    const scheduled = ['government_does_things', 'budget_trigger', 'plebiscite_election',
      'question_of_the_question', 'jxsi_formation', '2013parlament_sovereignty',
      '2014nonbinding_consultation_law', '2013congresdecidir', 'unio_breaks'];
    for (const id of scheduled) compiled.scenes[id].viewIf = {$code: 'return false;'};
    const adapter = DendryAdapter.fromJSONText(JSON.stringify(compiled));
    adapter.engine.setGameLib({...gameLib, ...engineOverrides});
    adapter.beginGame([seed, seed ^ 0x123456, seed ^ 0x765432, seed ^ 0x13579]);
    adapter.goToScene('root.start');
    adapter.goToScene('root.normal_mode');
    adapter.goToScene('root.esquerra');
    let frame = adapter.goToScene('root.esquerra_2');
    const q = adapter.qualities as Record<string, any>;
    const date = () => `${q.year}-${String(q.month).padStart(2, '0')}`;
    const economy = (): Record<string, any> => ({date: date(), ...Object.fromEntries(macroKeys.map(k => [k, q[k]]))});
    q.parlament_vote_trace_enabled = true;
    q.parlament_vote_trace = {ticks: 0, mechanisms: {}};
    const pathFailures: string[] = [];
    function observe() {
      observer?.(q, frame.sceneId);
      if (extended) {
        const snapshot = {...economy(), president:q.president, budgetYear:q.budget_year,
          railway:q.territory_rail1_countdown, rubicon:q.rubicon};
        if (checkpoints.at(-1)?.date === date()) checkpoints[checkpoints.length - 1] = snapshot;
        else checkpoints.push(snapshot);
        const government = {president:q.president, coalition:[...(q.cat_coalition || [])],
          support:[...(q.cat_coalition_support || [])], spanish:[...(q.spanish_coalition || [])], caretaker:q.cat_caretaker_gov};
        if (JSON.stringify(governments.at(-1)?.government) !== JSON.stringify(government))
          governments.push({date:date(),government});
      }
      if ((q.election_history_events || []).filter((e: any) => e.type === 'parlament-election').length > elections.length) {
        const parties = (q.parties as string[]).filter(p => p !== 'abstain');
        elections.push({date: date(), seats: Object.fromEntries(parties.map(p => [p, q[p + '_parlament_s']])),
          votes: Object.fromEntries(parties.map(p => [p, q[p + '_parlament_pv']]))});
      }
      if ((q.election_history_events || []).filter((e: any) => e.type === 'congreso-election').length > congresoElections.length) {
        const parties = (q.congreso_parties as string[]).filter(p => p !== 'abstain');
        const values = (suffix: string) => Object.fromEntries(parties.map(p => [p, Number(q[p + suffix] || 0)]));
        const regions = Object.fromEntries(q.congreso_constituencies.map((c: string) => {
          const local = q['congreso_parties_' + c] as string[];
          const stocks = Object.fromEntries(local.map(p => [p, Number(q[p + '_congreso_' + c + '_support'] || 0)]));
          const partyTotal = Object.values(stocks).reduce((s, n) => s + n, 0);
          const abstain = Number(q['abstain_congreso_' + c + '_support'] || 0);
          return [c, {seats: Object.fromEntries(local.map(p => [p, Number(q[p + '_congreso_s_' + c] || 0)])),
            votes: Object.fromEntries(local.map(p => [p, partyTotal > 0 ? stocks[p] / partyTotal * 100 : 0])),
            stocks, abstain, census: q.congreso_pop[c], seatCount: q.congreso_seats[c],
            turnout: partyTotal / (partyTotal + abstain) * 100}];
        }));
        congresoElections.push({date: date(), seats: values('_congreso_s'), votes: values('_congreso_pv'),
          catalunyaSeats: values('_congreso_s_catalunya'), catalunyaVotes: values('_congreso_pv_cat'),
          turnout: Number(q.congreso_participation_pc), regions});
      }
      if (q.consultation_happened && q.consultation_total_votes > 0 && !consultation)
        consultation = {date: date(), ...Object.fromEntries(Object.entries(q).filter(([k,v]) => k.startsWith('consultation_') && typeof v === 'number')),
          yesVotes: Number(String(q.consultation_yes).replaceAll('.', '')),
          yesNoVotes: Number(String(q.consultation_yesno).replaceAll('.', '')),
          noVotes: Number(String(q.consultation_no).replaceAll('.', ''))};
    }
    function drain() {
      for (let i = 0; i < 400; i++) {
        observe();
        // Direct world-law entry can inherit the desk presentation role. Stop
        // only at the actual main scene, not at a role-less law's first page.
        if (frame.gameOver) throw new Error(`Unexpected ending at ${date()}: ${frame.sceneId}`);
        if (['main', 'main.main_easy', 'main.main_hard'].includes(frame.sceneId) || !extended && elections.length === 2) return;
        const choices = frame.choices.map((c,index) => ({...c,index})).filter(c => c.canChoose);
        const continuation = extended && date()>='2015-09' ? [
          'parlament_coalition_joint_cup.erc_player_jxsi_led','parlament_coalition_joint_cup.jxsi_speaker',
          'parlament_coalition_sovereignty_replacement.puigdemont',
          q.parlament_coalition_sovereignty_ciu_pressure > 0
            ? 'parlament_coalition_sovereignty_candidate_erc.public'
            : 'parlament_coalition_sovereignty_candidate_erc.private',
          'parlament_coalition_joint_cup_phase_erc_player_jxsi_led.withdrawal',
          'parlament_coalition_joint_cup_phase_erc_player_jxsi_led.submit',
          'parlament_coalition_joint_cup_phase_erc_player_jxsi_led.accept',
          'congreso_coalition.new_congreso_elections','congreso_coalition_right.pp_minority',
        ] : [];
        const preferences = [...continuation, 'parlament_coalition_ciu_minority.erc', 'parlament_coalition_ciu_minority.ciu_speaker',
          'parlament_coalition_ciu_minority_phase.accept', 'parlament_coalition_ciu_erc_phase.accept',
          ...(rejectBudget ? ['parlament_budget.vote_yes', 'parlament_budget.walk_away'] : ['parlament_budget.pass', 'parlament_budget.vote_yes']),
          '2012parlamentvote.sovereign', '2012parlamentvote.2014',
          'jxsi_formation.jxsi_cup_no', 'parlament_lgbt.lgbt_enforcement',
          'parlament_energy_poverty.energy_disconnection_ban', 'parlament_energy_poverty.energy_poverty_vote',
          'parlament_housing.housing_compulsory_relief', 'parlament_housing.housing_vote',
          'question_of_the_question.double', 'jxsi_formation.erc_40', 'government_does_things.accept',
          'post_event', 'root'];
        const selected = preferences.map(id => choices.find(c => c.id === id)).find(Boolean) || choices[0];
        if (!selected) throw new Error(`No choices at ${frame.sceneId}`);
        log.push({date: date(), scene: frame.sceneId, choice: selected.id});
        frame = adapter.choose(selected.index);
        if (extended && selected.id === 'parlament_coalition_joint_cup.jxsi_speaker') {
          q.coalition_phase_timer = 2;
          interventions.push({date: date(), kind: 'calendar', firstInvestitureRound: '2015-11', finalInvestitureRound: '2016-01'});
        }
      }
      throw new Error(`Flow loop at ${frame.sceneId}`);
    }
    drain();
    function worldScene(id: string) {
      interventions.push({date: date(), kind: 'scheduled-world-scene', id});
      q.coming_from_parlament = false;
      frame = adapter.goToScene(id);
      drain();
    }
    function policy(field: string, to: number, holder: string) {
      const from = q[field];
      if (from === to) return;
      // Prescribe the NPC's policy decision, not its effects. The normal
      // prepare/resolve path still validates ownership, commitments and funding.
      q.npc_government_initiative = {id: `${holder}:${field}:${to}`, holder, field, from, to,
        party: q[holder] === 'indp.' ? q.president_party : q[holder], policyParty: q.president_party,
        profile: 'services', kind: to < from ? 'cut' : 'policy',
        title: `Reference policy: ${field} ${from} -> ${to}`, reason: 'Historical balance scenario', office: holder};
      interventions.push({date: date(), kind: 'npc-policy-input', field, from, to});
      worldScene('government_does_things');
      interventions.push({date: date(), kind: 'policy-result', field, requested: to, actual: q[field]});
      if (q[field] !== to) pathFailures.push(`${date()}: ${field} requested ${to}, reached ${q[field]}`);
    }
    function budget(cap: number, tax?: number, shouldPass = true) {
      const previousYear = q.budget_year;
      interventions.push({date: date(), kind: 'npc-budget-input', cap, tax, shouldPass});
      frame = adapter.goToScene('budget_trigger');
      // Fix the author's proposed spending authorization. Parliament still
      // votes on it; no revenue, fiscal balance or successful passage is forced.
      q.budget_draft_origin.social_spending_cap = cap;
      q.budget_draft.social_spending_cap = cap;
      if (tax !== undefined) q.budget_draft.tax_policy_level = q.budget_draft_origin.tax_policy_level = tax;
      frame = adapter.goToScene('parlament_budget.review');
      interventions.push({date: date(), kind: 'budget-choices', choices: frame.choices,
        projection: q.parlament_budget_projection, origin: q.budget_draft_origin});
      rejectBudget = !shouldPass;
      try { drain(); } finally { rejectBudget = false; }
      interventions.push({date: date(), kind: 'budget-result', enactedYear: q.budget_year, cap: q.social_spending_cap,
        positions: {...q.parlament_budget_positions}, shouldPass});
      if (shouldPass ? q.budget_year !== q.year || q.social_spending_cap !== cap : q.budget_year !== previousYear)
        pathFailures.push(`${date()}: budget did not ${shouldPass ? 'pass' : 'fail'} as scheduled`);
    }
    const completed = new Set<string>();
    let campaignCount = 0;
    for (let i = 0; i < (extended ? 120 : 60) && (extended ? date() < horizon : elections.length < 2); i++) {
      const now = date();
      if (!completed.has(now)) {
        completed.add(now);
        if (now === '2013-01') worldScene('2013parlament_sovereignty');
        if (now === '2013-03') policy('public_salary_level', -1, 'economy_minister_party');
        if (now === '2014-01') {
          // The historical project is a CiU ministry decision. Temporarily
          // execute its real guarded action as that actor, without giving ERC
          // a ministry or awarding ERC the project's initiation credit.
          const player = q.player_party;
          const playerBudget = q.government_budget;
          q.player_party = q.territory_minister_party;
          // External-support games do not initialize an NPC action allowance.
          // Supply one project action to the NPC, never to the ERC player.
          q.government_budget = 1;
          try { frame = adapter.goToScene('generalitat_territory_card.rail1'); }
          finally { q.player_party = player; q.government_budget = playerBudget; }
          interventions.push({date: date(), kind: 'npc-project', id: 'territory_rail1', prepared: q.territory_rail_approved,
            npcActionAllowance: 1, minister: q.territory_minister_party, caretaker: q.cat_caretaker_gov});
          if (q.territory_rail_approved !== 1) pathFailures.push(`${date()}: railway preparation failed`);
          drain();
          budget(strategy === 'services' ? 0 : -1);
          if (q.territory_rail1_countdown > 0) {
            q.territory_rail1_countdown = 30;
            interventions.push({date: date(), kind: 'calendar', railwayCompletion: '2016-07'});
          } else pathFailures.push(`${date()}: railway was not funded`);
        }
        if (now === '2014-04') worldScene('2013congresdecidir');
        if (now === '2014-05' && strategy === 'services') policy('education_spending', 0, 'education_minister_party');
        if (now === '2014-06' && strategy === 'services') policy('healthcare_spending', 0, 'health_minister_party');
        if (now === '2014-09') {
          worldScene('2014nonbinding_consultation_law');
          worldScene('question_of_the_question');
          q.consultation_day_countdown = 2;
          interventions.push({date: date(), kind: 'calendar', consultation: '2014-11'});
        }
        if (now === '2014-10') worldScene('parlament_lgbt');
        if (now === '2015-01') {
          worldScene('plebiscite_election');
          // January announcement is not August dissolution. The authored event
          // conflates them; retain ordinary government until the summer.
          q.cat_caretaker_gov = false;
          q.unio_breakup_countdown = 5;
          interventions.push({date: date(), kind: 'calendar', caretaker: '2015-08', unioSplit: '2015-06'});
        }
        if (now === '2015-03') { budget(0); policy('public_salary_level', 0, 'economy_minister_party'); }
        if (now === '2015-06') worldScene('unio_breaks');
        if (now === '2015-07') {
          worldScene('parlament_energy_poverty');
          worldScene('parlament_housing');
          worldScene('jxsi_formation');
        }
        if (now === '2015-08') q.cat_caretaker_gov = true;
        if (extended && now === '2016-06') budget(0, 1, false);
        if (extended && now === '2016-09') interventions.push({date:now,kind:'unmodeled-milestone',id:'confidence-vote',reason:'No dedicated confidence-vote scene; no invented effects.'});
        if (extended && now === '2017-03') budget(1, 2);
        if (extended && now === '2017-06') interventions.push({date:now,kind:'unmodeled-milestone',id:'referendum-announcement',reason:'Announcement only; the referendum law belongs in September, after this horizon.'});
      }
      monthly.push(economy());
      const wantsCampaign = strategy === 'campaign' || ['2012-10', '2015-08'].includes(date());
      const reserve = date() < '2014-09' ? 1 : 0;
      if (wantsCampaign && q.party_resources > reserve) {
        // Controlled card access isolates campaign effectiveness from draw luck.
        // Parent action accounting and the chosen option's resource costs apply.
        const id = q.player_in_jxsi ? 'unity_campaign' : 'erc_campaigning';
        frame = adapter.goToScene(id);
        const target = ['workers', 'retirees', 'rural', 'unemployed'][campaignCount % 4];
        const index = frame.choices.findIndex(c => c.id === `${id}.${target}` && c.canChoose);
        if (index < 0) throw new Error(`Unavailable campaign ${id}.${target}`);
        interventions.push({date: date(), kind: 'campaign', id: `${id}.${target}`, resourcesBefore: q.party_resources});
        campaignCount++;
        frame = adapter.choose(index);
        drain();
        continue;
      }
      frame = adapter.goToScene('debug_card.pass_time');
      drain();
    }
    monthly.push(economy());
    const laws = Object.fromEntries(['parlament_sovereignty_declaration', 'parlament_non_binding_consultations',
      'parlament_lgbt_passed', 'parlament_energy_poverty_passed', 'parlament_housing_passed'].map(k => [k, Boolean(q[k])]));
    for (const [law, passed] of Object.entries(laws)) if (!passed) pathFailures.push(`Law did not pass: ${law}`);
    if (!q.jxsi_formed || q.cup_in_jxsi || q.jxsi_formation_num !== 40) pathFailures.push('Historical JxSi composition not reached');
    if (extended && date() !== horizon) pathFailures.push(`Horizon ${horizon} not reached: ${date()}`);
    if (extended) {
      for (let i = 0; i < 61; i++) {
        const month = 7 + i;
        const expected = `${2012 + Math.floor(month / 12)}-${String(month % 12 + 1).padStart(2, '0')}`;
        if (checkpoints[i]?.date !== expected) pathFailures.push(`Missing monthly checkpoint: ${expected}`);
      }
      if (elections.length !== 2) pathFailures.push(`Unexpected Parlament elections: ${elections.length}`);
      if (!governments.some(r => r.date === '2016-01' && r.government.president === 'Carles Puigdemont')) pathFailures.push('Puigdemont not invested in January 2016');
      if (q.president !== 'Carles Puigdemont' || q.cat_coalition.join(',') !== 'jxsi' || q.cat_coalition_support.join(',') !== 'cup') pathFailures.push('JxSi/CUP government not retained');
      if (q.rubicon || q.parlament_referendum_law) pathFailures.push('Referendum law/Rubicon reached before September 2017');
      if (q.budget_year !== 2017) pathFailures.push('2017 budget not enacted');
      if (!interventions.some(r => r.kind === 'budget-result' && r.date === '2016-06' && r.enactedYear === 2015)) pathFailures.push('June 2016 budget rejection not reached');
      if (!interventions.some(r => r.kind === 'budget-result' && r.date === '2017-03' && r.enactedYear === 2017)) pathFailures.push('March 2017 budget approval not reached');
      if (checkpoints.find(r => r.date === '2016-06')?.railway !== 1 || checkpoints.find(r => r.date === '2016-07')?.railway !== -1) pathFailures.push('Railway did not complete in July 2016');
      if (q.spanish_coalition.join(',') !== 'pp') pathFailures.push('PP government not retained');
    }
    return {seed, strategy, elections, congresoElections, consultation: consultation as Record<string, unknown> | null, monthly, log, diagnostics, interventions, campaignCount,
      ...(extended ? {horizon,checkpoints,governments,finalGovernment:{president:q.president,coalition:q.cat_coalition,support:q.cat_coalition_support,
        spanish:q.spanish_coalition,caretaker:q.cat_caretaker_gov,rubicon:q.rubicon},
        finalPolicies:JSON.parse(JSON.stringify(q.active_mods))} : {}),
      laws, pathFailures, finalEconomy: economy(), voteTrace: JSON.parse(JSON.stringify(q.parlament_vote_trace))};
  } catch (error) {
    throw new Error(`${strategy} seed ${seed}: ${String(error)}\n${JSON.stringify(log.slice(-25))}`, {cause: error});
  } finally {
    Math.random = original.random; console.log = original.log; console.info = original.info; console.error = original.error;
  }
}
