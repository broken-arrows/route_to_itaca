import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const dir = resolve(import.meta.dirname, '../../source/scenes/events/elections');
const game = readFileSync(resolve(dir, '../../../../out/game.json'), 'utf8');
const P = 'parlament_coalition_sovereignty_';
const routes = readdirSync(dir).filter(f => /^parlament_coalition_(joint_cup|ciu_erc_cup)_phase_/.test(f)).map(f => f.replace('.scene.dry', ''));
function action(scene: string, marker: string, q: Record<string, any>) {
  const text = readFileSync(resolve(dir, scene + '.scene.dry'), 'utf8');
  const node = text.slice(text.search(new RegExp(`^@${marker}\\r?$`, 'm'))).split(/\r?\n@/)[0];
  new Function('Q', 'G', node.match(/on-arrival:\s*\{!([\s\S]*?)!\}/)![1]).call({ achieve() {} }, q, {random: () => 0.5});
}
function adapter(route: string, candidate = 'Artur Mas') {
  const a = DendryAdapter.fromJSONText(game);
  a.beginGame([1, 2, 3, 4]); a.goToScene('root.start_menu'); a.goToScene('root.start');
  const joint = route.includes('joint_cup');
  const player = route.includes('_erc_player_') ? 'erc' : 'cup';
  const leader = joint ? route.includes('jxsi') ? 'jxsi' : 'jxcat' : route.includes('_erc_led_') ? 'erc' : 'cdc';
  const kind = joint ? leader : route.endsWith('government') ? 'government' : 'support';
  Object.assign(a.qualities, {
    player_party: player, year: 2015, month: 2, time: 30, roadmap_year: 2017, rubicon: false,
    parlament_current_ciu: 'cdc', parlament_current_ciu_s: 35, cdc_parlament_s: 35,
    erc_parlament_s: 35, cup_parlament_s: 10, jxsi_parlament_s: 70, jxcat_parlament_s: 70,
    jxsi_formation_num: 50, jxcat_formation_num: 50,
    cdc_leader: candidate, erc_leader: 'Oriol Junqueras', jxsi_leader: candidate, jxcat_leader: candidate,
    cdc_relations: 70, erc_relations: 70, cup_relations: 60,
    cdc_roadmap: 2, erc_roadmap: 2, cup_roadmap: 3, jxsi_roadmap: 2, jxcat_roadmap: 2,
    cup_faction: 'Endavant', cup_endavant_strength: 60, cup_poblelliure_strength: 20, cup_thirdsector_strength: 20,
    parlament_coalition_leader: leader, parlament_coalition_follower: joint ? 'cup' : leader === 'erc' ? 'cdc' : 'erc',
    parlament_coalition_proposal_president: leader, parlament_coalition_proposal_president_name: candidate,
    parlament_coalition_support_seats: 80, parlament_coalition_cabinet_seats: kind === 'government' ? 80 : 70,
    [P + 'kind']: kind, [P + 'phase']: 1, [P + 'cabinet']: joint ? [leader] : ['cdc', 'erc', ...(kind === 'government' ? ['cup'] : [])],
    [P + 'prepared_round']: 0, [P + 'replacement_needed']: false,
    [P + 'assembly_result']: '', [P + 'assembly_terms']: [], [P + 'mas_pressure']: 0,
    [P + 'ciu_pressure']: 0, [P + 'dispute_round']: 0, [P + 'mediation_round']: 3, [P + 'feud']: 0,
    parlament_coalition_push: 0, parlament_coalition_push_round: 0, speaker_party: 'none',
    coalition_phase_timer: 0, generalitat_surplus: 0, social_spending_cap: 0,
    healthcare_spending: -1, education_spending: -1, public_salary_level: 0, tax_policy_level: 0,
    next_election_time: 100, next_spanish_election_time: 100, countdowns: [],
  });
  return a;
}
beforeEach(() => { vi.spyOn(console, 'info').mockImplementation(() => {}); vi.spyOn(console, 'log').mockImplementation(() => {}); });
afterEach(() => vi.restoreAllMocks());

describe('sovereignty counter-proposals', () => {
  it.each([{cup: 8, surplus: 0}, {cup: 16, surplus: 0}, {cup: 8, surplus: -10}])(
    'preserves ERC cabinet weight with external CUP support: %j', ({cup, surplus}) => {
      const route = 'parlament_coalition_ciu_erc_cup_phase_erc_player_ciu_led_support';
      const a = adapter(route);
      Object.assign(a.qualities, {
        parlament_current_ciu_s: 34, cdc_parlament_s: 34, erc_parlament_s: 33,
        cup_parlament_s: cup, parlament_coalition_support_seats: 67 + cup,
        parlament_coalition_cabinet_seats: 67, speaker_party: 'cdc', generalitat_surplus: surplus,
      });
      a.goToScene(route);
      const posts = ['vp', 'interior', 'economy', 'foreign', 'health', 'education', 'territory', 'bnl'];
      expect(posts.filter(o => a.qualities['parlament_coalition_proposal_' + o] === 'erc'))
        .toEqual(['vp', 'foreign', 'territory', 'bnl']);
      expect(a.qualities.parlament_coalition_leverage).toBe(3);
      expect(a.qualities.parlament_coalition_policy_health).toBe(true);
      expect(a.qualities.parlament_coalition_policy_education).toBe(true);
      a.goToScene(route);
      expect(a.qualities.parlament_coalition_leverage).toBe(3);
      a.goToScene(route + '.vp');
      expect(a.qualities.parlament_coalition_leverage).toBe(23);
      a.goToScene(route + '.vp');
      expect(a.qualities.parlament_coalition_leverage).toBe(3);
    },
  );

  it('does not offer ERC the VP when its seat deficit leaves little leverage', () => {
    const route = 'parlament_coalition_ciu_erc_cup_phase_erc_player_ciu_led_support';
    const a = adapter(route);
    Object.assign(a.qualities, {
      parlament_current_ciu_s: 50, cdc_parlament_s: 50, erc_parlament_s: 20,
      parlament_coalition_support_seats: 80, parlament_coalition_cabinet_seats: 70,
      speaker_party: 'cdc',
    });
    a.goToScene(route);
    expect(a.qualities.parlament_coalition_proposal_vp).toBe('cdc');
    expect(a.qualities.parlament_coalition_leverage).toBeGreaterThanOrEqual(0);
  });

  it.each(routes.filter(route => route.includes('_ciu_led_')))(
    'keeps Mas in the CiU opening and reserves withdrawal demands for CUP counter-offers in %s', route => {
      const a = adapter(route);
      const opening = a.goToScene(route);
      expect(a.qualities[P + 'withdrawal']).toBe(false);
      expect(opening.html).not.toContain('Artur Mas must step down');
      expect(a.qualities.parlament_coalition_proposal_president_name).toBe('Artur Mas');
      a.qualities[P + 'phase'] = 2;
      a.goToScene(route);
      expect(a.qualities[P + 'withdrawal']).toBe(a.qualities.player_party === 'erc');
    },
  );

  it.each(['jxsi', 'jxcat'])('keeps %s together after failed investiture when no dissolution was triggered', list => {
    const route = `parlament_coalition_joint_cup_phase_erc_player_${list}_led`;
    const a = adapter(route);
    a.goToScene(route);
    Object.assign(a.qualities, {
      [P + 'phase']: 3, [list + '_formed']: true, [list + '_dissolved']: false,
      [list + '_dissolution_trigger']: false, ['player_in_' + list]: true,
      ['erc_in_' + list]: true, ['cdc_in_' + list]: true,
      cdc_relations: 90, month_actions: 0,
    });
    a.goToScene('parlament_coalition_sovereignty.failed');
    expect(a.qualities.parlament_repeat_election).toBe(true);
    // Dendry randomly selects among matching go-to entries; exercise the first.
    vi.spyOn(a.engine.random, 'uint32').mockReturnValue(0);
    const frame = a.goToScene('parlament_coalition_sovereignty.continue');
    expect(frame.sceneId).not.toMatch(/jxsi_dissolution|jxcat_dissolution/);
    expect(a.qualities[list + '_dissolved']).toBe(false);
    expect(a.qualities['player_in_' + list]).toBe(true);
    expect(a.qualities[list + '_dissolution_trigger']).toBe(false);
  });

  it.each(['jxsi', 'jxcat'])('requires ERC to remove the CUP withdrawal demand before sending a %s offer', list => {
    const route = `parlament_coalition_joint_cup_phase_erc_player_${list}_led`;
    const a = adapter(route);
    let frame = a.goToScene(route);
    expect(frame.choices.some(c => c.id === route + '.withdrawal')).toBe(false);
    a.qualities[P + 'phase'] = 2;
    frame = a.goToScene(route);
    expect(a.qualities[P + 'withdrawal']).toBe(true);
    expect(a.qualities[P + 'candidate_reason']).toBe('');
    expect(frame.choices.filter(c => c.id === route + '.submit')).toHaveLength(1);
    expect(frame.choices.find(c => c.id === route + '.submit')?.canChoose).toBe(false);
    expect(frame.choices.find(c => c.id === route + '.accept')?.canChoose).toBe(false);
    frame = a.choose(frame.choices.findIndex(c => c.id === route + '.withdrawal'));
    expect(a.qualities[P + 'withdrawal']).toBe(false);
    expect(frame.choices.some(c => c.id === route + '.withdrawal')).toBe(false);
    expect(frame.choices.find(c => c.id === route + '.submit')?.canChoose).toBe(true);
    expect(a.qualities[P + 'candidate_reason']).toContain('CUP maintains its veto');
    // Re-entering the removal action cannot turn it into an ERC demand.
    a.goToScene(route + '.withdrawal');
    expect(a.qualities[P + 'withdrawal']).toBe(false);
  });

  it.each(routes)('shows a single strongest proposed roadmap in %s', route => {
    const a = adapter(route, 'Carles Puigdemont');
    a.goToScene(route);
    Object.assign(a.qualities, {
      parlament_coalition_concessions_indy: 1,
      cdc_roadmap: 3, erc_roadmap: 2, cup_roadmap: 1, jxsi_roadmap: 3, jxcat_roadmap: 3,
    });
    let frame = a.goToScene(route + '.phase_text');
    expect(frame.html.replace(/<[^>]*>/g, '')).toMatch(/Roadmap of\s*Unilateralism/);
    expect(frame.html).not.toContain('Hybrid Confrontation');
    a.qualities.cup_roadmap = 3;
    frame = a.goToScene(route + '.phase_text');
    expect(frame.html.match(/Unilateralism/g)).toHaveLength(1);
  });

  it.each(routes)('rebuilds the next NPC offer instead of copying unaccepted demands in %s', route => {
    const a = adapter(route, 'Carles Puigdemont');
    a.goToScene(route);
    // Draft a different programme and cabinet. The next NPC offer is prepared
    // from the negotiating position, not by preserving these player claims.
    Object.assign(a.qualities, {
      parlament_coalition_policy_health: false,
      parlament_coalition_policy_tax: true,
      parlament_coalition_policy_tax_target: 1,
      parlament_coalition_concessions_indy: 1,
      parlament_coalition_sovereignty_deadline: 2016,
    });
    if (route.includes('ciu_erc_cup') && (route.includes('_erc_player_') || route.endsWith('government'))) {
      a.qualities.parlament_coalition_proposal_vp = a.qualities.player_party;
      a.qualities.parlament_coalition_proposal_economy = a.qualities.player_party;
    }
    a.qualities[P + 'phase'] = 2;
    let frame = a.goToScene(route);
    expect(a.qualities.parlament_coalition_policy_health).toBe(true);
    expect(a.qualities.parlament_coalition_policy_tax).toBe(false);
    expect(a.qualities.parlament_coalition_concessions_indy).toBe(0);
    expect(a.qualities[P + 'deadline']).toBe(0);
    expect(frame.choices.some(c => c.id === route + '.submit')).toBe(false);
    frame = a.goToScene(route + '.policies');
    a.choose(frame.choices.findIndex(c => c.id === 'parlament_coalition_fiscal_policies.health_policy'));
    frame = a.goToScene(route + '.choices');
    expect(frame.choices.find(c => c.id === route + '.submit')?.canChoose).toBe(true);
    expect(frame.choices.find(c => c.id === route + '.accept')?.canChoose).toBe(false);
    a.goToScene(route);
    expect(a.qualities.parlament_coalition_policy_health).toBe(false);
  });

  it('keeps voted roadmap and social commitments in the existing assembly record', () => {
    const route = 'parlament_coalition_joint_cup_phase_erc_player_jxsi_led';
    const a = adapter(route); a.goToScene(route);
    Object.assign(a.qualities, {coalition_phase_timer: 1, parlament_coalition_concessions_indy: 1,
      [P + 'phase']: 2, [P + 'last_terms']: [{policy: 'tax', target: 1}]});
    a.goToScene(P + 'assembly.vote');
    expect(a.qualities[P + 'assembly_terms']).toEqual([{policy: 'tax', target: 1}, {policy: 'indy', target: 1}]);
    Object.assign(a.qualities, {coalition_phase_timer: 0, [P + 'phase']: 3,
      parlament_coalition_policy_tax: false, parlament_coalition_concessions_indy: 0});
    a.goToScene(route);
    expect(a.qualities.parlament_coalition_policy_tax).toBe(true);
    expect(a.qualities.parlament_coalition_policy_tax_target).toBe(1);
    expect(a.qualities.parlament_coalition_concessions_indy).toBe(1);
  });

  it.each(routes)('can accept a received social agreement without a roadmap upgrade in %s', route => {
    const a = adapter(route, 'Carles Puigdemont');
    a.goToScene(route);
    a.qualities[P + 'phase'] = 2;
    const frame = a.goToScene(route);
    expect(a.qualities.parlament_coalition_concessions_indy).toBe(0);
    const accept = frame.choices.findIndex(c => c.id === route + '.accept');
    expect(frame.choices[accept]?.canChoose, String(a.qualities[P + 'candidate_reason'])).toBe(true);
    a.choose(accept);
    expect(a.qualities.parlament_coalition_proposal_accept, String(a.qualities[P + 'reject_reasons'])).toBe(true);
    expect(a.qualities[P + 'phase']).toBe(0);
  });

  it.each(routes)('generates an NPC response that stays stable during player editing in %s', route => {
    const a = adapter(route);
    let frame = a.goToScene(route);
    const q = a.qualities;
    const author = q.player_party === 'erc' && (route.includes('joint_cup') || route.includes('_erc_led_'));
    expect(frame.choices.some(c => c.id === route + '.accept')).toBe(!author);
    expect(frame.choices.some(c => c.id.endsWith('.candidate_impasse'))).toBe(false);
    // A concrete offer is sent through the real submission/round transition.
    if (!q.parlament_coalition_policy_health) a.goToScene('parlament_coalition_fiscal_policies.health_policy');
    const target = q.parlament_coalition_policy_health_target;
    a.goToScene(route + '.submit');
    expect(q[P + 'phase']).toBe(2);
    q.coalition_phase_timer = 0;
    frame = a.goToScene(route);
    expect(frame.html).toContain(q.player_party === 'erc' ? 'following demands' : 'following proposal');
    expect(frame.html).not.toContain('We have amended');
    expect(frame.html).not.toContain('The partners retain');
    expect(q.parlament_coalition_policy_health).toBe(true);
    expect(q.parlament_coalition_policy_health_target).toBe(target);
    const leverage = Number(q.parlament_coalition_leverage);
    a.goToScene(route); // Reopening does not mint leverage or discard the offer.
    expect(q.parlament_coalition_leverage).toBe(leverage);
    const saved = a.exportStateJSON();
    const restored = DendryAdapter.fromJSONText(game); restored.importStateJSON(saved);
    restored.goToScene(route);
    expect(restored.qualities.parlament_coalition_leverage).toBe(leverage);
    if (q.player_party === 'erc') {
      expect(q[P + 'withdrawal']).toBe(true);
      expect(frame.choices.find(c => c.id.endsWith('.accept'))?.canChoose).toBe(false);
      a.goToScene(route + '.withdrawal');
      expect(q[P + 'withdrawal']).toBe(false);
      a.goToScene(route + '.submit');
      expect(q[P + 'phase']).toBe(3);
      expect(q.parlament_coalition_proposal_president_name).toBe('Artur Mas');
      expect(q[P + 'reject_reasons']).toContain('Artur Mas');
    }
  });

  it.each(['jxsi', 'jxcat'])('uses ERC constituent strength, existing costs and cumulative pushes in %s', list => {
    const route = `parlament_coalition_joint_cup_phase_erc_player_${list}_led`;
    const balances = [40, 50, 60].map(share => {
      const a = adapter(route); a.qualities[list + '_formation_num'] = share;
      a.goToScene(route);
      return Number(a.qualities.parlament_coalition_leverage);
    });
    expect(balances[0]).toBeLessThan(balances[1]); expect(balances[1]).toBeLessThan(balances[2]);
    const a = adapter(route); a.goToScene(route);
    const before = Number(a.qualities.parlament_coalition_leverage);
    a.goToScene(route + '.roadmap'); a.goToScene(route + '.roadmap');
    expect(a.qualities.parlament_coalition_leverage).toBe(before);
    a.goToScene(route + '.push');
    expect(a.qualities.parlament_coalition_leverage).toBe(before + 4);
    a.goToScene(route + '.submit'); a.qualities.coalition_phase_timer = 0;
    a.goToScene(route);
    // Two received social commitments cost 10, the push carries forward once.
    expect(a.qualities.parlament_coalition_leverage).toBeCloseTo(before + 4 + 4 - 10);
  });

  it.each(['erc', 'cup'])('makes programme concessions affect the %s assembly without guaranteeing approval', player => {
    const a = adapter(`parlament_coalition_joint_cup_phase_${player}_player_jxsi_led`);
    const q = {...a.qualities, coalition_phase_timer: 1, [P + 'phase']: 3, [P + 'last_terms']: []};
    const generous = {...q, [P + 'last_terms']: [{policy: 'health', target: 0}, {policy: 'education', target: 0}, {policy: 'tax', target: 1}]};
    action(P + 'assembly', 'vote', q); action(P + 'assembly', 'vote', generous);
    expect(Number(generous[P + 'assembly_yes'])).toBeGreaterThan(Number(q[P + 'assembly_yes']));
    expect(generous[P + 'assembly_result']).toBe('veto');
  });

  it.each(['approve', 'tie', 'veto'])('announces replacement separately before the final round after %s', result => {
    const route = 'parlament_coalition_joint_cup_phase_erc_player_jxsi_led';
    const a = adapter(route); a.goToScene(route);
    Object.assign(a.qualities, {[P + 'phase']: 3, [P + 'assembly_result']: result, [P + 'ciu_pressure']: 3,
      [P + 'dispute_round']: 3, year: 2013, time: 6, roadmap_year: 2015, month_actions: 1, coalition_phase_timer: 1});
    const month = a.qualities.month;
    let frame = a.goToScene('post_event');

    const replacement = frame.choices.findIndex(c => c.id === P + 'replacement');
    expect(replacement >= 0).toBe(result !== 'approve');
    if (replacement >= 0) {
      expect(frame.choices.some(c => c.id === route)).toBe(false);
      frame = a.choose(replacement);
      frame = a.choose(frame.choices.findIndex(c => c.id === P + 'replacement.puigdemont'));
      expect(a.qualities.parlament_coalition_proposal_president_name).toBe('Carles Puigdemont');
      expect(frame.choices.some(c => c.id === route)).toBe(true);
      expect(frame.choices.some(c => c.id === P + 'replacement')).toBe(false);
    }
    expect(a.qualities.month).toBe(Number(month) + 1);
  });

  it('refreshes an early assembly decision after later pressure, without reevaluating it on menu visits', () => {
    const route = 'parlament_coalition_joint_cup_phase_cup_player_jxsi_led';
    const a = adapter(route); a.goToScene(route);
    Object.assign(a.qualities, {[P + 'phase']: 2, [P + 'assembly_result']: 'veto', coalition_phase_timer: 1});
    a.goToScene(P + 'assembly.withdrawal');
    expect(a.qualities[P + 'replacement_needed']).toBe(false);

    Object.assign(a.qualities, {[P + 'phase']: 3, [P + 'ciu_pressure']: 3, [P + 'dispute_round']: 3, year: 2013, time: 6, roadmap_year: 2015, coalition_phase_timer: 1, month_actions: 1});
    const arrivals = vi.spyOn(a.engine, '_runActions');
    a.goToScene('post_event');
    expect(a.qualities[P + 'replacement_needed']).toBe(true);
    const checks = () => arrivals.mock.calls.filter(args => args[1] === 'sovereignty withdrawal').length;
    expect(checks()).toBe(1);
    a.goToScene('post_event');
    expect(checks()).toBe(1);
    a.goToScene(P + 'replacement.check');
    expect(a.qualities[P + 'replacement_needed']).toBe(true);
  });

  it('marks fiscal amendments and refunds only the cost included in the received offer', () => {
    const route = 'parlament_coalition_joint_cup_phase_cup_player_jxsi_led';
    const a = adapter(route, 'Carles Puigdemont'); a.goToScene(route);
    const before = Number(a.qualities.parlament_coalition_leverage);
    const target = a.qualities.parlament_coalition_policy_health_target;
    let frame = a.goToScene(route + '.policies');
    frame = a.choose(frame.choices.findIndex(c => c.id === 'parlament_coalition_fiscal_policies.health_policy'));
    expect(a.qualities.parlament_coalition_leverage).toBe(before + 5);
    frame = a.goToScene(route + '.choices');
    expect(frame.choices.find(c => c.id.endsWith('.accept'))?.canChoose).toBe(false);
    frame = a.goToScene(route + '.policies');
    a.choose(frame.choices.findIndex(c => c.id === 'parlament_coalition_fiscal_policies.health_policy'));
    expect(a.qualities.parlament_coalition_leverage).toBe(before);
    expect(a.qualities.parlament_coalition_policy_health_target).toBe(target);
  });
});
