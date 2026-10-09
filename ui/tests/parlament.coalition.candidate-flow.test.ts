import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const P = 'parlament_coalition_sovereignty_';
const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');
const source = (name: string) => readFileSync(resolve(import.meta.dirname, '../../source/scenes/events/elections', name + '.scene.dry'), 'utf8');
function action(name: string, marker: string, q: Record<string, any>, draw = vi.fn(() => 0.5), achieve = vi.fn()) {
  const s = source(name);
  const index = marker ? s.search(new RegExp(`^@${marker}\\r?$`, 'm')) : 0;
  expect(index).toBeGreaterThanOrEqual(0);
  const node = s.slice(index).split(/\r?\n@/)[0];
  const code = node.match(/on-arrival:\s*\{!([\s\S]*?)!\}/)![1];
  new Function('Q', 'G', code).call({ achieve }, q, { random: draw });
  return draw;
}
function nominate(q: Record<string, any>, achieve = vi.fn()) {
  action(P + 'replacement', 'check', q);
  action(P + 'replacement', 'commit', q, vi.fn(() => 0.5), achieve);
}
function evaluate(q: Record<string, any>) {
  const draw = action(P + 'replacement', 'check', q);
  action(P + 'replacement', 'evaluate', q, draw);
  return draw;
}
function qbase(overrides: Record<string, any> = {}) {
  return {
    player_party: 'erc', rubicon: false, year: 2015, president: 'Artur Mas',
    coalition_phase_timer: 1, party_resources: 3,
    parlament_current_ciu: 'cdc', cdc_leader: 'Artur Mas', cdc_relations: 75,
    erc_relations: 75, cup_relations: 75, cup_faction: 'Poble Lliure',
    cup_poblelliure_strength: 50, cup_endavant_strength: 30, cup_thirdsector_strength: 20,
    cup_poblelliure_dissent: 15, cup_endavant_dissent: 15, cup_thirdsector_dissent: 15,
    erc_parlament_s: 0, cdc_parlament_s: 0, cup_parlament_s: 10,
    jxsi_parlament_s: 62, jxsi_formation_num: 50,
    parlament_coalition_proposal_president_name: 'Artur Mas',
    parlament_coalition_concessions_indy: 1,
    parlament_coalition_policy_education: true, parlament_coalition_policy_education_target: 0,
    parlament_coalition_policy_health: true, parlament_coalition_policy_health_target: 0,
    [P + 'phase']: 3, [P + 'kind']: 'jxsi',
    [P + 'dispute_round']: 0, [P + 'ciu_pressure']: 3, [P + 'mas_pressure']: 3,
    [P + 'brokerage']: 1, [P + 'deadline']: 2017, [P + 'last_terms']: [],
    ...overrides,
  };
}
let errors: unknown[][];
beforeEach(() => {
  errors = [];
  vi.spyOn(console, 'error').mockImplementation((...args) => { if (!String(args[0]).startsWith('Page loaded from unexpected origin')) errors.push(args); });
  vi.spyOn(console, 'info').mockImplementation(() => {});
  vi.spyOn(console, 'log').mockImplementation(() => {});
});
afterEach(() => { vi.restoreAllMocks(); expect(errors).toEqual([]); });

describe('presidential pressure, assembly and nomination', () => {
  it.each(['jxsi', 'jxcat', 'support'])('names the party retaining Mas in CUP %s negotiations', kind => {
    const q = qbase({ player_party: 'cup', [P + 'kind']: kind, [P + 'phase']: 2, [P + 'assembly_result']: 'veto' });
    evaluate(q);
    const partner = kind === 'jxsi' ? 'JxSí' : kind === 'jxcat' ? 'JxCat' : 'CDC';
    expect(q[P + 'candidate_reason']).toContain(partner + ' is unwilling to withdraw Artur Mas');
    expect(q[P + 'candidate_reason']).toContain('assembly has voted against');
  });

  it.each([false, true])('submits a selectable excessive CUP demand, with withdrawal %s', withdraw => {
    const a = DendryAdapter.fromJSONText(gameText);
    a.beginGame([1, 2, 3, 4]); a.goToScene('root.start_menu'); a.goToScene('root.start');
    const phase = 'parlament_coalition_joint_cup_phase_cup_player_jxsi_led';
    Object.assign(a.qualities, qbase({ player_party: 'cup', coalition_phase_timer: 0, [P + 'phase']: 1,
      [P + 'cabinet']: ['jxsi'], parlament_coalition_leader: 'jxsi', parlament_coalition_follower: 'cup',
      parlament_coalition_support_seats: 72, parlament_coalition_cabinet_seats: 62,
      jxsi_leader: 'Carles Puigdemont', cdc_leader: 'Carles Puigdemont',
      parlament_coalition_proposal_president_name: 'Carles Puigdemont', healthcare_spending: -1,
      generalitat_surplus: 0, social_spending_cap: 0,
    }));
    a.goToScene(phase);
    a.qualities.parlament_coalition_leverage = 0;
    const menu = a.goToScene(phase + '.policies');
    const index = menu.choices.findIndex(c => c.id === 'parlament_coalition_fiscal_policies.health_policy');
    expect(index).toBeGreaterThanOrEqual(0);
    a.choose(index);
    expect(a.qualities.parlament_coalition_leverage).toBe(-5);
    if (withdraw) {
      const revised = a.goToScene(phase + '.policies');
      a.choose(revised.choices.findIndex(c => c.id === 'parlament_coalition_fiscal_policies.health_policy'));
      expect(a.qualities.parlament_coalition_leverage).toBe(0);
    }
    a.goToScene(phase + '.submit');
    if (withdraw) {
      expect(a.qualities.parlament_coalition_proposal_accept).toBe(true);
      expect(a.qualities[P + 'phase']).toBe(0);
      expect(a.qualities.cat_coalition_support).toEqual(['cup']);
      return;
    }
    expect(a.qualities.parlament_coalition_proposal_accept).toBe(false);
    expect(a.qualities[P + 'phase']).toBe(2);
    expect(a.qualities[P + 'reject_reasons']).toContain('JxSí is unwilling');
    expect(a.qualities.parlament_coalition_policy_reject_reasons).toBe('');
    a.qualities.coalition_phase_timer = 0;
    const next = a.goToScene(phase);
    expect(next.html).toContain('The previous round ended without agreement');
    expect(next.html).toContain('JxSí is unwilling');
    expect(next.html).not.toContain('has rejected our previous');
  });

  it.each(['jxsi', 'jxcat', 'government', 'support'])('matches public pressure and event copy to the current %s talks', kind => {
    const joint = kind === 'jxsi' || kind === 'jxcat';
    const q = qbase({ [P + 'kind']: kind, independence_trust: 80, [P + 'feud']: 0 });
    action(P + 'candidate_erc', 'public', q);
    expect(q.independence_trust).toBe(joint ? 72 : 76);
    expect(q.cdc_relations).toBe(joint ? 66 : 69);
    expect(q[P + 'feud']).toBe(1);
    expect(q[P + 'ciu_pressure']).toBe(5);

    const a = DendryAdapter.fromJSONText(gameText);
    a.beginGame([1, 2, 3, 4]); a.goToScene('root.start_menu'); a.goToScene('root.start');
    Object.assign(a.qualities, qbase({ [P + 'kind']: kind, [P + 'assembly_result']: 'veto', parlament_coalition_leader: joint ? kind : 'cdc', jxsi_formed: true, jxcat_formed: true }));
    const pressure = a.goToScene(P + 'candidate_erc.public');
    expect(pressure.html.includes('breaking the coalition')).toBe(joint);
    expect(pressure.html.includes('never agreed to stand together')).toBe(!joint);
    const replacement = a.goToScene(P + 'replacement');
    expect(replacement.html.includes('for the good of the unity list')).toBe(joint);
    expect(replacement.html.includes('to make an agreement between')).toBe(!joint);
  });

  it.each(['erc', 'cup'])('requires both other partners above 65 and a seat advantage for standalone %s to nominate Romeva', player => {
    const other = player === 'erc' ? 'cup' : 'erc';
    const q = qbase({ player_party: player, [P + 'kind']: 'government', [P + 'assembly_result']: 'veto', [P + 'nominee']: 'Raül Romeva', cdc_parlament_s: 40, erc_parlament_s: 31, cdc_relations: 66, [other + '_relations']: 66, [player + '_relations']: 0 });
    action(P + 'replacement', 'check', q);
    expect(q[P + 'romeva_available']).toBe(true);
    for (const field of ['cdc_relations', other + '_relations']) {
      q[field] = 65;
      action(P + 'replacement', 'check', q);
      expect(q[P + 'romeva_available']).toBe(false);
      q[field] = 66;
    }
    q.erc_parlament_s = 30;
    action(P + 'replacement', 'check', q);
    expect(q[P + 'romeva_available']).toBe(false);
    q.erc_parlament_s = 31;
    nominate(q);
    expect(q.parlament_coalition_proposal_president_name).toBe('Raül Romeva');
    expect(q.party_resources).toBe(2);
  });

  it.each(['jxsi', 'jxcat'].flatMap(list => ['cdc', 'pdcat'].map(party => [list, party])))(
    'uses canonical %s constituent %s from the menu through pressure and nomination', (list, party) => {
      const a = DendryAdapter.fromJSONText(gameText);
      a.beginGame([1, 2, 3, 4]); a.goToScene('root.start_menu'); a.goToScene('root.start');
      const seats = Object.fromEntries((a.qualities.parties as string[]).map(p => [p + '_parlament_s', 0]));
      const memberships = Object.fromEntries(['jxsi', 'jxcat'].flatMap(name =>
        ['ciu', 'cdc', 'dl', 'pdcat', 'junts'].map(member => [`${member}_in_${name}`, false]),
      ));
      Object.assign(a.qualities, seats, memberships, {
        player_party: 'erc', parlament_current_ciu: 'junts', rubicon: false,
        [`${list}_parlament_s`]: 62, [`${list}_formation_num`]: 50,
        [`${list}_leader`]: 'Artur Mas', [`${list}_leader_gender`]: 'm',
        [`${party}_in_${list}`]: true, [`${party}_relations`]: 75,
        [`${party}_leader`]: 'Artur Mas', erc_relations: 75, cup_relations: 75,
        erc_in_jxsi: list === 'jxsi', erc_in_jxcat: list === 'jxcat',
        cup_in_jxsi: false, cup_in_jxcat: false, cup_parlament_s: 10,
        junts_relations: 15, junts_leader: 'Unrelated leader',
      });
      const menu = a.goToScene('parlament_coalition');
      expect(a.qualities.parlament_current_ciu).toBe(party);
      const index = menu.choices.findIndex(c => c.id === `parlament_coalition_joint_cup.erc_player_${list}_led`);
      expect(index).toBeGreaterThanOrEqual(0);
      expect(menu.choices[index].canChoose).toBe(true);
      a.choose(index);
      const restored = DendryAdapter.fromJSONText(gameText);
      restored.importStateJSON(a.exportStateJSON());
      const q = restored.qualities;
      expect(q.parlament_current_ciu).toBe(party);
      action('parlament_coalition_sovereignty_candidate_erc', 'private', q);
      expect(q[party + '_relations']).toBe(73);
      expect(q.junts_relations).toBe(15);
      Object.assign(q, {
        [P + 'phase']: 3, [P + 'assembly_result']: 'veto', [P + 'ciu_pressure']: 3,
        [P + 'brokerage']: true, [P + 'nominee']: 'Raül Romeva', party_resources: 3,
      });
      nominate(q);
      expect(q.parlament_coalition_proposal_president_name).toBe('Raül Romeva');
      expect(q[list + '_leader']).toBe('Raül Romeva');
      expect(q[party + '_leader']).toBe('Carles Puigdemont');
      expect(q.junts_leader).toBe('Unrelated leader');
    },
  );

  it('shares the campaign/defer interval guard without charging twice', () => {
    const q = qbase({ player_party: 'cup', [P + 'phase']: 1 });
    action(P + 'assembly', 'campaign', q);
    expect(q.party_resources).toBe(2);
    expect(q[P + 'assembly_campaign']).toBe(1);
    action(P + 'assembly', 'campaign', q);
    expect(q.party_resources).toBe(2);
    q[P + 'phase'] = 2;
    action(P + 'assembly', 'defer', q);
    action(P + 'assembly', 'campaign', q);
    expect(q.party_resources).toBe(2);
    expect(q[P + 'assembly_campaign']).toBe(1);
    q[P + 'phase'] = 3;
    action(P + 'assembly', 'campaign', q);
    expect(q.party_resources).toBe(2);
  });

  it.each([
    ['cup', false, true], ['cup', true, false], ['erc', false, false],
  ])('awards Mas withdrawal only to CUP outside JxSi: %s, membership %s', (player, member, expected) => {
    const q = qbase({ player_party: player, cup_in_jxsi: member, player_in_jxsi: member, [P + 'assembly_result']: 'veto', [P + 'nominee']: 'Carles Puigdemont' });
    const achieve = vi.fn();
    nominate(q, achieve);
    expect(achieve.mock.calls).toEqual(expected ? [['paperera_de_la_historia']] : []);
    nominate(q, achieve);
    expect(achieve).toHaveBeenCalledTimes(expected ? 1 : 0);
  });

  it.each([
    [{}, true],
    [{ [P + 'kind']: 'jxcat' }, false],
    [{ [P + 'phase']: 2 }, false],
    [{ [P + 'ciu_pressure']: 1 }, false],
    [{ [P + 'mas_pressure']: 0 }, false],
    [{ [P + 'assembly_result']: 'approve' }, false],
    [{ parlament_coalition_proposal_president_name: 'Neus Munté' }, false],
    [{ player_party: 'cup' }, false],
    [{ player_in_jxsi: false }, false],
  ])('awards Mas o Marc only for persisting with Mas into a failed JxSi investiture: %j', (overrides, expected) => {
    const q = qbase({ player_in_jxsi: true, [P + 'ciu_pressure']: 0, [P + 'assembly_result']: 'veto', ...overrides });
    const achieve = vi.fn();
    action('parlament_coalition_sovereignty', 'failed', q, vi.fn(() => 0.5), achieve);
    expect(q.parlament_repeat_election).toBe(true);
    expect(achieve.mock.calls).toEqual(expected ? [['mas_o_marc']] : []);
    action('parlament_coalition_sovereignty', 'failed', q, vi.fn(() => 0.5), achieve);
    expect(achieve).toHaveBeenCalledTimes(expected ? 1 : 0);
  });

  it('compounds pressure in all three intervals without randomness or replacement', () => {
    const q = qbase({ [P + 'ciu_pressure']: 0 });
    for (const round of [1, 2, 3]) {
      q[P + 'phase'] = round;
      const draw = action('parlament_coalition_sovereignty_candidate_erc', 'private', q);
      expect(draw).not.toHaveBeenCalled();
      expect(q[P + 'ciu_pressure']).toBe(round);
      action('parlament_coalition_sovereignty_candidate_erc', 'private', q);
      expect(q[P + 'ciu_pressure']).toBe(round);
      expect(q.parlament_coalition_proposal_president_name).toBe('Artur Mas');
    }
  });

  it.each([['veto', 20, 60, 0.5], ['approve', 70, 10, 0.5], ['tie', 50, 30, 0.0625]])
    ('NPC ballot produces %s from faction strength and one saved draw', (result, pl, en, roll) => {
      const q = qbase({ cup_poblelliure_strength: pl, cup_endavant_strength: en });
      const draw = vi.fn(() => Number(roll));
      action('parlament_coalition_sovereignty_assembly', 'ballot', q, draw);
      expect(q[P + 'assembly_result']).toBe(result);
      expect(q[P + 'assembly_yes']).toBeGreaterThanOrEqual(0);
      expect(q[P + 'assembly_yes']).toBeLessThanOrEqual(3034);
      const saved = JSON.parse(JSON.stringify(q));
      action('parlament_coalition_sovereignty_assembly', 'ballot', saved, draw);
      expect(draw).toHaveBeenCalledTimes(1);
      expect(saved[P + 'assembly_yes']).toBe(q[P + 'assembly_yes']);
    });

  it('NPC ballot ignores hidden faction dissent', () => {
    const a = qbase(), b = qbase({ cup_poblelliure_dissent: 100, cup_endavant_dissent: 100, cup_thirdsector_dissent: 100 });
    action('parlament_coalition_sovereignty_assembly', 'ballot', a);
    action('parlament_coalition_sovereignty_assembly', 'ballot', b);
    expect(a[P + 'assembly_yes']).toBe(b[P + 'assembly_yes']);
  });

  it('a CUP player cannot approve Mas without a mandate, and must retain the voted terms', () => {
    const q = qbase({ player_party: 'cup', parlament_coalition_proposal_accept: true, parlament_coalition_sovereignty_reject_reasons: '' });
    evaluate(q);
    expect(q.parlament_coalition_proposal_accept).toBe(false);
    Object.assign(q, { [P + 'assembly_result']: 'approve', [P + 'assembly_deadline']: 2017, [P + 'assembly_terms']: [{ policy: 'health', target: 0 }], parlament_coalition_proposal_accept: true });
    evaluate(q);
    expect(q.parlament_coalition_proposal_accept).toBe(true);
    q.parlament_coalition_policy_health = false;
    evaluate(q);
    expect(q.parlament_coalition_proposal_accept).toBe(false);
  });

  it('lets ERC knowingly nominate Munte even though the CUP will reject her', () => {
    const q = qbase({ [P + 'assembly_result']: 'veto', [P + 'nominee']: 'Neus Munté' });
    nominate(q);
    expect(q.parlament_coalition_proposal_president_name).toBe('Neus Munté');
    q.parlament_coalition_proposal_accept = true;
    evaluate(q);
    expect(q.parlament_coalition_proposal_accept).toBe(false);
  });

  it.each(['erc', 'cup'].flatMap(player => ['Artur Mas', 'Neus Munté'].flatMap(name =>
    ['', 'veto', 'tie', 'approve'].map(result => ({ player, name, result }))
  )))('explains candidate feedback for $player, $name and assembly "$result"', ({ player, name, result }) => {
    const q = qbase({
      player_party: player, parlament_coalition_proposal_president_name: name,
      cup_faction: 'Endavant', cdc_relations: 59,
      [P + 'assembly_result']: result, [P + 'assembly_deadline']: 2017,
      parlament_coalition_proposal_accept: false,
      [P + 'reject_reasons']: 'The proposed cabinet is incomplete.',
    });
    evaluate(q);
    expect(q.parlament_coalition_proposal_accept).toBe(false);
    const explanation = q[P + 'candidate_reason'];
    if (result === 'approve' && name === 'Artur Mas') {
      expect(explanation).toBe('');
      expect(q[P + 'reject_reasons']).toBe('The proposed cabinet is incomplete.');
      return;
    }
    expect(explanation).toContain(name);
    expect(q[P + 'reject_reasons']).toBe('The proposed cabinet is incomplete. ' + explanation);
    if (!result) {
      expect(explanation).not.toMatch(/has voted|ended in a tie|assembly.*(?:rejected|authorized)/);
    } else if (result === 'veto') {
      expect(explanation).toMatch(/assembly.*(?:voted against|rejected)/);
    } else if (result === 'tie') {
      expect(explanation).toMatch(/\btie\b/);
      expect(explanation).not.toContain('voted against');
    } else {
      expect(explanation).not.toMatch(/voted against|assembly.*rejected/);
    }
    if (name === 'Neus Munté') {
      expect(explanation).toMatch(/Artur Mas|CDC/);
      expect(explanation).not.toContain('voted against the investiture of Neus Munté');
    }
  });

  it('explains outstanding exceptional-investiture terms together without inventing an assembly vote', () => {
    const q = qbase({
      [P + 'assembly_result']: '', parlament_coalition_proposal_accept: true,
      [P + 'reject_reasons']: '', parlament_coalition_concessions_indy: 0,
      parlament_coalition_policy_education: false, parlament_coalition_policy_health: false,
      [P + 'deadline']: 2019,
    });
    evaluate(q);
    expect(q.parlament_coalition_proposal_accept).toBe(false);
    expect(q[P + 'reject_reasons']).toContain('independence');
    expect(q[P + 'reject_reasons']).toContain('social commitments');
    expect(q[P + 'reject_reasons']).toContain('next two years');
    expect(q[P + 'reject_reasons']).not.toContain('assembly');
  });

  it.each(['approve', 'tie'])('explains weakening the programme put to an assembly with result %s', result => {
    const q = qbase({
      parlament_coalition_proposal_president_name: result === 'tie' ? 'Neus Munté' : 'Artur Mas',
      cup_poblelliure_strength: 60, parlament_coalition_policy_tax: true,
      [P + 'assembly_result']: result, [P + 'assembly_deadline']: 2016,
      [P + 'assembly_terms']: [{ policy: 'health', target: 1 }],
      parlament_coalition_proposal_accept: true, [P + 'reject_reasons']: '',
    });
    evaluate(q);
    expect(q.parlament_coalition_proposal_accept).toBe(false);
    expect(q[P + 'reject_reasons']).toContain('weakens social commitments');
    expect(q[P + 'reject_reasons']).toContain('pushes the roadmap deadline');
    expect(q[P + 'reject_reasons']).not.toContain('voted against');
  });

  it.each([
    ['Artur Mas', 45, 60, 2, true],
    ['Artur Mas', 46, 60, 2, true],
    ['Artur Mas', 46, 59, 2, true],
    ['Artur Mas', 46, 60, 1, false],
    ['Neus Munté', 50, 65, 3, false],
    ['Neus Munté', 51, 65, 3, true],
    ['Neus Munté', 51, 64, 3, false],
    ['Neus Munté', 51, 65, 2, false],
  ])('checks deterministic NPC acceptance for %s (PL %s, relations %s, policies %s)', (name, strength, relations, policies, accepted) => {
    const q = qbase({
      parlament_coalition_proposal_president_name: name,
      cup_poblelliure_strength: strength, cup_relations: relations,
      cup_poblelliure_dissent: 100, cup_thirdsector_dissent: 100,
      parlament_coalition_policy_education: Number(policies) >= 2,
      parlament_coalition_policy_tax: Number(policies) >= 3,
      parlament_coalition_proposal_accept: true,
      [P + 'assembly_result']: 'approve', [P + 'assembly_deadline']: 2017,
    });
    const draw = evaluate(q);
    expect(q.parlament_coalition_proposal_accept).toBe(accepted);
    expect(draw).not.toHaveBeenCalled();
  });

  it.each(['approve', 'veto', 'tie'])('CUP Munte nomination respects its %s mandate', result => {
    const q = qbase({ player_party: 'cup', [P + 'assembly_result']: result, cup_poblelliure_strength: 60 });
    action('parlament_coalition_sovereignty_replacement', 'check', q);
    expect(q[P + 'munte_available']).toBe(result === 'tie');
  });

  it('does not replace Mas without withdrawal pressure, or after an authorized agreement', () => {
    const q = qbase({ [P + 'assembly_result']: 'veto', [P + 'ciu_pressure']: 0 });
    action('parlament_coalition_sovereignty_replacement', 'check', q);
    expect(q[P + 'replacement_needed']).toBe(false);
    q[P + 'assembly_result'] = 'approve'; q[P + 'ciu_pressure'] = 4;
    action('parlament_coalition_sovereignty_replacement', 'check', q);
    expect(q[P + 'replacement_needed']).toBe(false);
    Object.assign(q, { player_party: 'cup', cdc_relations: 45 });
    action('parlament_coalition_sovereignty_replacement', 'check', q);
    expect(q[P + 'replacement_needed']).toBe(false);
  });

  it.each(['erc', 'cup'])('honours the assembly approval of Mas for %s despite political resistance', player => {
    const q = qbase({
      player_party: player, [P + 'assembly_result']: 'approve', [P + 'assembly_deadline']: 2017,
      [P + 'mas_pressure']: 0, [P + 'ciu_pressure']: 5,
      cup_faction: 'Endavant', cup_poblelliure_strength: 0, cup_endavant_strength: 100,
      cdc_relations: 0, erc_relations: 0, cup_relations: 0,
      parlament_coalition_mas_veto_failures: 5,
      parlament_coalition_proposal_accept: true, [P + 'reject_reasons']: '',
    });
    evaluate(q);
    expect(q[P + 'replacement_needed']).toBe(false);
    expect(q[P + 'candidate_reason']).toBe('');
    expect(q.parlament_coalition_proposal_accept).toBe(true);
    expect(q.parlament_coalition_proposal_president_name).toBe('Artur Mas');
    q.parlament_coalition_concessions_indy = 0;
    evaluate(q);
    expect(q.parlament_coalition_proposal_accept).toBe(false);
    expect(q[P + 'candidate_reason']).toContain('independence');
    expect(q[P + 'replacement_needed']).toBe(false);
  });

  it.each(['tie', 'veto'].flatMap(result => [
    ['erc', result, 60, 1, false], ['erc', result, 60, 2, true],
    ['erc', result, 59, 2, false], ['erc', result, 59, 3, true],
    ['cup', result, 60, 0, true], ['cup', result, 59, 0, true],
  ]))('preserves the replacement decision for %s, %s, CiU relations %s and pressure %s', (player, result, relations, pressure, expected) => {
    const q = qbase({
      player_party: player, [P + 'assembly_result']: result,
      cdc_relations: relations, [P + 'ciu_pressure']: pressure,
    });
    action('parlament_coalition_sovereignty_replacement', 'check', q);
    expect(q[P + 'replacement_needed']).toBe(expected);
  });

  it('charges Romeva once and uses constituent influence inside a joint list', () => {
    const q = qbase({ [P + 'assembly_result']: 'veto', [P + 'nominee']: 'Raül Romeva' });
    nominate(q);
    expect(q.parlament_coalition_proposal_president_name).toBe('Raül Romeva');
    expect(q.cdc_leader).toBe('Carles Puigdemont');
    expect(q.party_resources).toBe(2);
    nominate(q);
    expect(q.party_resources).toBe(2);
  });

  it.each([
    ['approve', 0, true], ['approve', 2, false], ['veto', 0, false],
  ])('submits through candidate and fiscal checks: mandate %s, health target %s', (mandate, target, accepted) => {
    const a = DendryAdapter.fromJSONText(gameText);
    a.beginGame([1, 2, 3, 4]); a.goToScene('root.start_menu'); a.goToScene('root.start');
    const phase = 'parlament_coalition_joint_cup_phase_erc_player_jxsi_led';
    Object.assign(a.qualities, qbase({
      coalition_phase_timer: 0, [P + 'phase']: 2,
      [P + 'assembly_result']: mandate, [P + 'assembly_deadline']: 2017,
      [P + 'cabinet']: ['jxsi'], parlament_coalition_cabinet_seats: 62,
      parlament_coalition_leader: 'jxsi', parlament_coalition_follower: 'cup',
      education_spending: -1, healthcare_spending: -1,
      ...(mandate === 'approve' ? {
        cup_faction: 'Endavant', cup_poblelliure_strength: 0, cup_relations: 0, [P + 'mas_pressure']: 0,
      } : {}),
    }));
    const frame = a.goToScene(phase);
    Object.assign(a.qualities, {
      parlament_coalition_concessions_indy: 1, [P + 'deadline']: 2017,
      parlament_coalition_policy_education: true, parlament_coalition_policy_education_target: 0,
      parlament_coalition_policy_health: true, parlament_coalition_policy_health_target: target,
    });
    a.choose(frame.choices.findIndex(c => c.id === phase + '.submit'));
    expect(a.qualities.parlament_coalition_proposal_accept).toBe(accepted);
    if (accepted) {
      expect(a.qualities.cat_coalition).toEqual(['jxsi']);
      expect(a.qualities[P + 'phase']).toBe(0);
    } else if (mandate === 'veto') {
      expect(a.qualities[P + 'reject_reasons']).toContain('assembly has voted against the investiture of Artur Mas');
    } else {
      expect(a.qualities.parlament_coalition_policy_reject_reasons).toContain('healthcare funding');
    }
  });

  it.each(['erc', 'cup'].flatMap(player => [
    [`parlament_coalition_joint_cup_phase_${player}_player_jxsi_led`, player, 'jxsi', 'jxsi'],
    [`parlament_coalition_joint_cup_phase_${player}_player_jxcat_led`, player, 'jxcat', 'jxcat'],
    [`parlament_coalition_ciu_erc_cup_phase_${player}_player_ciu_led_support`, player, 'support', 'cdc'],
    [`parlament_coalition_ciu_erc_cup_phase_${player}_player_ciu_led_government`, player, 'government', 'cdc'],
  ]).flatMap(route => ['approve', 'tie', 'veto'].map(result => [...route, result])))('routes the final offer for %s (%s player, %s kind, %s leader, %s assembly)', (phase, player, kind, leader, result) => {
    const a = DendryAdapter.fromJSONText(gameText);
    a.beginGame([1, 2, 3, 4]); a.goToScene('root.start_menu'); a.goToScene('root.start');
    Object.assign(a.qualities, qbase({ player_party: player, coalition_phase_timer: 0, [P + 'kind']: kind, [P + 'assembly_result']: result, parlament_coalition_leader: leader, parlament_coalition_follower: 'erc', parlament_coalition_policy_health: true, [P + 'mas_pressure']: 0 }));
    const frame = a.goToScene(phase);
    if (result === 'approve') {
      expect(frame.choices.some(c => c.id.startsWith('parlament_coalition_sovereignty_replacement.'))).toBe(false);
      expect(frame.choices.some(c => c.id === phase + '.submit')).toBe(true);
      expect(a.qualities.parlament_coalition_proposal_president_name).toBe('Artur Mas');
      return;
    }
    expect(frame.choices.some(c => c.id === 'parlament_coalition_sovereignty_replacement.puigdemont')).toBe(true);
    expect(a.qualities.parlament_coalition_policy_health).toBe(true);
    const after = a.choose(frame.choices.findIndex(c => c.id.endsWith('.puigdemont')));
    expect(a.qualities.parlament_coalition_proposal_president_name).toBe('Carles Puigdemont');
    expect(a.qualities.parlament_coalition_policy_health).toBe(false);
    expect(after.choices.some(c => c.id === phase + '.submit')).toBe(true);
    expect(a.qualities.cat_coalition).not.toEqual(['jxsi']);
  });

  it('forces the CUP ballot in the last interval while ERC sees its own result event', () => {
    const a = DendryAdapter.fromJSONText(gameText);
    a.beginGame([1, 2, 3, 4]); a.goToScene('root.start_menu'); a.goToScene('root.start');
    Object.assign(a.qualities, qbase({ player_party: 'cup', [P + 'dispute_round']: 3 }));
    const frame = a.goToScene('parlament_coalition_sovereignty_assembly');
    expect(frame.choices.find(c => c.id.endsWith('.defer'))?.canChoose).toBe(false);
    expect(frame.choices.find(c => c.id.endsWith('.campaign'))?.canChoose).toBe(false);
    expect(frame.choices.find(c => c.id.endsWith('.vote'))?.canChoose).toBe(true);
    a.qualities.player_party = 'erc';
    const report = a.goToScene('parlament_coalition_sovereignty_assembly_erc');
    expect(report.html).toContain('membership has voted');
    expect(a.qualities[P + 'assembly_result']).toBeTruthy();
    const saved = a.exportStateJSON();
    const restored = DendryAdapter.fromJSONText(gameText); restored.importStateJSON(saved);
    const result = restored.qualities[P + 'assembly_yes'];
    restored.goToScene('parlament_coalition_sovereignty_assembly_erc');
    expect(restored.qualities[P + 'assembly_yes']).toBe(result);
  });

  it('queues ERC pressure, then the CUP result, before consuming the final interval', () => {
    const a = DendryAdapter.fromJSONText(gameText);
    a.beginGame([1, 2, 3, 4]); a.goToScene('root.start_menu'); a.goToScene('root.start');
    Object.assign(a.qualities, qbase({
      year: 2013, month: 2, time: 6, month_actions: 1, countdowns: [],
      next_election_time: 100, next_spanish_election_time: 100,
      cup_leader: 'David Fernàndez', cup_poblelliure_strength: 20, cup_endavant_strength: 60,
      [P + 'dispute_round']: 2, [P + 'mediation_round']: 3,
      parlament_coalition_leader: 'jxsi', parlament_coalition_follower: 'cup',
    }));
    let frame = a.goToScene('post_event');
    const pick = (id: string) => {
      const index = frame.choices.findIndex(c => c.id === id && c.canChoose);
      expect(index, id).toBeGreaterThanOrEqual(0);
      frame = a.choose(index);
    };
    pick('parlament_coalition_sovereignty_candidate_erc');
    pick('parlament_coalition_sovereignty_candidate_erc.private');
    expect(a.qualities.month).toBe(2);
    pick('parlament_coalition_sovereignty_assembly_erc');
    expect(a.qualities[P + 'assembly_result']).toBe('veto');
    expect(a.qualities.month).toBe(2);
    pick('post_event');
    expect(a.qualities.month).toBe(3);
    expect(a.qualities.coalition_phase_timer).toBe(0);
    expect(frame.choices.some(c => c.id === 'parlament_coalition_joint_cup_phase_erc_player_jxsi_led')).toBe(true);
  });
});
