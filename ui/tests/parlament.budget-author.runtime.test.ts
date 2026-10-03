import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');

function setup(author = 'ciu', overrides: Record<string, any> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start_menu');
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string, any>;
  for (const key of Object.keys(q)) if (key.endsWith('_parlament_s')) q[key] = 0;
  Object.assign(q, {
    player_party: author === 'erc' ? 'cup' : 'erc', player_in_jxsi: false, player_in_jxcat: false,
    cat_coalition: [author], cat_coalition_support: [], cat_coalition_abstain: [],
    president_party: author, economy_minister_party: author,
    parlament_current_ciu: 'ciu', parlament_current_icv: 'icv',
    [author + '_parlament_s']: 70, parlament_size: 135, parlament_s_majority: 68,
    year: 2013, budget_year: 2012, generalitat_surplus: -2.3,
    tax_policy_level: 1, social_spending_cap: 0,
    education_spending: 0, healthcare_spending: -1, public_salary_level: 0,
    education_prepared_level: null, healthcare_prepared_level: null, public_salary_prepared_level: null,
    coalition_policy_commitments: [], active_mods: {}, mod_log: [],
    cat_caretaker_gov: false, party_resources: 10, government_budget: 2,
    ciu_roadmap: 0, erc_roadmap: 0, cup_roadmap: 0,
    jxsi_formation_num: 40, jxcat_leader_party: 'cdc',
    erc_in_jxsi: false, cup_in_jxsi: false, erc_in_jxcat: false, cup_in_jxcat: false,
    ...overrides,
  });
  return { adapter, q };
}

function propose(adapter: DendryAdapter) {
  const q = adapter.qualities as Record<string, any>;
  q.budget_parlament_trigger = true;
  const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
  try {
    adapter.goToScene('budget_trigger');
    expect(adapter.engine.state.sceneId).toBe('parlament_budget.review');
    expect(errors.mock.calls.filter(([message]) => !String(message).startsWith('Page loaded from unexpected origin'))).toEqual([]);
  } finally {
    errors.mockRestore();
  }
  return q;
}

describe('NPC budget authors and fiscal conversion', () => {
  it('supports the untouched CiU budget after ERC secures Education spending at formation', () => {
    const { adapter, q } = setup('ciu', {
      cat_coalition: ['ciu', 'erc'], ciu_parlament_s: 50, erc_parlament_s: 21,
      ppc_parlament_s: 64, education_spending: -1, countdowns: [],
      speaker_party: 'ciu', speaker: 'Nuria de Gispert', speaker_gender: 'f',
      parlament_coalition_proposal_president: 'ciu',
      parlament_coalition_proposal_president_name: 'Artur Mas',
      parlament_coalition_proposal_president_gender: 'm',
      parlament_coalition_policy_accepted: true, parlament_coalition_policy_scope: 'cabinet',
      parlament_coalition_policy_education: true, parlament_coalition_policy_education_target: 0,
      parlament_coalition_policy_cap: false, parlament_coalition_policy_tax: false,
      parlament_coalition_policy_health: false, parlament_coalition_policy_pay: false,
    });
    for (const ministry of ['vp', 'economy', 'interior', 'health', 'education', 'territory', 'foreign', 'bnl'])
      q['parlament_coalition_proposal_' + ministry] = ministry === 'education' ? 'erc' : 'ciu';
    adapter.goToScene('parlament_coalition_resolve');
    expect(q.parlament_coalition_resolution_valid).toBe(true);
    expect(q.education_spending).toBe(0);
    propose(adapter);
    expect(q.parlament_budget_changes_made).toBe(false);
    expect(q.parlament_budget_projection.brokenCabinet).toEqual([]);
    expect(q.parlament_budget_positions).toMatchObject({ ciu: 'yes', erc: 'yes' });
    expect(q.parlament_budget_can_pass).toBe(true);
    expect(q.parlament_budget_shortfall).toBeCloseTo(1.3);
    adapter.goToScene('parlament_budget.pass');
    expect(q.budget_year).toBe(2013);
    expect(q.education_spending).toBe(0);
  });

  it.each(['ciu', 'cdc', 'dl', 'pdcat', 'junts', 'unio', 'ppc', 'cs', 'vox',
    'erc', 'cup', 'psc', 'icv', 'csqp', 'cecp', 'ecp', 'si', 'pxc', 'fnc', 'jxsi', 'jxcat'])
  ('%s supports its own valid opening proposal', (party) => {
    const { adapter, q } = setup(party);
    propose(adapter);
    expect(q.budget_draft_origin.proposer_party).toBe(party);
    expect(q.parlament_budget_projection.valid).toBe(true);
    expect(q.parlament_budget_positions[party]).toBe('yes');
    expect(q.parlament_budget_can_pass).toBe(true);
  });

  it.each(['ciu', 'cdc', 'dl', 'pdcat', 'junts', 'unio', 'ppc', 'cs', 'vox',
    'erc', 'cup', 'psc', 'icv', 'csqp', 'cecp', 'ecp', 'si', 'pxc', 'fnc', 'jxsi', 'jxcat'])
  ('%s can oppose an amendment that increases its submitted financing gap', (party) => {
    const { adapter, q } = setup(party);
    propose(adapter);
    const originalGap = q.parlament_budget_shortfall;
    adapter.goToScene('parlament_budget.tax_lower');
    expect(q.parlament_budget_shortfall).toBeCloseTo(originalGap + 1);
    expect(q.parlament_budget_positions[party]).toBe('no');
  });

  it.each([[-5, 4], [-2.3, 1.3], [-1, 0], [0, 0], [0.5, 0]])
  ('uses one funding point per surplus percentage point at %s', (surplus, gap) => {
    const { adapter, q } = setup('ciu', { generalitat_surplus: surplus });
    propose(adapter);
    expect(q.parlament_budget_margin).toBeCloseTo(1 + surplus);
    expect(q.parlament_budget_shortfall).toBeCloseTo(gap);
  });

  it('keeps a separate CiU cabinet partner opposed to the gap in an ERC proposal', () => {
    const { adapter, q } = setup('erc', {
      cat_coalition: ['erc', 'ciu'], ciu_parlament_s: 30, erc_parlament_s: 70,
    });
    propose(adapter);
    expect(q.parlament_budget_positions).toMatchObject({ erc: 'yes', ciu: 'no' });
    expect(q.parlament_budget_cabinet_yes).toBe(false);
    expect(q.parlament_budget_can_pass).toBe(false);
  });

  it('records authorship at submission even if Economy changes later', () => {
    const { adapter, q } = setup('ciu', { erc_parlament_s: 21 });
    propose(adapter);
    q.economy_minister_party = 'erc';
    adapter.goToScene('parlament_budget.review');
    expect(q.budget_draft_origin.proposer_party).toBe('ciu');
    expect(q.parlament_budget_positions.ciu).toBe('yes');
  });

  it('rejects a larger amended gap and supports the original proposal again when restored', () => {
    const { adapter, q } = setup('ciu');
    propose(adapter);
    adapter.goToScene('parlament_budget.tax_lower');
    expect(q.parlament_budget_shortfall).toBeCloseTo(2.3);
    expect(q.parlament_budget_positions.ciu).toBe('no');
    adapter.goToScene('parlament_budget.tax_higher');
    expect(q.parlament_budget_shortfall).toBeCloseTo(1.3);
    expect(q.parlament_budget_positions.ciu).toBe('yes');
    expect(q.budget_draft_origin.tax_policy_level).toBe(1);
  });

  it('keeps the author opposed to an amended Very high tax package', () => {
    const { adapter, q } = setup('ciu');
    propose(adapter);
    adapter.goToScene('parlament_budget.tax_higher');
    expect(q.parlament_budget_shortfall).toBeCloseTo(0.3);
    expect(q.parlament_budget_positions.ciu).toBe('no');
  });

  it('allows removing a project while retaining the originally accepted deficit financing', () => {
    const { adapter, q } = setup('ciu', {
      active_mods: { territory_county_roads: { active: true, def: { colour: 'gray', targets: {} } } },
    });
    propose(adapter);
    expect(q.budget_draft.projects.county).toBe(true);
    expect(q.parlament_budget_shortfall).toBeCloseTo(2.3);
    adapter.goToScene('parlament_budget.county');
    expect(q.parlament_budget_shortfall).toBeCloseTo(1.3);
    expect(q.parlament_budget_positions.ciu).toBe('yes');
  });

  it('does not give a player-authored proposal the NPC financing exception', () => {
    const { adapter, q } = setup('ciu', {
      player_party: 'erc', economy_minister_party: 'erc', cat_coalition: ['ciu', 'erc'], erc_parlament_s: 21,
    });
    adapter.goToScene('generalitat_economy_card.prepare_budget');
    adapter.goToScene('generalitat_economy_card.submit_budget');
    expect(q.budget_draft_origin.author).toBe('player');
    expect(q.parlament_budget_positions.ciu).toBe('no');
  });

  it('does not let sponsorship override a breached cabinet agreement', () => {
    const { adapter, q } = setup('ciu');
    propose(adapter);
    q.coalition_policy_commitments = [{ kind: 'education_spending', target: 1,
      scope: 'cabinet', signatories: ['ciu'], status: 'applied' }];
    adapter.goToScene('parlament_budget.review');
    expect(q.parlament_budget_projection.brokenCabinet).toHaveLength(1);
    expect(q.parlament_budget_positions.ciu).toBe('no');
    expect(q.parlament_budget_can_pass).toBe(false);
  });

  it('does not let sponsorship override an invalid draft', () => {
    const { adapter, q } = setup('ciu');
    propose(adapter);
    q.budget_draft.projects.county = true;
    adapter.goToScene('parlament_budget.review');
    expect(q.parlament_budget_projection.valid).toBe(false);
    expect(q.parlament_budget_positions.ciu).toBe('no');
  });

  it('honours an explicit player vote on a unity-list ballot', () => {
    const { adapter, q } = setup('jxsi', {
      player_party: 'erc', player_in_jxsi: true, erc_in_jxsi: true,
    });
    propose(adapter);
    adapter.goToScene('parlament_budget.vote_no');
    expect(q.parlament_budget_positions.jxsi).toBe('no');
    expect(q.parlament_budget_yes + q.parlament_budget_no + q.parlament_budget_abstain).toBe(70);
  });
});
