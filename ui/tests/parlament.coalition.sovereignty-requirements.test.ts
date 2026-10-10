import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const directory = resolve(import.meta.dirname, '../../source/scenes/events/elections');
const files = readdirSync(directory).filter(name => /^parlament_coalition_(joint_cup|ciu_erc_cup)_phase_/.test(name));
function run(file: string, q: Record<string, any>, marker = '') {
  if (!marker && file.includes('_phase_')) marker = 'prepare';
  let source = readFileSync(resolve(directory, file), 'utf8');
  if (marker) {
    const start = source.search(new RegExp(`^@${marker}\\r?$`, 'm'));
    expect(start).toBeGreaterThanOrEqual(0);
    source = source.slice(start);
  }
  const block = source.match(/on-arrival:\s*\{!([\s\S]*?)!\}/)![1];
  new Function('Q', 'G', block)(q, { random: () => 0.8 });
}
function fixture(file: string): Record<string, any> {
  const joint = file.includes('joint_cup');
  const leader = joint ? (file.includes('jxsi') ? 'jxsi' : 'jxcat') : file.includes('_erc_led') ? 'erc' : 'ciu';
  const q: Record<string, any> = {
    player_party: file.includes('_erc_player_') ? 'erc' : 'cup',
    parlament_current_ciu: 'ciu', parlament_coalition_leader: leader,
    parlament_coalition_follower: leader === 'erc' ? 'ciu' : 'erc',
    parlament_coalition_sovereignty_kind: joint ? leader : file.includes('_government') ? 'government' : 'support',
    parlament_coalition_sovereignty_cabinet: joint ? [leader] : ['ciu', 'erc', 'cup'],
    parlament_coalition_sovereignty_phase: 3, parlament_coalition_sovereignty_mas_pressure: 0,
    parlament_coalition_support_seats: 80, erc_parlament_s: 35, cup_parlament_s: 10,
    parlament_coalition_proposal_president_name: 'Carles Puigdemont',
    parlament_coalition_leverage: 40, coalition_phase_timer: 0,
    parlament_coalition_concessions_indy: 1, parlament_coalition_policy_health: true,
    parlament_coalition_sovereignty_deadline: 2015, speaker_party: 'cup',
  };
  for (const office of ['vp', 'interior', 'economy', 'foreign', 'health', 'education', 'territory', 'bnl']) {
    q[`parlament_coalition_proposal_${office}`] = joint ? leader : 'erc';
  }
  if (!joint) {
    q.parlament_coalition_proposal_economy = 'ciu';
    q.parlament_coalition_proposal_health = 'cup';
    q.parlament_coalition_proposal_education = 'cup';
  }
  return q;
}

describe('sovereignty pact requirements', () => {
  const governments = files.filter(file => file.includes('ciu_erc_cup') && file.includes('_government'));
  const cabinetEditors = files.filter(file => file.includes('ciu_erc_cup') && (file.includes('_government') || file.includes('_erc_player_')));
  it.each(cabinetEditors)('offers only player claims in the compiled cabinet menu: %s', file => {
    const a = DendryAdapter.fromJSONText(readFileSync(resolve(directory, '../../../../out/game.json'), 'utf8'));
    a.beginGame([1, 2, 3, 4]); a.goToScene('root.start_menu'); a.goToScene('root.start');
    const q = fixture(file);
    Object.assign(a.qualities, q, { parlament_current_ciu_s: 35 });
    const scene = file.replace('.scene.dry', '');
    let frame = a.goToScene(scene + '.ministries');
    expect(frame.choices.filter(c => /\.(vp|interior|economy|foreign|health|education|territory|bnl)$/.test(c.id))).toHaveLength(8);
    frame = a.choose(frame.choices.findIndex(c => c.id === scene + '.vp'));
    expect(frame.choices.some(c => /_(ciu|erc|cup)$/.test(c.id))).toBe(false);
    a.choose(frame.choices.findIndex(c => c.id === scene + '.redistribute'));
    for (const office of ['vp', 'interior', 'economy', 'foreign', 'health', 'education', 'territory', 'bnl']) {
      expect(['ciu', 'erc', ...(file.includes('_government') ? ['cup'] : [])]).toContain(a.qualities[`parlament_coalition_proposal_${office}`]);
    }
    expect(readFileSync(resolve(directory, file), 'utf8')).not.toMatch(/title: (Assign|Allocate) the/);
  });

  it.each(governments.filter(file => file.includes('_erc_player_')))('allocates NPC posts by preferences and changes their share with leverage: %s', file => {
    const q = fixture(file);
    Object.assign(q, { parlament_current_ciu_s: 20, cup_parlament_s: 30 });
    for (const office of ['vp', 'interior', 'economy', 'foreign', 'health', 'education', 'territory', 'bnl']) q[`parlament_coalition_proposal_${office}`] = 'erc';
    for (const office of ['economy', 'health', 'bnl']) q[`parlament_coalition_proposal_${office}`] = null;
    run(file, q, 'redistribute');
    expect(q.parlament_coalition_proposal_economy).toBe('ciu');
    expect(q.parlament_coalition_proposal_bnl).toBe('cup');
    expect(q.parlament_coalition_proposal_health).toBe('cup');
    // Reallocate existing NPC holdings, not just null offices.
    q.parlament_current_ciu_s = 60;
    q.cup_parlament_s = 5;
    run(file, q, 'redistribute');
    expect(q.parlament_coalition_proposal_health).toBe('ciu');
    expect(q.parlament_coalition_proposal_foreign).toBe('erc');
  });

  it.each(governments)('keeps a complete opening and redistributes within partner red lines in every round: %s', file => {
    for (const round of [1, 2, 3]) {
      const q = fixture(file);
      Object.assign(q, { parlament_coalition_sovereignty_phase: round, parlament_current_ciu_s: 35 });
      run(file, q, 'prepare');
      for (const office of ['vp', 'interior', 'economy', 'foreign', 'health', 'education', 'territory', 'bnl']) {
        expect(['ciu', 'erc', 'cup']).toContain(q[`parlament_coalition_proposal_${office}`]);
        q[`parlament_coalition_proposal_${office}`] = null;
      }
      run(file, q, 'redistribute');
      expect(q.parlament_coalition_proposal_economy).not.toBe('cup');
      if (round === 1) expect(q.parlament_coalition_proposal_interior).toBe('ciu');
      if (round < 3) expect(q.parlament_coalition_proposal_interior).not.toBe('cup');
      const snapshot = JSON.stringify(q);
      run(file, q, 'redistribute');
      expect(JSON.stringify(q)).toBe(snapshot);
    }
  });

  it.each(governments)('negotiates only the player posts and redistributes released offices: %s', file => {
    const q = fixture(file);
    q.parlament_current_ciu_s = 35;
    run(file, q, 'prepare');
    const president = q.parlament_coalition_proposal_president_name;
    for (const office of ['vp', 'interior', 'economy', 'foreign', 'health', 'education', 'territory', 'bnl']) {
      const key = `parlament_coalition_proposal_${office}`;
      if (q[key] !== q.player_party) run(file, q, office);
      const leverage = q.parlament_coalition_leverage;
      run(file, q, office);
      expect(q[key]).toBeNull();
      run(file, q, office);
      expect(q[key]).toBe(q.player_party);
      expect(q.parlament_coalition_leverage).toBe(leverage);
      run(file, q, office);
      run(file, q, 'redistribute');
      expect(['ciu', 'erc', 'cup'].filter(party => party !== q.player_party)).toContain(q[key]);
    }
    expect(q.parlament_coalition_proposal_president_name).toBe(president);
    const source = readFileSync(resolve(directory, file), 'utf8');
    expect(source).not.toMatch(/title: Assign the/);
  });

  it.each(governments)('requires each NPC partner to accept independently: %s', file => {
    for (const partner of ['ciu', file.includes('_erc_player_') ? 'cup' : 'erc']) {
      const q = fixture(file);
      run(file, q, 'submit');
      expect(q.parlament_coalition_proposal_accept).toBe(true);
      for (const office of ['interior', 'economy', 'foreign', 'health', 'education', 'territory', 'bnl']) {
        if (q[`parlament_coalition_proposal_${office}`] === partner) q[`parlament_coalition_proposal_${office}`] = q.player_party;
      }
      run(file, q, 'submit');
      expect(q.parlament_coalition_proposal_accept).toBe(false);
      expect(q.parlament_coalition_sovereignty_reject_reasons.toLowerCase()).toMatch(partner === 'ciu' ? /(ciu expects|convergents expect)/ : new RegExp(partner + ' expects'));
    }
  });

  it.each(files)('preserves previous rejection feedback until submission: %s', file => {
    const q = fixture(file);
    q.parlament_coalition_sovereignty_reject_reasons = 'The partners rejected our proposed commitments.';
    run(file, q, 'prepare');
    expect(q.parlament_coalition_sovereignty_reject_reasons).toBe('The partners rejected our proposed commitments.');
    run(file, q, 'submit');
    expect(q.parlament_coalition_sovereignty_reject_reasons).not.toContain('The partners rejected our proposed commitments.');
  });

  it.each(files)('records candidate objections through the full submission: %s', file => {
    const q = fixture(file);
    q.parlament_coalition_sovereignty_reject_reasons = 'The partners rejected our proposed commitments.';
    q.parlament_coalition_sovereignty_candidate_reason = 'The assembly has vetoed Artur Mas.';
    run(file, q, 'submit');
    run('parlament_coalition_sovereignty_replacement.scene.dry', q, 'evaluate');
    expect(q.parlament_coalition_sovereignty_reject_reasons).toContain(q.parlament_coalition_sovereignty_candidate_reason);
    expect(q.parlament_coalition_proposal_accept).toBe(false);
  });

  it.each(files)('aligns received-offer acceptance, full submission and submenu pushes in the engine: %s', file => {
    const a = DendryAdapter.fromJSONText(readFileSync(resolve(directory, '../../../../out/game.json'), 'utf8'));
    a.beginGame([1, 2, 3, 4]); a.goToScene('root.start_menu'); a.goToScene('root.start');
    Object.assign(a.qualities, fixture(file), {
      parlament_coalition_sovereignty_phase: 1, parlament_coalition_push: 0,
      parlament_coalition_changes: false, parlament_coalition_policy_health: false,
      parlament_coalition_policy_indy_demanded: false, rubicon: false,
    });
    const scene = file.replace('.scene.dry', '');
    let frame = a.goToScene(scene + '.choices');
    const author = file.includes('_erc_player_') && (file.includes('joint_cup') || file.includes('_erc_led_'));
    expect(frame.choices.find(c => c.id === scene + '.accept')?.canChoose).toBe(author ? undefined : true);
    expect(frame.choices.some(c => /\.(reject|push|push_policies|push_ministries|candidate_impasse)$/.test(c.id))).toBe(false);
    frame = a.goToScene(scene + '.policies');
    const before = Number(a.qualities.parlament_coalition_leverage);
    frame = a.choose(frame.choices.findIndex(c => c.id === scene + '.push_policies'));
    expect(a.qualities.parlament_coalition_leverage).toBe(before + 4);
    expect(a.qualities.parlament_coalition_push).toBe(1);
    expect(frame.choices.find(c => c.id === scene + '.push_policies')?.canChoose).toBe(false);
    if (cabinetEditors.includes(file)) {
      frame = a.goToScene(scene + '.ministries');
      expect(frame.choices.find(c => c.id === scene + '.push_ministries')?.canChoose).toBe(false);
    }
    a.qualities.parlament_coalition_sovereignty_phase = 2;
    frame = a.goToScene(scene + '.choices');
    expect(frame.choices.find(c => c.id === scene + '.accept')?.canChoose).toBe(false);
    Object.assign(a.qualities, {
      parlament_coalition_changes: false, parlament_coalition_proposal_president_name: 'Artur Mas',
      parlament_coalition_sovereignty_assembly_result: 'veto',
    });
    frame = a.goToScene(scene + '.choices');
    expect(frame.choices.find(c => c.id === scene + '.accept')?.canChoose).toBe(false);
    expect(frame.choices.some(c => c.id === scene + '.candidate_impasse')).toBe(false);
    a.choose(frame.choices.findIndex(c => c.id === scene + '.submit'));
    expect(a.qualities.parlament_coalition_sovereignty_phase).toBe(3);
    expect(a.qualities.coalition_phase_timer).toBe(1);
    expect(a.qualities.parlament_coalition_sovereignty_reject_reasons).toContain('Artur Mas');
    a.qualities.parlament_coalition_sovereignty_phase = 3;
    a.qualities.coalition_phase_timer = 0;
    frame = a.goToScene(scene + '.choices');
    expect(frame.choices.find(c => c.id === scene + '.submit')?.canChoose).toBe(true);
    frame = a.goToScene(scene + '.policies');
    expect(frame.choices.find(c => c.id === scene + '.push_policies')?.canChoose).toBe(false);
  });

  it.each(files.filter(file => file.includes('_cup_player_')))('accepts an unchanged eligible pact through the normal checks: %s', file => {
    const a = DendryAdapter.fromJSONText(readFileSync(resolve(directory, '../../../../out/game.json'), 'utf8'));
    a.beginGame([1, 2, 3, 4]); a.goToScene('root.start_menu'); a.goToScene('root.start');
    const q = fixture(file);
    Object.assign(q, {
      parlament_coalition_changes: false, parlament_coalition_policy_health: false,
      parlament_coalition_policy_indy_demanded: false, parlament_coalition_cabinet_seats: 80,
    });
    if (file.includes('_support')) {
      q.parlament_coalition_sovereignty_cabinet = ['ciu', 'erc'];
      q.parlament_coalition_proposal_health = 'erc';
      q.parlament_coalition_proposal_education = 'erc';
    }
    Object.assign(a.qualities, q);
    const scene = file.replace('.scene.dry', '');
    const frame = a.goToScene(scene + '.choices');
    expect(frame.choices.find(c => c.id === scene + '.accept')?.canChoose).toBe(true);
    a.choose(frame.choices.findIndex(c => c.id === scene + '.accept'));
    expect(a.qualities.parlament_coalition_proposal_accept).toBe(true);
    expect(a.qualities.parlament_coalition_sovereignty_phase).toBe(0);
    expect(a.qualities.cat_coalition).toEqual(q.parlament_coalition_sovereignty_cabinet);
  });

  it('resolves both dispute types independently without repeating either', () => {
    const q = fixture(files[0]);
    Object.assign(q, {
      coalition_phase_timer: 1, parlament_coalition_sovereignty_phase: 2,
      parlament_coalition_sovereignty_dispute_round: 0,
      parlament_coalition_sovereignty_mediation_round: 0,
      parlament_coalition_sovereignty_feud: 1,
      cup_endavant_strength: 30, cup_poblelliure_strength: 40, cup_thirdsector_strength: 30,
      cup_endavant_dissent: 0, cup_poblelliure_dissent: 0, cup_thirdsector_dissent: 0,
      parlament_coalition_sovereignty_dispute_action: 'defend',
    });
    q.parlament_coalition_proposal_president_name = 'Artur Mas';
    const file = 'parlament_coalition_sovereignty_dispute.scene.dry';
    run('parlament_coalition_sovereignty_candidate_erc.scene.dry', q, 'defend');
    expect(q.parlament_coalition_sovereignty_dispute_round).toBe(2);
    expect(q.parlament_coalition_sovereignty_mediation_round).toBe(0);
    q.parlament_coalition_sovereignty_dispute_action = 'mediate';
    run(file, q, 'resolve');
    expect(q.parlament_coalition_sovereignty_mediation_round).toBe(2);
    const snapshot = JSON.stringify(q);
    run('parlament_coalition_sovereignty_candidate_erc.scene.dry', q, 'defend');
    expect(JSON.stringify(q)).toBe(snapshot);
    q.parlament_coalition_sovereignty_dispute_action = 'defend';
    const repeated = JSON.stringify(q);
    run(file, q, 'resolve');
    expect(JSON.stringify(q)).toBe(repeated);
  });

  it('resets negotiation state through the shared initializer in all twelve openings', () => {
    for (const file of ['parlament_coalition_joint_cup.scene.dry', 'parlament_coalition_ciu_erc_cup.scene.dry']) {
      const source = readFileSync(resolve(directory, file), 'utf8');
      const entries = [...source.matchAll(/^@(\w+_player_\w+)\r?$/gm)].map(match => match[1]);
      expect(entries.length).toBe(file.includes('joint') ? 4 : 8);
      for (const entry of entries) {
        const q = JSON.parse(JSON.stringify({
          ...fixture(files[0]), parlament_coalition_sovereignty_assembly_result: 'veto',
          parlament_coalition_sovereignty_banos_departure: true,
          parlament_coalition_sovereignty_mediation_round: 3,
          parlament_current_ciu: 'ciu', parlament_current_ciu_s: 35,
          ciu_parlament_s: 35, jxsi_parlament_s: 60, jxcat_parlament_s: 60,
          cup_endavant_strength: 30, cup_poblelliure_strength: 40, cup_thirdsector_strength: 30,
        }));
        run(file, q, entry);
        const node = source.slice(source.indexOf('@' + entry)).split(/\r?\n@/)[0];
        const called = node.match(/^call: (\w+)\.(\w+)/m)!;
        expect(called, entry).toBeTruthy();
        run(called[1] + '.scene.dry', q, called[2]);
        expect(q.parlament_coalition_sovereignty_assembly_result, entry).toBe('');
        expect(q.parlament_coalition_sovereignty_banos_departure, entry).toBe(false);
        expect(q.parlament_coalition_sovereignty_mediation_round).toBe(0);
      }
    }
  });

  it.each(files.filter(file => file.includes('_erc_player_')))('accepts a social agreement without purchasing a stronger roadmap: %s', file => {
    const q = fixture(file);
    q.parlament_coalition_concessions_indy = 0;
    run(file, q, 'submit');
    expect(q.parlament_coalition_proposal_accept).toBe(true);
    expect(q.parlament_coalition_sovereignty_reject_reasons).toBe('');
    q.parlament_coalition_concessions_indy = 1;
    run(file, q, 'submit');
    expect(q.parlament_coalition_sovereignty_reject_reasons).toBe('');
    expect(q.parlament_coalition_proposal_accept).toBe(true);
  });

  it.each(files.filter(file => file.includes('_erc_player_') && (file.includes('joint_cup') || file.includes('_support'))))('still requires social concessions when the roadmap upgrade is omitted: %s', file => {
    const q = fixture(file);
    q.parlament_coalition_concessions_indy = 0;
    q.parlament_coalition_sovereignty_deadline = 0;
    q.parlament_coalition_policy_health = false;
    run(file, q, 'submit');
    expect(q.parlament_coalition_proposal_accept).toBe(false);
    expect(q.parlament_coalition_sovereignty_reject_reasons).toContain('social');
  });

  it.each(files.filter(file => file.includes('_cup_player_')))('lets CUP choose its demands and attributes excessive concessions to the partner: %s', file => {
    const q = fixture(file);
    q.parlament_coalition_concessions_indy = 0;
    q.parlament_coalition_sovereignty_deadline = 0;
    q.parlament_coalition_policy_health = false;
    run(file, q, 'submit');
    expect(q.parlament_coalition_proposal_accept).toBe(true);
    q.parlament_coalition_leverage = -1;
    run(file, q, 'submit');
    expect(q.parlament_coalition_proposal_accept).toBe(false);
    const names: Record<string, string> = { jxsi: 'JxSí', jxcat: 'JxCat', erc: 'ERC', ciu: 'CiU' };
    const partner = names[q.parlament_coalition_leader];
    expect(q.parlament_coalition_sovereignty_reject_reasons).toContain(partner);
    expect(q.parlament_coalition_sovereignty_reject_reasons).toContain('concessions');
  });

  it.each(['spending_cap', 'education_policy', 'health_policy', 'pay_policy', 'regional_tax'])('allows an excessive %s demand only during CUP sovereignty talks and refunds its withdrawal', marker => {
    for (const [player, phase, allowed] of [['cup', 1, true], ['erc', 1, false], ['cup', 0, false]] as const) {
      const q = fixture(files[0]);
      Object.assign(q, { player_party: player, parlament_coalition_sovereignty_phase: phase,
        parlament_coalition_leverage: 0, generalitat_surplus: 0,
        social_spending_cap: 0, education_spending: 0, healthcare_spending: 0, public_salary_level: 0, tax_policy_level: 0 });
      const file = 'parlament_coalition_fiscal_policies.scene.dry';
      run(file, q, 'initialize');
      run(file, q, marker);
      expect(q.parlament_coalition_leverage).toBe(allowed ? -5 : 0);
      if (allowed) { run(file, q, marker); expect(q.parlament_coalition_leverage).toBe(0); }
    }
  });

  it.each(files)('charges the Speaker once in the total offer allowance: %s', file => {
    const costs: Record<string, number> = {vp: 20, interior: 10, economy: 10, foreign: 6, health: 7, education: 7, territory: 5, bnl: 5};
    function total(q: Record<string, any>) {
      let value = q.parlament_coalition_leverage + (q.parlament_coalition_concessions_indy ? 5 : 0) + (q.parlament_coalition_sovereignty_deadline ? 5 : 0);
      for (const p of ['health', 'education', 'pay', 'tax', 'cap']) if (q['parlament_coalition_policy_' + p]) value += p === 'tax' ? 5 : q.parlament_coalition_policy_spending_cost;
      if (!file.includes('joint_cup') && (q.player_party === 'erc' || file.includes('_government'))) for (const [o, cost] of Object.entries(costs)) if (q['parlament_coalition_proposal_' + o] === q.player_party) value += cost;
      return value;
    }
    for (const phase of [1, 2, 3]) {
      const original = {...fixture(file), parlament_coalition_sovereignty_phase: phase, generalitat_surplus: 0};
      const q = {...original, speaker_party: 'none'};
      run('parlament_coalition_sovereignty_offer.scene.dry', q, 'prepare');
      const baseline = total(q);
      const speaker = {...original, speaker_party: file.includes('joint_cup') && original.player_party === 'erc' ? original.parlament_coalition_leader : original.player_party};
      run('parlament_coalition_sovereignty_offer.scene.dry', speaker, 'prepare');
      const ordinaryCiuErc = file === 'parlament_coalition_ciu_erc_cup_phase_erc_player_ciu_led_support.scene.dry';
      // The ordinary CiU-ERC formula already deducts the speakership when
      // CiU does not hold it; assigning it to ERC must not deduct it twice.
      expect(total(speaker)).toBeCloseTo(baseline - (ordinaryCiuErc ? 0 : 5));
      if (ordinaryCiuErc) {
        const ciuSpeaker = {...original, speaker_party: original.parlament_current_ciu};
        run('parlament_coalition_sovereignty_offer.scene.dry', ciuSpeaker, 'prepare');
        expect(total(ciuSpeaker) - total(speaker)).toBeCloseTo(original.parlament_current_ciu_s > 40 ? 13 : 15);
      }
      const snapshot = JSON.stringify(speaker);
      run('parlament_coalition_sovereignty_offer.scene.dry', speaker, 'prepare');
      expect(JSON.stringify(speaker)).toBe(snapshot);
    }
  });
});
