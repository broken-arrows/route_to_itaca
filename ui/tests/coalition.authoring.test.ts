import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { convertLine } from 'dendrynexus-ten/lib/ui/content/html.js';
import { DendryAdapter } from '../src/engine/adapter';

const root = resolve(import.meta.dirname, '..', '..');
const gameText = readFileSync(resolve(root, 'out/game.json'), 'utf8');

function renderTitle(sceneId: string, qualities: Record<string, unknown>) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  Object.assign(adapter.qualities, qualities);
  const scene = adapter.engine.game.scenes[sceneId];
  return convertLine(adapter.engine._makeDisplayContent(scene.title, true));
}

describe('authored coalition tooltip metadata', () => {
  it('compiles all four ERC-CUP player/leader phase combinations', () => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    const ids = [
      'parlament_coalition_erc_cup_phase_erc_player_erc_led',
      'parlament_coalition_erc_cup_phase_erc_player_cup_led',
      'parlament_coalition_erc_cup_phase_cup_player_erc_led',
      'parlament_coalition_erc_cup_phase_cup_player_cup_led',
    ];

    for (const id of ids) {
      expect(adapter.engine.game.scenes[id], id).toBeTruthy();
    }
  });

  it('exposes the four ERC-CUP combinations explicitly from the coalition menu', () => {
    const source = readFileSync(
      resolve(root, 'source/scenes/events/elections/parlament_coalition.scene.dry'),
      'utf8',
    );
    const options = [
      '@parlament_coalition_erc_cup.erc_player_erc_led',
      '@parlament_coalition_erc_cup.erc_player_cup_led',
      '@parlament_coalition_erc_cup.cup_player_erc_led',
      '@parlament_coalition_erc_cup.cup_player_cup_led',
    ];

    for (const option of options) expect(source).toContain(`- ${option}`);
    expect(source).not.toContain('- @parlament_coalition_erc_cup.government');
  });

  it.each([
    ['erc', 'erc', 'cup', 'parlament_coalition_erc_cup.erc_player_erc_led', 'parlament_coalition_erc_cup_phase_erc_player_erc_led'],
    ['erc', 'cup', 'erc', 'parlament_coalition_erc_cup.erc_player_cup_led', 'parlament_coalition_erc_cup_phase_erc_player_cup_led'],
    ['cup', 'erc', 'cup', 'parlament_coalition_erc_cup.cup_player_erc_led', 'parlament_coalition_erc_cup_phase_cup_player_erc_led'],
    ['cup', 'cup', 'erc', 'parlament_coalition_erc_cup.cup_player_cup_led', 'parlament_coalition_erc_cup_phase_cup_player_cup_led'],
  ])('routes the explicit %s-player/%s-led menu entry to its own phase tree', (player, leader, follower, entryId, phaseId) => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, {
      player_party: player,
      year: 2018,
      erc_parlament_s: leader === 'erc' ? 40 : 32,
      cup_parlament_s: leader === 'cup' ? 40 : 32,
      parlament_coalition_erc_cup_s: 72,
      parlament_s_majority: 68,
      parlament_erc_cup_government_available: true,
    });

    const speakerFrame = adapter.goToScene(entryId);
    const phaseFrame = adapter.choose(speakerFrame.choices.findIndex((choice) => choice.canChoose));

    expect(adapter.qualities.parlament_coalition_leader).toBe(leader);
    expect(adapter.qualities.parlament_coalition_follower).toBe(follower);
    expect(phaseFrame.sceneId).toBe(`${phaseId}.choices`);
  });

  it.each([
    ['erc', 'erc', 'cup', true, 'parlament_coalition_erc_cup_phase_erc_player_erc_led'],
    ['erc', 'cup', 'erc', false, 'parlament_coalition_erc_cup_phase_erc_player_cup_led'],
    ['cup', 'erc', 'cup', false, 'parlament_coalition_erc_cup_phase_cup_player_erc_led'],
    ['cup', 'cup', 'erc', true, 'parlament_coalition_erc_cup_phase_cup_player_cup_led'],
  ])('initializes the %s-player/%s-led phase according to the negotiation role', (player, leader, follower, playerLeads, sceneId) => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, {
      player_party: player,
      parlament_coalition_leader: leader,
      parlament_coalition_follower: follower,
      erc_cup_coalition_phase: 1,
      erc_parlament_s: leader === 'erc' ? 40 : 32,
      cup_parlament_s: leader === 'cup' ? 40 : 32,
      parlament_coalition_erc_cup_s: 72,
      parlament_s_majority: 68,
      speaker_party: follower,
    });

    const frame = adapter.goToScene(sceneId);
    const choiceIds = frame.choices.map((choice) => choice.id);

    expect(adapter.qualities.parlament_coalition_proposal_president).toBe(leader);
    if (playerLeads) {
      expect(adapter.qualities.parlament_coalition_proposal_vp).toBeNull();
      expect(adapter.qualities.parlament_coalition_proposal_interior).toBeNull();
      expect(choiceIds).not.toContain(`${sceneId}.accept`);
      expect(choiceIds).not.toContain(`${sceneId}.send`);
    } else {
      expect(adapter.qualities.parlament_coalition_proposal_vp).toBe(player);
      expect([leader, player]).toContain(adapter.qualities.parlament_coalition_proposal_interior);
      expect(choiceIds).toContain(`${sceneId}.accept`);
    }
    expect(adapter.qualities.parlament_coalition_leverage).toBeGreaterThan(0);
  });

  it.each([
    ['erc', 'cup', 'parlament_coalition_erc_cup_phase_erc_player_erc_led'],
    ['cup', 'erc', 'parlament_coalition_erc_cup_phase_cup_player_cup_led'],
  ])('has the %s leader receive the %s counter-offer in round two', (player, follower, sceneId) => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, {
      player_party: player,
      parlament_coalition_leader: player,
      parlament_coalition_follower: follower,
      erc_cup_coalition_phase: 2,
      erc_parlament_s: player === 'erc' ? 40 : 32,
      cup_parlament_s: player === 'cup' ? 40 : 32,
      parlament_coalition_erc_cup_s: 72,
      parlament_s_majority: 68,
      speaker_party: follower,
    });

    const frame = adapter.goToScene(sceneId);

    expect(adapter.qualities.parlament_coalition_proposal_president).toBe(player);
    expect(adapter.qualities.parlament_coalition_proposal_vp).toBe(follower);
    expect(frame.choices.map((choice) => choice.id)).toContain(`${sceneId}.accept`);
  });

  it('persists ERC and CUP as cabinet partners when their phase succeeds', () => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, {
      parlament_coalition_leader: 'cup',
      parlament_coalition_follower: 'erc',
      parlament_coalition_erc_cup_s: 72,
      parlament_s_majority: 68,
      speaker: 'Roger Torrent',
      speaker_party: 'erc',
      erc_cup_coalition_phase: 2,
    });

    adapter.goToScene('parlament_coalition_erc_cup_phase_cup_player_cup_led.accepted');

    expect(adapter.qualities.cat_coalition).toEqual(['cup', 'erc']);
    expect(adapter.qualities.cat_coalition_support).toEqual([]);
    expect(adapter.qualities.cat_coalition_abstain).toEqual([]);
    expect(adapter.qualities.erc_in_gob).toBe(true);
    expect(adapter.qualities.cup_in_gob).toBe(true);
  });

  it('keeps CUP outside an ERC minority cabinet that CUP supports', () => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, {
      player_party: 'cup',
      parlament_coalition_erc_cup_s: 70,
      parlament_s_majority: 68,
      parlament_coalition_concessions_social: 1,
      parlament_coalition_concessions_indy: 1,
      speaker_party: 'cup',
      cup_support_coalition_phase: 1,
    });

    adapter.goToScene('parlament_coalition_erc_cup_support.form');

    expect(adapter.qualities.cat_coalition).toEqual(['erc']);
    expect(adapter.qualities.cat_coalition_support).toEqual(['cup']);
    expect(adapter.qualities.erc_in_gob).toBe(true);
    expect(adapter.qualities.cup_in_gob).toBe(false);
    expect(adapter.qualities.cup_supporting).toBe(true);
  });

  it.each([
    ['CiU-family-led', 'parlament_coalition_ciu_erc.cup', 'jxcat', 35, 30],
    ['ERC-led', 'parlament_coalition_erc_ciu.cup', 'jxcat', 30, 35],
  ])('keeps PDeCAT outside the cabinet in the CUP-view %s route', (_label, sceneId, ciuParty, ciuSeats, ercSeats) => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, {
      player_party: 'cup',
      year: 2018,
      parlament_current_ciu: ciuParty,
      parlament_current_ciu_s: ciuSeats,
      jxcat_parlament_s: ciuSeats,
      jxcat_leader: 'Carles Puigdemont',
      erc_parlament_s: ercSeats,
      erc_leader: 'Oriol Junqueras',
      parlament_coalition_ciu_erc_s: ciuSeats + ercSeats,
      parlament_coalition_erc_ciu_s: ciuSeats + ercSeats,
      parlament_coalition_pdcat_support_s: ciuSeats + ercSeats + 5,
      parlament_s_majority: 68,
      pdcat_external_support: true,
    });

    const frame = adapter.goToScene(sceneId);

    expect(adapter.qualities.cat_coalition).toEqual(sceneId.includes('ciu_erc') ? ['jxcat', 'erc'] : ['erc', 'jxcat']);
    expect(adapter.qualities.cat_coalition_support).toEqual(['pdcat']);
    expect(adapter.qualities.cat_coalition).not.toContain('pdcat');
    expect(adapter.qualities.parlament_coalition_pdcat_support_resolved).toBe(true);
    expect(adapter.qualities.pdcat_external_support).toBe(false);
    expect(frame.html).not.toContain('Thanks to our mediation efforts');
    expect(frame.html).not.toContain('Thanks to our efforts');
    expect(frame.html).toContain('keep the CUP out of the negotiations');
  });

  it('resolves Dendry arithmetic and party ids inside option-title attributes', () => {
    const html = renderTitle('congreso_coalition_right.pp_cs', {
      pp_congreso_s: 120,
      cs_congreso_s: 30,
      congreso_s_majority: 176,
    });
    expect(html).toContain('class="gobierno-coalition"');
    expect(html).toContain('data-parties="pp cs"');
    expect(html).toContain('data-summary="Minority government - 150/176"');
    expect(html).toMatch(/<\/span> \(PP - csspa\)$/);
  });

  it('gives every semantic coalition title a nested custom colour span', () => {
    const directories = [resolve(root, 'source/scenes/events/elections')];
    const titleLines = directories.flatMap((directory) =>
      readdirSync(directory)
        .filter((name) => /^(parlament|congreso)_coalition.*\.scene\.dry$/.test(name))
        .flatMap((name) => readFileSync(resolve(directory, name), 'utf8').split(/\r?\n/))
        .filter((line) => line.startsWith('title:') && /class="(?:generalitat|gobierno)-coalition"/.test(line)),
    );

    expect(titleLines.length).toBeGreaterThan(0);
    for (const line of titleLines) {
      const semantic = line.search(/class="(?:generalitat|gobierno)-coalition"/);
      expect(line, line).toContain('<span style=');
      expect(semantic, line).toBeGreaterThanOrEqual(0);
      expect(semantic, line).toBeLessThan(line.indexOf('<span style='));
    }
  });

  it('migrates election-simulation government labels too', () => {
    const source = readFileSync(resolve(root, 'source/scenes/election_simulation.scene.dry'), 'utf8');
    expect(source).toContain('data-parties="jxsi"');
    expect(source).toContain('data-parties="jxcat erc"');
    expect(source).toContain('data-parties="psoe up"');
  });

  it('authors concise summaries instead of cleaning them up in the renderer', () => {
    const sourceRoot = resolve(root, 'source/scenes');
    const collect = (directory: string): string[] => readdirSync(directory, { withFileTypes: true })
      .flatMap((entry) => entry.isDirectory()
        ? collect(resolve(directory, entry.name))
        : entry.name.endsWith('.dry') ? [resolve(directory, entry.name)] : []);
    const source = collect(sourceRoot).map((file) => readFileSync(file, 'utf8')).join('\n');

    expect(source).not.toMatch(/data-summary="[^"]*\bseats"/);
    expect(source).not.toContain(' + " seats"');
  });

  it.each([
    ['Catalonia 2015', 'election_simulation.post_election_sim', {
      ciu_parlament_showviz: 1, jxsi_parlament_showviz: 0, jxcat_parlament_showviz: 0,
      ciu_parlament_s: 50, parlament_s_majority: 68,
    }, 'ciu', 'Minority government - 50/68'],
    ['Catalonia 2017', 'election_simulation.post_election_sim', {
      ciu_parlament_showviz: 0, jxsi_parlament_showviz: 0, jxcat_parlament_showviz: 1,
      jxcat_parlament_s: 34, erc_parlament_s: 32, parlament_s_majority: 68,
    }, 'jxcat erc', 'Minority government - 66/68'],
    ['Congreso 2016', 'election_simulation.post_election_sim_congreso', {
      dl_congreso_showviz: 0, pdcat_congreso_showviz: 1, jxcat_congreso_showviz: 0,
      pp_congreso_s: 137, congreso_s_majority: 176,
    }, 'pp', 'Minority government - 137/176'],
    ['Congreso 2019', 'election_simulation.post_election_sim_congreso', {
      dl_congreso_showviz: 0, pdcat_congreso_showviz: 0, jxcat_congreso_showviz: 1,
      fr_congreso_showviz: 0, psoe_congreso_s: 120, up_congreso_s: 35,
      congreso_s_majority: 176,
    }, 'psoe up', 'Minority government - 155/176'],
  ])('renders the %s simulation tooltip as valid resolved markup', (_label, sceneId, q, parties, summary) => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, q);
    const host = document.createElement('div');
    host.innerHTML = adapter.renderView(sceneId);
    const marker = host.querySelector('.generalitat-coalition, .gobierno-coalition');
    expect(marker?.getAttribute('data-parties')).toBe(parties);
    expect(marker?.getAttribute('data-summary')).toBe(summary);
    expect(marker?.outerHTML).not.toContain('[+');
  });

  it('uses the old shell\'s expanding solid underline, with no resting dotted line', () => {
    const css = readFileSync(resolve(root, 'out/html/game.css'), 'utf8');
    expect(css).not.toContain('text-decoration: underline dotted currentColor 1px');
    expect(css).toContain('.generalitat-coalition:hover::after');
    expect(css).toContain('width: calc(100% - 10px)');
    expect(css).toContain('background: var(--mytooltip-color, currentColor)');
    expect(css).toMatch(/\.coalition-tooltip-logos\s*\{[^}]*justify-content:\s*center/s);
    expect(css).toMatch(/\.coalition-tooltip-logos\s*\{[^}]*margin-bottom:\s*12px/s);
  });

  it('centres and separates coalition logos in the new UI too', () => {
    const css = readFileSync(resolve(root, 'ui/src/styles/tooltips.css'), 'utf8');
    expect(css).toMatch(/\.coalition-logos\s*\{[^}]*justify-content:\s*center/s);
    expect(css).toMatch(/\.coalition-logos\s*\{[^}]*margin-bottom:\s*12px/s);
  });
});
