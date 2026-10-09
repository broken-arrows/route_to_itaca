import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const directory = resolve(import.meta.dirname, '../../source/scenes/events/elections');
const files = readdirSync(directory).filter(n => n.includes('_phase') && n.endsWith('.scene.dry'));
describe('formation push coverage', () => {
  it.each(files)('provides a push in each negotiation submenu: %s', file => {
    const source = readFileSync(resolve(directory, file), 'utf8');
    const nodes = source.split(/\r?\n@/);
    for (const name of ['policies', 'policy_concessions', 'ministries', 'changes', 'counterproposal', 'cabinet']) {
      const node = nodes.find(n => n.startsWith(name + '\n') || n.startsWith(name + '\r\n'));
      if (node) expect(node, name).toMatch(/^- @push\w*/m);
    }
    const top = nodes.find(n => n.startsWith('choices_menu\n') || n.startsWith('choices_menu\r\n'));
    if (top) expect(top).not.toMatch(/^- @push\w*/m);
  });
  it.each(files)('cannot push twice in round two after skipping round one: %s', file => {
    const source = readFileSync(resolve(directory, file), 'utf8');
    const nodes = source.split(/\r?\n@/);
    const menu = nodes.find(n => /^- @push\w*/m.test(n));
    expect(menu).toBeTruthy();
    const scene = file.replace('.scene.dry', '');
    const round = source.match(/^view-if: (\w+(?:coalition_phase|sovereignty_phase))/m)![1];
    const a = DendryAdapter.fromJSONText(readFileSync(resolve(directory, '../../../../out/game.json'), 'utf8'));
    a.beginGame([1, 2, 3, 4]); a.goToScene('root.start_menu'); a.goToScene('root.start');
    Object.assign(a.qualities, {
      [round]: 2, player_party: file.includes('cup_player') ? 'cup' : 'erc',
      parlament_current_ciu: 'ciu', parlament_current_icv: 'icv',
      parlament_coalition_push: 0, parlament_coalition_push_round: 0,
      parlament_coalition_leverage: 0, parlament_coalition_changes: false,
    });
    let frame = a.goToScene(scene + '.' + menu!.split(/\r?\n/)[0]);
    const index = frame.choices.findIndex(c => c.id.startsWith(scene + '.push') && c.canChoose);
    expect(index).toBeGreaterThanOrEqual(0);
    const chosen = frame.choices[index].id;
    frame = a.choose(index);
    expect(frame.choices.some(c => c.id === chosen)).toBe(true);
    expect(a.qualities.parlament_coalition_leverage).toBe(4);
    expect(a.qualities.parlament_coalition_push).toBe(1);
    expect(a.qualities.parlament_coalition_changes).toBe(true);
    for (const node of nodes.filter(n => /^- @push\w*/m.test(n))) {
      frame = a.goToScene(scene + '.' + node.split(/\r?\n/)[0]);
      expect(frame.choices.filter(c => c.id.startsWith(scene + '.push') && c.canChoose)).toHaveLength(0);
    }
    const restored = DendryAdapter.fromJSONText(readFileSync(resolve(directory, '../../../../out/game.json'), 'utf8'));
    restored.importStateJSON(a.exportStateJSON());
    frame = restored.goToScene(scene + '.' + menu!.split(/\r?\n/)[0]);
    expect(frame.choices.filter(c => c.id.startsWith(scene + '.push') && c.canChoose)).toHaveLength(0);
    a.qualities[round] = 3;
    for (const node of nodes.filter(n => /^- @push\w*/m.test(n))) {
      frame = a.goToScene(scene + '.' + node.split(/\r?\n/)[0]);
      expect(frame.choices.filter(c => c.id.startsWith(scene + '.push') && c.canChoose)).toHaveLength(0);
    }
    // A new first round, followed by round two, gives one push in each.
    Object.assign(a.qualities, { [round]: 1, parlament_coalition_push: 0, parlament_coalition_push_round: 0, parlament_coalition_leverage: 0 });
    frame = a.goToScene(scene + '.' + menu!.split(/\r?\n/)[0]);
    a.choose(frame.choices.findIndex(c => c.id.startsWith(scene + '.push') && c.canChoose));
    Object.assign(a.qualities, { [round]: 2, parlament_coalition_leverage: 0 });
    frame = a.goToScene(scene + '.' + menu!.split(/\r?\n/)[0]);
    a.choose(frame.choices.findIndex(c => c.id.startsWith(scene + '.push') && c.canChoose));
    expect(a.qualities.parlament_coalition_push).toBe(2);
  });
});
