import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, beforeAll, expect, it, vi } from 'vitest';
import { compileGame, convertGameToJSON } from 'dendrynexus-ten/lib/parsers/compiler.js';
import { DendryAdapter } from '../src/engine/adapter';

let gameText: string;
beforeAll(async () => {
  // Compile the actual action/menu sources without rebuilding unrelated content.
  const files = [
    'generalitat/generalitat_foreign_card',
    'erc_party_affairs/erc_international',
    'unity_party_affairs/unity_international',
  ].map(scene => {
    const name = resolve(import.meta.dirname, `../../source/scenes/${scene}.scene.dry`);
    return { name, contents: readFileSync(name, 'utf8') };
  });
  for (const id of ['root', 'post_event', 'easy_discard', 'cancel_advisor_action']) {
    files.push({ name: `${id}.scene.dry`, contents: `title: ${id}\n\nFixture destination.\n` });
  }
  const freshText = await new Promise<string>((done, reject) => {
    compileGame(files, (error, game) => {
      if (error) return reject(error);
      convertGameToJSON(game, 0, (serializationError, json) => serializationError ? reject(serializationError) : done(json));
    });
  });
  const baseline = JSON.parse(readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8'));
  for (const [id, scene] of Object.entries(JSON.parse(freshText).scenes)) {
    if (/^(generalitat_foreign_card|erc_international|unity_international)(\.|$)/.test(id)) baseline.scenes[id] = scene;
  }
  gameText = JSON.stringify(baseline);
});
const errors = vi.spyOn(console, 'error').mockImplementation(() => {});

afterEach(() => {
  const actionErrors = errors.mock.calls.filter(([message]) => String(message).includes('Scene action error'));
  errors.mockClear();
  expect(actionErrors).toEqual([]);
});

function setup(overrides: Record<string, unknown> = {}) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  adapter.goToScene('root.start');
  const q = adapter.qualities as Record<string, any>;
  Object.assign(q, {
    player_party: 'erc', player_in_jxsi: false, player_in_jxcat: false,
    cat_caretaker_gov: false, foreign_minister_party: 'erc',
    generalitat_foreign_timer: 0, erc_international_timer: 0, unity_international_timer: 0,
    art155_ever: false, government_budget: 2, party_resources: 3,
    cat_spa_relations: 30, erc_advisor_action: false, raul_romeva_advisor: 0,
    ...overrides,
  });
  return { adapter, q };
}

it.each([
  ['pp', 5], ['psoe', 8], ['cs', 2], ['vox', 2],
  ['podemos', 12], ['iu', 12], ['up', 12],
] as const)('Foreign Affairs improves relations under %s by %i', (party, increase) => {
  const { adapter, q } = setup({ spanish_pm_party: party });
  const frame = adapter.goToScene('generalitat_foreign_card');
  const index = frame.choices.findIndex(choice => choice.id === 'generalitat_foreign_card.spain_relations' && choice.canChoose);
  expect(index).toBeGreaterThanOrEqual(0);
  adapter.choose(index);
  expect(q.cat_spa_relations).toBe(30 + increase);
  expect(q.government_budget).toBe(1);
});

it.each([
  ['erc_international', false, false, 1],
  ['erc_international', true, false, 1],
  ['unity_international', false, false, 2],
  ['unity_international', false, true, 2],
] as const)('%s respects accumulated outreach (advisor: %s, JxCat: %s)', (scene, advisor, jxcat, increase) => {
  const { adapter, q } = setup({
    erc_advisor_action: advisor, player_in_jxsi: scene === 'unity_international' && !jxcat,
    player_in_jxcat: jxcat,
    // Existing games already persist their progress under this spelling.
    minor_contries_reach: 4,
  });
  const frame = adapter.goToScene(scene);
  const id = `${scene}.minor_countries_partyreach`;
  const index = frame.choices.findIndex(choice => choice.id === id && choice.canChoose);
  expect(index).toBeGreaterThanOrEqual(0);
  adapter.choose(index);
  expect(q.minor_contries_reach).toBe(4 + increase);
  expect(q.party_resources).toBe(advisor ? 3 : 2);

  const next = adapter.goToScene(scene).choices.find(choice => choice.id === id);
  expect(next).toBeDefined();
  expect(next!.canChoose).toBe(false);
  expect(next!.subtitle?.trim()).toBeTruthy();
});
