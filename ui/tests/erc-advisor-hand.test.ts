import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '../../out/game.json'), 'utf8');

function normalErc() {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  for (const id of ['root.start', 'root.normal_mode', 'root.esquerra', 'root.esquerra_2']) adapter.goToScene(id);
  adapter.goToScene('main');
  return adapter;
}

// The engine drops hand cards whose view-if fails whenever the hand is shown, so an
// advisor action must not start the borrowed card's own cooldown (playtest 2026-11-20).
describe('ERC advisor actions keep the borrowed card in hand', () => {
  const cases = [
    { advisor: 'sergi_sabria', flag: 'sergi_sabria_advisor', option: 'open_media', card: 'erc_media' },
    { advisor: 'carme_forcadell', flag: 'carme_forcadell_advisor', option: 'open_rally', card: 'erc_rally' },
    { advisor: 'junqueras', flag: 'oriol_junqueras_advisor', option: 'open_disunity', card: 'erc_disunity' },
    { advisor: 'marta_rovira', flag: 'marta_rovira_advisor', option: 'open_cup', card: 'erc_inter_party' },
  ];
  for (const { advisor, flag, option, card } of cases) {
    it(`${advisor}.${option} leaves ${card} in hand and its timer untouched`, () => {
      const a = normalErc();
      const q = a.qualities as Record<string, any>;
      Object.assign(q, { [flag]: 1, advisor_action_timer: 0, erc_advisor_action: null, month: 10,
        erc_dissent: 40, erc_bside_pending: false, cup_in_jxsi: false, cup_in_jxcat: false });
      const hand = (a.engine as any).state.currentHands.main as { id: string }[];
      if (!hand.some(c => c.id === card)) {
        if (hand.length >= 5) hand.shift();
        hand.push({ id: card, title: card } as any);
      }
      a.goToScene('main');
      expect(hand.map(c => c.id)).toContain(card);
      const timer = q[`${card}_timer`];

      const errors: string[] = [];
      const original = console.error;
      console.error = (...args: unknown[]) => { errors.push(args.join(' ')); };
      try {
        a.goToScene(`${advisor}.${option}`);
        expect(q.erc_advisor_action?.target).toBe(card);
        expect(q.advisor_action_timer).toBe(6);
        expect(q[`${card}_timer`]).toBe(timer);
        a.goToScene('main');
      } finally { console.error = original; }
      expect(errors.filter(e => /Scene (action|predicate|expression) error/.test(e))).toEqual([]);
      expect(hand.map(c => c.id)).toContain(card);
    });
  }
});
