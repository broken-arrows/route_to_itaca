import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';

const gameText = readFileSync(resolve(import.meta.dirname, '..', '..', 'out', 'game.json'), 'utf8');

function enterDiada(scene: string, state: Record<string, unknown>) {
  const adapter = DendryAdapter.fromJSONText(gameText);
  adapter.beginGame([1, 2, 3, 4]);
  Object.assign(adapter.qualities, {
    year: Number(scene.slice(0, 4)),
    month: 9,
    rubicon: false,
    broken_roadmap: false,
    referendum_pending: false,
    consultation_pending: false,
    cat_caretaker_gov: false,
    podemos_channeling: 0,
    ...state,
  });
  const frame = adapter.goToScene(scene);
  return { frame, q: adapter.qualities };
}

describe('Diada participation', () => {
  it('keeps the exceptional 2013 and 2014 mobilizations larger than the 2015 election Diada', () => {
    const y2013 = enterDiada('2013diada', {
      independence_movement: 65.77,
      independence_trust: 31.66,
      social_dissent: 68.26,
      cat_spa_relations: 35.72,
    });
    const y2014 = enterDiada('2014diada', {
      independence_movement: 58.14,
      independence_trust: 33.8,
      social_dissent: 64.81,
      cat_spa_relations: 33.57,
      consultation_pending: true,
    });
    const y2014WithoutConsultation = enterDiada('2014diada', {
      independence_movement: 58.14,
      independence_trust: 33.8,
      social_dissent: 64.81,
      cat_spa_relations: 33.57,
    });
    const y2015 = enterDiada('2015diada', {
      independence_movement: 68.02,
      independence_trust: 42.34,
      social_dissent: 59.94,
      cat_spa_relations: 27.41,
      cat_caretaker_gov: true,
    });

    expect(y2013.q.diada_size).toBe(1.57);
    expect(y2014.q.diada_size).toBe(1.76);
    expect(y2014WithoutConsultation.q.diada_size).toBe(1.56);
    expect(y2015.q.diada_size).toBe(1.37);
  });

  it('prints the calculated participation in the 2014 report', () => {
    const { frame } = enterDiada('2014diada', {
      independence_movement: 58.14,
      independence_trust: 33.8,
      social_dissent: 64.81,
      cat_spa_relations: 33.57,
      consultation_pending: true,
    });

    expect(frame.html).toContain('1.76 million Catalans');
  });

  it('preserves the later decline, referendum mobilization, and post-155 grief pattern', () => {
    const cases = [
      ['2016diada', 68.93, 50.33, 54.69, 15.06, {}, 0.88],
      ['2017diada', 63.71, 46.22, 51.19, 12.13, { referendum_pending: true }, 1.0],
      ['2018diada', 70.94, 39.0, 53.73, 5.2, { broken_roadmap: true }, 0.94],
      ['2019diada', 64.3, 38.73, 45.07, 6.15, { broken_roadmap: true }, 0.6],
    ] as const;

    for (const [scene, movement, trust, dissent, relations, flags, expected] of cases) {
      const { q } = enterDiada(scene, {
        independence_movement: movement,
        independence_trust: trust,
        social_dissent: dissent,
        cat_spa_relations: relations,
        ...flags,
      });
      expect(q.diada_size, scene).toBe(expected);
    }
  });
});

describe('consultation scheduling', () => {
  it('marks a consultation pending whenever its voting day is scheduled', () => {
    const adapter = DendryAdapter.fromJSONText(gameText);
    adapter.beginGame([1, 2, 3, 4]);
    Object.assign(adapter.qualities, {
      consultation_pending: false,
      consultation_happened: false,
      congreso_decidir_trigger: true,
      question_of_the_question_trigger: true,
      president_party: 'ciu',
      player_party: 'erc',
      countdowns: [],
    });

    adapter.goToScene('question_of_the_question');

    expect(adapter.qualities.consultation_pending).toBe(true);
    expect(adapter.qualities.consultation_day_countdown).toBe(3);
    expect(adapter.qualities.countdowns).toContain('consultation_day');
  });
});
