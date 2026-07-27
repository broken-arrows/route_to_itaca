import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const game = JSON.parse(
  readFileSync(path.join(__dirname, '..', '..', 'out', 'game.json'), 'utf8'),
);

const CASES = [
  {
    scene: '2012parlamentvote.parlament2012voteresult',
    configFrom: 'parlament2012_vote_result',
  },
  {
    scene: 'parlament_sovereignty_declaration.results',
    configFrom: 'parlament_sovereignty_declaration_vote_result',
  },
  {
    scene: 'parlament_non_binding_consultations.calc',
    configFrom: 'parlament_consultations_vote_result',
  },
  {
    scene: '2013congresdecidir.calc',
    configFrom: 'congreso_decidir_vote_result',
  },
] as const;

function stringsIn(node: unknown): string[] {
  if (typeof node === 'string') return [node];
  if (Array.isArray(node)) return node.flatMap(stringsIn);
  if (node && typeof node === 'object') return Object.values(node).flatMap(stringsIn);
  return [];
}

describe('compiled legislative vote scenes', () => {
  for (const entry of CASES) {
    it(`${entry.scene} declares one chamber-vote model and no injected table`, () => {
      const scene = game.scenes[entry.scene];
      expect(scene, `missing compiled scene ${entry.scene}`).toBeDefined();
      const renderedStrings = stringsIn(scene.content);
      const markers = renderedStrings.filter((s) => s.includes('data-widget="chamber-vote"'));

      expect(markers).toHaveLength(1);
      expect(markers[0]).toContain(`"configFrom":"${entry.configFrom}"`);
      expect(renderedStrings.some((s) => /<table\b/i.test(s))).toBe(false);

      const arrivalCode = stringsIn(scene.onArrival).join('\n');
      expect(arrivalCode).toContain(`Q.${entry.configFrom} =`);
      expect(arrivalCode).toContain('outcomes:');
      expect(arrivalCode).toContain('parties:');
    });
  }
});
