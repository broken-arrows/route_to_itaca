import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { gameLib } from '../src/game-bindings';

function run(file: string, q: Record<string, any>, section?: string) {
  let source = readFileSync(path.join(__dirname, '../../source/scenes', file), 'utf8').replace(/\r\n/g, '\n');
  if (section) source = source.slice(source.indexOf(section));
  const code = source.match(/on-arrival:\s*\{!([\s\S]*?)!\}/)![1];
  new Function('Q', 'G', code).call({ state: { currentHands: {} } }, q, gameLib);
}
const key = (p: string, d: string) => `${p}_parlament_girona_${d}_support`;
describe('structural-event electorate accounting', () => {
  it('caps repeated JxSi refusal at ERC support while conserving every demographic', () => {
    const parties = ['erc', 'cdc', 'cup', 'fnc', 'icv'];
    const demographics = ['buss', 'ind', 'young', 'rural', 'retired', 'unemployed'];
    const q: Record<string, any> = {
      parties, parlament_constituencies: ['girona'], parlament_demographics: demographics,
      parlament_current_ciu: 'cdc', cdc_parlament_s: 50, icv_parlament_s: 10,
      fnc_formed: true, pxc_dissolved: true,
    };
    for (const d of demographics) {
      for (const p of [...parties, 'abstain']) q[key(p, d)] = 0;
      q[key('erc', d)] = 2; q[key('cdc', d)] = 30;
      q[key('cup', d)] = 8; q[key('icv', d)] = 10; q[key('abstain', d)] = 50;
    }
    for (let i = 0; i < 3; i++) {
      run('party_paths/jxsi/jxsi_formation.scene.dry', q, '@erc_no_for_sure\non-arrival:');
      for (const d of demographics) {
        const support = [...parties, 'abstain'].map(p => q[key(p, d)]);
        expect(support.every(v => v >= 0)).toBe(true);
        expect(support.reduce((a,b) => a+b,0)).toBeCloseTo(100);
      }
    }
    expect(q[key('erc', 'rural')]).toBe(0);
    expect(q[key('fnc', 'retired')]).toBeGreaterThan(0);
  });

  it.each([
    ['icv/csqp_formation', 'icv', 'csqp', .95], ['icv/cecp_formation', 'csqp', 'cecp', .95],
    ['icv/ecp_formation', 'cecp', 'ecp', .95], ['ciu/dl_formation', 'cdc', 'dl', .95],
    ['ciu/pdcat_formation', 'dl', 'pdcat', .95], ['ciu/junts_formation', 'jxcat', 'junts', .98],
  ] as const)('keeps the organizational loss in the electorate for %s', (scene, from, to, retained) => {
    const q: Record<string, any> = {
      parlament_constituencies: ['girona'], parlament_demographics: ['rural'],
      cat_coalition: [], countdowns: [], [key(from, 'rural')]: 20,
      [key(to, 'rural')]: 0, [key('abstain', 'rural')]: 80,
    };
    run(`party_paths/${scene}.scene.dry`, q);
    expect(q[key(from, 'rural')]).toBe(0);
    expect(q[key(to, 'rural')]).toBeCloseTo(20 * retained);
    expect(q[key('abstain', 'rural')]).toBeCloseTo(100 - 20 * retained);
  });
});
