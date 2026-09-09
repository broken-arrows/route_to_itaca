import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { gameLib } from '../src/game-bindings';

function run(file: string, q: Record<string, any>, section?: string) {
  let source = readFileSync(path.resolve(__dirname, '../../source/scenes', file), 'utf8');
  if (section) {
    const header = new RegExp(`^@${section}\\r?$`, 'm').exec(source);
    if (!header) throw new Error(`Missing scene section ${section}`);
    source = source.slice(header.index);
  }
  const code = source.match(/on-arrival:\s*\{!([\s\S]*?)!\}/)![1];
  new Function('Q','G',code).call({achieve:()=>{}},q,gameLib);
}

describe('consultation alternate paths', () => {
  it('changes relations with the current CiU-family carrier without touching nonexistent successors', () => {
    for (const carrier of ['ciu','cdc','dl','pdcat','jxcat','junts']) for (const section of ['short','complex']) {
      const q: Record<string, any> = { player_party:'erc',president_party:'erc',parlament_current_ciu:carrier,
        [`${carrier}_relations`]: 50, cup_relations:30, independence_trust:30, independence_movement:45,
        erc_civic_dissent:10,erc_civic_strength:10,erc_left_dissent:0,erc_core_dissent:0,erc_pragmatic_dissent:0,erc_pragmatic_strength:10 };
      run('events/referendum/question_of_the_question.scene.dry',q,section);
      expect(q[`${carrier}_relations`]).toBe(50 + (section === 'complex' ? 5 : carrier === 'ciu' ? -5 : -2));
      expect(Object.entries(q).filter(([k])=>k.endsWith('_relations')).every(([,v])=>Number.isFinite(v))).toBe(true);
      for (const dormant of ['ciu','cdc','dl','pdcat','jxcat','junts'].filter(p=>p!==carrier)) expect(q[`${dormant}_relations`]).toBeUndefined();
    }
  });

  it('conserves consultation-launch transfers, including cells with insufficient ERC support', () => {
    for (const opening of [10,0.2]) {
      const q: Record<string, any> = { player_party:'erc',coming_from_parlament:false,parlament_current_ciu:'cdc',
        parties:['ciu','cdc','erc','cup'],parlament_constituencies:['barcelona','girona','lleida','tarragona'],parlament_demographics:['middle','rural'], cdc_parlament_s:50 };
      for(const p of q.parlament_constituencies) for(const d of q.parlament_demographics) {
        q[`erc_parlament_${p}_${d}_support`]=opening;
        q[`cdc_parlament_${p}_${d}_support`]=30;
        q[`cup_parlament_${p}_${d}_support`]=10;
        q[`ciu_parlament_${p}_${d}_support`]=0;
        q[`abstain_parlament_${p}_${d}_support`]=60-opening;
      }
      run('parlament/parlament_non_binding_consultations.scene.dry',q);
      for(const p of q.parlament_constituencies) for(const d of q.parlament_demographics) {
        const supports=[...q.parties,'abstain'].map(party=>q[`${party}_parlament_${p}_${d}_support`]);
        expect(supports.every(v=>Number.isFinite(v)&&v>=0)).toBe(true);
        expect(supports.reduce((n,v)=>n+v,0)).toBeCloseTo(100,9);
      }
      // Preserve the authored loss budget and recipient ratio, not the old mismatch.
      const loss=Math.min(opening,3);
      expect(q.erc_parlament_girona_rural_support).toBeCloseTo(opening-loss);
      expect(q.cdc_parlament_girona_rural_support).toBeCloseTo(30+loss/2);
      expect(q.cup_parlament_girona_rural_support).toBeCloseTo(10+loss/2);
    }
  });
});
