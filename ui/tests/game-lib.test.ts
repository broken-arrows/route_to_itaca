import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { gameLib } from '../src/game-bindings';

describe('source/lib', () => {
  it('exports the two functions content calls as G.*', () => {
    expect(typeof gameLib.engineTick).toBe('function');
    expect(typeof gameLib.spaSupportInject).toBe('function');
    expect(typeof gameLib.getLawsForUI).toBe('function');
    expect(typeof gameLib.governmentTooltip).toBe('function');
  });

  it('builds escaped semantic government markup without swallowing an adjacent party name', () => {
    const span = gameLib.governmentTooltip(
      'gobierno',
      ['pp', 'vox'],
      'Majority government - 180/176',
      '<i>National Right government</i>',
    );
    expect(`PP ${span}`).toBe(
      'PP <span class="gobierno-coalition" data-parties="pp vox" ' +
      'data-summary="Majority government - 180/176"><i>National Right government</i></span>',
    );
  });

  it('supports all three pinned institutions and safely escapes authored summaries', () => {
    expect(gameLib.governmentTooltip('generalitat', ['jxsi'], 'Caretaking capacities only', 'caretaker'))
      .toContain('class="generalitat-coalition" data-parties="jxsi"');
    expect(gameLib.governmentTooltip('ajuntament', ['bcomu', 'psc'], '18/21 & "stable"', 'coalition'))
      .toContain('data-summary="18/21 &amp; &quot;stable&quot;"');
  });

  it('derives display-ready law rows and hides flagged laws', () => {
    const rows = gameLib.getLawsForUI({
      active_mods: {
        digital_agency_core: {
          def: {
            id: 'digital_agency_core', colour: 'green',
            description: 'Coordinates cybersecurity.',
          },
          active: true,
          hidden: false,
          ticks_active: 2,
          live_effect: { gdp_growth: 0.01 },
        },
        vacant_homes_tax: {
          def: {
            id: 'vacant_homes_tax', colour: 'green',
            description: 'Taxes vacant homes.',
          },
          active: false,
          hidden: false,
          colour: 'red',
          ticks_active: 10,
          live_effect: {},
        },
        old_measure: {
          def: { id: 'old_measure' },
          active: false,
          hidden: true,
          ticks_active: 12,
          live_effect: {},
        },
      },
    });
    expect(rows.map((row) => row.id)).toEqual(['digital_agency_core', 'vacant_homes_tax']);
    expect(rows.map((row) => [row.colour, row.description])).toEqual([
      ['green', 'Coordinates cybersecurity.'],
      ['red', 'Taxes vacant homes.'],
    ]);
  });

  it('keeps authored display data separate from law lifecycle and targets', () => {
    const laws = gameLib as any;
    const q: Record<string, any> = { year: 2015, month: 4 };
    const targets = { gdp_growth: 0.03 };
    laws.registerLaw(q, {
      id: 'civil_code', title: 'Civil Code', icon: 'img/scales_icon.svg',
      colour: 'green', description: 'Updates civil law.', targets,
    });
    expect(laws.getLawsForUI(q)[0]).toMatchObject({
      colour: 'green', description: 'Updates civil law.',
    });
    expect(q.active_mods.civil_code.active).toBe(true);

    laws.deactivateLaw(q, 'civil_code', 'The courts and Parlament disagree on its status.', false, 'orange');
    expect(laws.getLawsForUI(q)[0]).toMatchObject({
      colour: 'orange',
      description: 'The courts and Parlament disagree on its status.',
    });
    expect(q.active_mods.civil_code.active).toBe(false);
    expect(q.active_mods.civil_code.hidden).toBe(false);
    expect(q.active_mods.civil_code.def.colour).toBe('green');
    expect(q.active_mods.civil_code.def.targets).toBe(targets);
    expect(q.mod_log.map((entry: { action: string }) => entry.action)).toEqual(['enacted', 'deactivated']);

    laws.registerLaw(q, {
      id: 'temporary', title: 'Temporary law', icon: 'img/scales_icon.svg',
      colour: 'green', description: 'A temporary measure.', targets: { gdp_growth: 0.01 },
    });
    laws.deactivateLaw(q, 'temporary', undefined, true);
    expect(q.active_mods.temporary).toMatchObject({ active: false, hidden: true });
    expect(laws.getLawsForUI(q).map((row: { id: string }) => row.id)).toEqual(['civil_code']);
  });

  it('is DOM-free — that is the only reason it can be shared', () => {
    const src = readFileSync(resolve(__dirname, '../../source/lib/cat_engine.js'), 'utf8');
    // `window.RTI_CAT_ENGINE` in the no-module fallback is the one allowed
    // mention; strip it before asserting.
    const body = src.replace(/window\.RTI_CAT_ENGINE/g, '');
    expect(body).not.toMatch(/\bdocument\./);
    expect(body).not.toMatch(/\bd3\./);
    expect(body).not.toMatch(/\bwindow\./);
  });
});
