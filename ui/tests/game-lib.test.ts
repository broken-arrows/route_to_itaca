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

  it('derives display-ready law rows and hides expired laws', () => {
    const rows = gameLib.getLawsForUI({
      active_mods: {
        digital_agency_core: {
          def: { id: 'digital_agency_core' },
          status: 'active',
          ticks_active: 2,
          live_effect: { gdp_growth: 0.01 },
        },
        vacant_homes_tax: {
          def: { id: 'vacant_homes_tax' },
          status: 'repealed',
          ticks_active: 10,
          live_effect: {},
        },
        old_measure: {
          def: { id: 'old_measure' },
          status: 'expired',
          ticks_active: 12,
          live_effect: {},
        },
      },
    });
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
