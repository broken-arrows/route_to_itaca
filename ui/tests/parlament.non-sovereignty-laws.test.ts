import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const repo = resolve(import.meta.dirname, '..', '..');
const scene = (relative: string) =>
  readFileSync(resolve(repo, 'source', 'scenes', relative), 'utf8');

describe('non-sovereignty law pilots', () => {
  it('keeps authorship shifts on explicit electoral variables', () => {
    const urbanism = scene('parlament/parlament_urbanism.scene.dry');
    const inheritance = scene('parlament/parlament_inheritance_tax.scene.dry');
    const digital = scene('parlament/parlament_digital.scene.dry');

    for (const source of [urbanism, inheritance, digital]) {
      expect(source).not.toContain('spaSupportInject');
      expect(source).toContain('"_young_support"');
    }
    expect(urbanism).toContain('"_unemployed_support"');
    expect(inheritance).toContain('"_unemployed_support"');
    expect(urbanism).toContain('"_middle_support"');
    expect(inheritance).toContain('"_buss_support"');
    expect(inheritance).toContain('"_retired_support"');
    expect(digital).toContain('"_middle_support"');
    expect(digital).toContain('"_buss_support"');
  });

  it('allows failed bills to be retried but disables bills that passed', () => {
    for (const relative of [
      'parlament/parlament_urbanism.scene.dry',
      'parlament/parlament_inheritance_tax.scene.dry',
      'parlament/parlament_digital.scene.dry',
    ]) {
      const law = scene(relative);
      expect(law).toMatch(/choose-if: party_resources > 0 and not parlament_.+_passed/);
      expect(law).toContain('Already approved!');
      expect(law).not.toContain('_attempted');
      expect(law).not.toMatch(/@.+_already/);
    }
  });

  it('forces the proposing player party into the yes bloc', () => {
    for (const relative of [
      'parlament/parlament_urbanism.scene.dry',
      'parlament/parlament_inheritance_tax.scene.dry',
      'parlament/parlament_digital.scene.dry',
    ]) {
      const law = scene(relative);
      expect(law).toContain('var playerProposal = Q.coming_from_parlament');
      expect(law).toContain(
        'if (playerProposal && !yesNames.includes(proposingParty)) yesNames.push(proposingParty)',
      );
    }
  });

  it('challenges both vacant-homes variants and fully upholds the enacted law', () => {
    const law = scene('parlament/parlament_urbanism.scene.dry');
    const ruling = scene('events/constitutional/constitutional_vacant_homes.scene.dry');

    expect(law).toContain('id:"vacant_homes_tax"');
    expect(law).toContain('id:"vacant_homes_mobilization"');
    expect(law).toContain('Q.constitutional_vacant_homes_countdown = 30');
    expect(ruling).not.toContain('deactivateLaw');
  });

  it('models the digital agency core and national scope as severable provisions', () => {
    const law = scene('parlament/parlament_digital.scene.dry');
    const ruling = scene(
      'events/constitutional/constitutional_digital.scene.dry',
    );

    expect(law).toContain('id:"digital_agency_core"');
    expect(law).toContain('gdp_growth:0.015');
    expect(law).toContain('generalitat_surplus:-0.010');
    expect(law).toContain('id:"digital_agency_national_scope"');
    expect(law).toContain('gdp_growth:0.020');
    expect(law).toContain('generalitat_surplus:-0.015');
    expect(law).toContain('Q.independence_trust += 1');
    expect(law).not.toContain('digital_capacity');
    expect(law).not.toContain('social_dissent');
    expect(ruling).toContain(
      'G.deactivateLaw(Q, "digital_agency_national_scope", "struck_down")',
    );
    expect(ruling).not.toContain(
      'G.deactivateLaw(Q, "digital_agency_core"',
    );
    expect(ruling).toContain(
      'Q.cybersecurity_agency_scope = "generalitat"',
    );
  });

  it('uses a countdown rather than a hardcoded court year for both digital drafts', () => {
    const law = scene('parlament/parlament_digital.scene.dry');

    expect(law).toContain('Q.countdowns.push("constitutional_digital")');
    expect(law).not.toMatch(/constitutional_digital.+20\d{2}/);
  });

  it('varies the maximal digital vote by PSC and federal-left leadership', () => {
    const law = scene('parlament/parlament_digital.scene.dry');

    for (const leader of [
      'Montserrat Tura',
      'Àngel Ros',
      'Núria Parlon',
      'Arcadi Oliveres',
      'Jaume Asens',
      'Joan Coscubiela',
    ]) {
      expect(law).toContain(`"${leader}"`);
    }
    expect(law).toContain(
      "TODO: Revisit the CUP's votes once CUP leadership dynamics are modelled.",
    );
  });

  it('leaves inheritance tax outside the constitutional countdown system', () => {
    const law = scene('parlament/parlament_inheritance_tax.scene.dry');

    expect(law).toContain('id:"inheritance_tax_partial"');
    expect(law).toContain('id:"inheritance_tax_strong"');
    expect(law).not.toContain('Q.inequality');
    expect(law).not.toContain('constitutional_');
    expect(law).not.toContain('Q.countdowns.push');
  });

  it('uses only real engine effects for vacant-home mobilization', () => {
    const law = scene('parlament/parlament_urbanism.scene.dry');

    expect(law).not.toContain('Q.housing_mobilization');
    expect(law).toContain('welfare_index_growth:0.12');
    expect(law).toContain('social_dissent_eq:-2');
  });
});
