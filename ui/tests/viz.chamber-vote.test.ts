import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import ChamberVote from '../src/components/viz/ChamberVote.vue';
import { useGameStore } from '../src/stores/game';
import { i18n } from '../src/i18n';

const GAME = {
  scenes: {
    root: {
      id: 'root',
      type: 'scene',
      title: 'Root',
      content: [{ type: 'paragraph', content: ['Hi.'] }],
      options: [],
    },
  },
  qualities: {},
  qdisplays: {},
  data: {
    glossary: {
      terms: [
        { id: 'ciu', match: ['CiU'], display: 'CiU', colour: 'ciu', tooltip: { title: 'Convergència i Unió' } },
        { id: 'psc', match: ['PSC'], display: 'PSC', colour: 'psc' },
      ],
    },
  },
};

function mountVote() {
  const pinia = createPinia();
  setActivePinia(pinia);
  const game = useGameStore();
  game.initFromText(JSON.stringify(GAME));
  game.newGame();
  return mount(ChamberVote, {
    props: {
      outcomes: [
        {
          kind: 'yes',
          label: 'Yes',
          votes: 84,
          parties: [{ label: 'CiU' }, { label: 'PSC', count: 3 }],
        },
        { kind: 'abstain', label: 'Abstention', votes: 0, parties: [{ label: 'PSC' }] },
        { kind: 'no', label: 'No', votes: 51 },
      ],
    },
    global: { plugins: [pinia, i18n] },
  });
}

describe('ChamberVote', () => {
  it('uses vote totals as flex weights and omits zero-vote outcomes', () => {
    const w = mountVote();
    const outcomes = w.findAll('.chamber-vote-outcome');
    expect(outcomes).toHaveLength(2);
    expect(outcomes[0].attributes('style')).toContain('flex-grow: 84');
    expect(outcomes[1].attributes('style')).toContain('flex-grow: 51');
    expect(w.text()).not.toContain('Abstention');
  });

  it('marks the absolute majority over all seats, including absent legislators', async () => {
    const w = mountVote();
    const bar = w.get('.chamber-vote-bar');
    expect(bar.attributes('data-majority')).toBe('68');
    expect(bar.attributes('title')).toBe('Majority: 68 yes votes');
    expect(bar.attributes('style')).toContain('--chamber-vote-majority-left: 50.37037037037037%');

    await w.setProps({
      outcomes: [
        { kind: 'yes', label: 'Yes', votes: 60 },
        { kind: 'abstain', label: 'Abstention', votes: 10 },
        { kind: 'not-present', label: 'Not present', votes: 15 },
        { kind: 'no', label: 'No', votes: 50 },
      ],
    });
    expect(bar.attributes('data-majority')).toBe('68');

    await w.setProps({
      outcomes: [
        { kind: 'yes', label: 'Yes', votes: 60 },
        { kind: 'abstain', label: 'Abstention', votes: 10 },
        { kind: 'no', label: 'No', votes: 50 },
      ],
    });
    expect(bar.attributes('data-majority')).toBe('61');
    expect(bar.attributes('style')).toContain('--chamber-vote-majority-left: 50.83333333333333%');

    await w.setProps({ outcomes: [] });
    expect(bar.attributes('data-majority')).toBeUndefined();
    expect(bar.attributes('title')).toBeUndefined();
    expect(bar.attributes('style')).toBeUndefined();
  });

  it('accepts optional party breakdowns and optional split-caucus counts', () => {
    const w = mountVote();
    expect(w.findAll('.chamber-vote-parties li')).toHaveLength(2);
    expect(w.text()).toContain('PSC (3)');
    expect(w.find('.chamber-vote-breakdown--no .chamber-vote-parties').exists()).toBe(false);
  });

  it('marks generated party names and opens their glossary tooltip', async () => {
    const w = mountVote();
    const ciu = w.get('[data-term="ciu"]');
    expect(ciu.text()).toBe('CiU');
    expect(ciu.attributes('style')).toContain('var(--ciu)');
    expect(ciu.classes()).toContain('term-hoverable');
    await ciu.trigger('mouseover');
    expect(document.querySelector('[data-test="glossary-popover"]')?.textContent).toContain('Convergència i Unió');
    w.unmount();
  });

  it('aligns the readable abstention column with its proportional bar segment', async () => {
    const w = mountVote();
    await w.setProps({
      outcomes: [
        { kind: 'yes', label: 'Yes', votes: 84 },
        {
          kind: 'abstain',
          label: 'Abstention',
          votes: 15,
          parties: [{ label: 'PSC' }],
        },
        { kind: 'no', label: 'No', votes: 51 },
      ],
    });

    const shift = parseFloat(
      (w.get('.chamber-vote').attributes('style') ?? '').match(
        /--chamber-vote-abstain-shift:\s*([^%;]+)%/,
      )?.[1] ?? '',
    );
    expect(shift).toBeCloseTo(33);
  });

  it('places absent parties between abstention and no in a transparent segment', async () => {
    const w = mountVote();
    await w.setProps({
      outcomes: [
        { kind: 'no', label: 'No', votes: 30 },
        { kind: 'not-present', label: 'Not present', votes: 15, parties: [{ label: 'PSC' }] },
        { kind: 'yes', label: 'Yes', votes: 80 },
        { kind: 'abstain', label: 'Abstention', votes: 10 },
      ],
    });

    expect(w.findAll('.chamber-vote-outcome').map((part) => part.attributes('aria-label'))).toEqual([
      'Yes: 80 votes', 'Abstention: 10 votes', 'Not present: 15 votes', 'No: 30 votes',
    ]);
    expect(w.get('.chamber-vote-outcome--not-present').attributes('style')).toContain('flex-grow: 15');
    expect(w.get('.chamber-vote-breakdown--not-present [data-term="psc"]').text()).toBe('PSC');
    const shift = parseFloat(
      (w.get('.chamber-vote').attributes('style') ?? '').match(
        /--chamber-vote-not-present-shift:\s*([^%;]+)%/,
      )?.[1] ?? '',
    );
    expect(shift).toBeCloseTo(38.888889);
    const source = readFileSync(path.join(__dirname, '..', 'src', 'components', 'viz', 'ChamberVote.vue'), 'utf8');
    expect(source).toMatch(/\.chamber-vote-outcome--not-present\s*{[^}]*background:\s*transparent;/s);

    await w.setProps({
      outcomes: [
        { kind: 'yes', label: 'Yes', votes: 84 },
        { kind: 'not-present', label: 'Not present', votes: 0, parties: [] },
        { kind: 'no', label: 'No', votes: 51 },
      ],
    });
    expect(w.text()).not.toContain('Not present');
    expect(w.find('.chamber-vote-outcome--not-present').exists()).toBe(false);
  });
});
