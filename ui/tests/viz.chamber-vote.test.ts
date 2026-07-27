import { describe, expect, it } from 'vitest';
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
        { id: 'ciu', match: ['CiU'], display: 'CiU', colour: 'ciu' },
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

  it('accepts optional party breakdowns and optional split-caucus counts', () => {
    const w = mountVote();
    expect(w.findAll('.chamber-vote-parties li')).toHaveLength(2);
    expect(w.text()).toContain('PSC (3)');
    expect(w.find('.chamber-vote-breakdown--no .chamber-vote-parties').exists()).toBe(false);
  });

  it('marks generated party names through the Desk glossary pipeline', () => {
    const w = mountVote();
    const ciu = w.get('[data-term="ciu"]');
    expect(ciu.text()).toBe('CiU');
    expect(ciu.attributes('style')).toContain('var(--ciu)');
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
});
