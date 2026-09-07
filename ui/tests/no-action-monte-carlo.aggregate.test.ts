import { describe, expect, it } from 'vitest';
import {
  aggregateNoActionRuns,
  formatNoActionAggregate,
  type NoActionRunResult,
} from './support/no-action-monte-carlo';

function fixture(): NoActionRunResult {
  return {
    seed: 1,
    difficulty: 'normal',
    majorityRetained: true,
    seatDelta: 0,
    monthsAdvanced: 34,
    forcedEventChoices: 0,
    historicalStructuralEvents: ['plebiscite_election', 'jxsi_formation_offered'],
    structuralState: {
      unioSplit: true,
      dlFormed: false,
      jxsiFormed: true,
      cupInJxsi: false,
      consultationPending: false,
      consultationHappened: true,
      referendumPending: false,
      referendumHappened: false,
    },
    structuralValues: {
      ciuRoadmap: 2,
      cdcRoadmap: 2,
      dlRoadmap: 0,
      cdcRelations: 55,
      dlRelations: 0,
    },
    targetElectionKeys: ['2015-9'],
    baseline: {
      year: 2012,
      month: 11,
      broadSovereigntySeats: 70,
      seats: {},
      support: { ciu: 25, jxsi: 0, jxcat: 0 },
      validVoteShare: { ciu: 35, jxsi: 0, jxcat: 0 },
      families: {
        sovereignty: 45,
        ciu_successors_outside_joint_lists: 25,
        erc_outside_joint_lists: 15,
        cup_outside_joint_lists: 5,
        sovereignty_joint_lists: 0,
      },
      familySeats: {},
      familyValidVoteShare: {},
      totalSeats: 135,
      totalValidVoteShare: 100,
    },
    final: {
      year: 2015,
      month: 9,
      broadSovereigntySeats: 70,
      seats: {},
      support: { ciu: 0, jxsi: 40, jxcat: 0 },
      validVoteShare: { ciu: 0, jxsi: 50, jxcat: 0 },
      families: {
        sovereignty: 45,
        ciu_successors_outside_joint_lists: 0,
        erc_outside_joint_lists: 0,
        cup_outside_joint_lists: 5,
        sovereignty_joint_lists: 40,
      },
      familySeats: {},
      familyValidVoteShare: {},
      totalSeats: 135,
      totalValidVoteShare: 100,
    },
  };
}

describe('no-action Monte Carlo aggregation', () => {
  it('keeps joint lists visible instead of claiming continuous ICR/ERC deltas', () => {
    const aggregate = aggregateNoActionRuns([fixture()]);
    expect(aggregate.meanFamilyDelta).not.toHaveProperty('icr');
    expect(aggregate.meanFamilyDelta).not.toHaveProperty('erc');
    expect(aggregate.meanFamilyDelta.sovereignty).toBe(0);
    expect(aggregate.meanFamilyDelta.sovereignty_joint_lists).toBe(40);
    expect(aggregate.meanFamilyDelta.ciu_successors_outside_joint_lists).toBe(-25);
    expect(aggregate.meanValidVoteShareDelta.jxsi).toBe(50);

    const output = formatNoActionAggregate(aggregate);
    expect(output).toContain('jxsi=+40.000');
    expect(output).toContain('sovereignty_joint_lists=+40.000');
    expect(output).toContain('Mean party valid-vote-share deltas (pp)');
    expect(output).not.toMatch(/\bicr=/);
    expect(output.split('\n').at(-1)).not.toMatch(/\berc=/);
  });
});
