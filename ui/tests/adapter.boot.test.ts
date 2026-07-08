import { describe, it, expect } from 'vitest';
import { DendryAdapter } from '../src/engine/adapter';
import { miniGameText } from './fixtures/mini-game';

describe('DendryAdapter boot', () => {
  it('parses a compiled game and begins at root', () => {
    const adapter = DendryAdapter.fromJSONText(miniGameText);
    const frame = adapter.beginGame();
    expect(frame.sceneId).toBe('root');
    expect(frame.html).toContain('Welcome to the mini game.');
    expect(frame.isHand).toBe(false);
    expect(frame.choices).toEqual([
      expect.objectContaining({ id: 'desk', title: 'The Desk', canChoose: true }),
    ]);
    expect(frame.gameOver).toBe(false);
  });

  it('runs onArrival code (Q is live)', () => {
    const adapter = DendryAdapter.fromJSONText(miniGameText);
    adapter.beginGame();
    expect(adapter.qualities.gold).toBe(2);
    expect(adapter.qualities.player_party).toBe('erc');
  });

  it('throws on invalid JSON', () => {
    expect(() => DendryAdapter.fromJSONText('{nope')).toThrow();
  });
});
