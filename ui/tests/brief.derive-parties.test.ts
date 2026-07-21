import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DendryAdapter } from '../src/engine/adapter';
import { gameLib } from '../src/game-bindings';

const GAME = join(__dirname, '..', '..', 'out', 'game.json');

describe('G.brief party derivations', () => {
  let Q: Record<string, unknown>;
  beforeAll(() => {
    const a = DendryAdapter.fromJSONText(readFileSync(GAME, 'utf8'));
    a.beginGame([1, 2, 3, 4]);
    Q = a.qualities;
  });

  it('exposes brief on the game lib', () => {
    expect(typeof (gameLib as any).brief).toBe('object');
  });

  it('benches lists only parties holding seats, ordered by seats desc', () => {
    const rows = (gameLib as any).brief.benches(Q);
    expect(rows.length).toBeGreaterThan(0);
    for (const r of rows) expect(r.value).toBeGreaterThan(0);
    const seats = rows.map((r: any) => r.value);
    expect([...seats].sort((a, b) => b - a)).toEqual(seats);
  });

  it('benches rows carry an id usable as a glossary key, and no HTML', () => {
    const rows = (gameLib as any).brief.benches(Q);
    expect(rows.length).toBeGreaterThan(0);
    for (const r of rows) {
      expect(r.id).toMatch(/^[a-z_]+$/);
      expect(r.label).not.toMatch(/</);
      expect(r.subtitle).not.toMatch(/</);
    }
  });

  // Strengthened vs. the brief's sample: at a fresh beginGame() the real
  // fixture's Q.player_party is `undefined` (it's only ever set by the
  // party-selection branches in root.scene.dry, e.g. `Q.player_party =
  // "erc"` at :1369 — none of which run on a bare beginGame). So
  // `rows.filter(r => r.isPlayer)` is `[]` on the untouched fixture and
  // `mine.length <= 1` / the `if (mine.length)` guard both pass vacuously
  // REGARDLESS of whether isPlayer is implemented at all. Force the
  // scenario the test claims to check: a real seated party (erc, 10 seats
  // per root.scene.dry:141) set as the player.
  it('marks exactly one row as the player when the player holds seats', () => {
    const rows = (gameLib as any).brief.benches(Q);
    const mineOnFixture = rows.filter((r: any) => r.isPlayer);
    expect(mineOnFixture.length).toBeLessThanOrEqual(1);

    const Qerc = Object.assign({}, Q, { player_party: 'erc' });
    const rowsErc = (gameLib as any).brief.benches(Qerc);
    const mine = rowsErc.filter((r: any) => r.isPlayer);
    expect(mine.length).toBe(1);
    expect(mine[0].id).toBe('erc');
    // The player has no "relation with themselves" — design stamps YOU
    // instead, so the row must carry no stamp at all.
    expect(mine[0].stamp).toBeNull();
  });

  // The finding this replaces: `stamp` used to be `String(Q[id +
  // '_relations'])`, a pre-stringified NUMBER wearing the name of a band
  // word ('warm'/'hostile'/...). brief.js can't classify it (it only ever
  // gets `Q`, never the engine), so the row must instead carry the raw
  // value plus the id of the qdisplay that classifies it — a generic
  // renderer bands it via adapter.qdisplay(value, qdisplayId), keeping
  // source/qdisplays/*.dry the only place a threshold is written.
  it('non-player rows carry the raw relation value and the qdisplay id, not a band', () => {
    // Force the scenario a bare beginGame() never reaches (see LEARNINGS):
    // player_party AND every party's _relations are only ever assigned
    // inside root.scene.dry's three party-choice branches.
    const Qerc = Object.assign({}, Q, {
      player_party: 'erc',
      cs_relations: 3,
      ciu_relations: 55,
    });
    const rows = (gameLib as any).brief.benches(Qerc);
    const cs = rows.find((r: any) => r.id === 'cs');
    const ciu = rows.find((r: any) => r.id === 'ciu');
    expect(cs).toBeDefined();
    expect(ciu).toBeDefined();
    // A NUMBER, not the old `String(...)` — this is what the old
    // implementation got wrong (it coerced to a string band-word shape).
    expect(cs.stamp).toBe(3);
    expect(typeof cs.stamp).toBe('number');
    expect(cs.stampDisplay).toBe('relationships');
    expect(ciu.stamp).toBe(55);
    expect(ciu.stampDisplay).toBe('relationships');

    // A non-player row whose `_relations` is genuinely undefined (the real
    // fixture state for every seated party except erc here) must carry
    // `null`, never the old `''` fallback — that undocumented third state
    // is exactly what the reviewer flagged.
    const psc = rows.find((r: any) => r.id === 'psc');
    expect(psc).toBeDefined();
    expect(psc.stamp).toBeNull();
    expect(psc.stampDisplay).toBe('relationships');
  });

  // Proves the round trip actually classifies, through the real engine and
  // the real source/qdisplays/relationships.qdisplay.dry bands — not a
  // hand-guessed threshold. Bands per that file: (..5) hostile, (5..14.9)
  // frigid, (14.9..29.9) cold, (29.9..39.9) cool, (39.9..54.9) neutral,
  // (54.9..64.9) warm, (64.9..74.9) friendly, (74.9..) very friendly.
  it('adapter.qdisplay classifies a raw relation value through the real qdisplay', () => {
    const a = DendryAdapter.fromJSONText(readFileSync(GAME, 'utf8'));
    a.beginGame([1, 2, 3, 4]);
    expect(a.qdisplay(3, 'relationships')).toBe('hostile');
    expect(a.qdisplay(70, 'relationships')).toBe('friendly');
  });

  // Strengthened vs. the brief's sample: the loop body over `rows` never
  // executes if `composition(Q)` returns `[]` (e.g. an unimplemented stub),
  // so `share`/`majority` would never be checked and the test would pass
  // green with zero derivation logic behind it. Assert non-empty first.
  it('composition share is 0..1 and majority is floor(size/2)+1', () => {
    const rows = (gameLib as any).brief.composition(Q);
    expect(rows.length).toBeGreaterThan(0);
    for (const r of rows) {
      expect(r.share).toBeGreaterThan(0);
      expect(r.share).toBeLessThanOrEqual(1);
      expect(r.majority).toBe(Math.floor((Q.parlament_size as number) / 2) + 1);
    }
  });

  it('writes nothing to Q', () => {
    const before = JSON.stringify(Q);
    (gameLib as any).brief.benches(Q);
    (gameLib as any).brief.composition(Q);
    expect(JSON.stringify(Q)).toBe(before);
  });
});
