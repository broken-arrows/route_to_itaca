import { createRequire } from 'node:module';
import { gameLib } from '../../src/game-bindings';

const require = createRequire(import.meta.url);
const { DendryEngine, UserInterface } = require('dendrynexus-ten/lib/engine.js');

/** Bind standalone authored-action fixtures to the real runtime text helpers.
 * The supplied object stays the proxy's backing target, so callers still see
 * their original Q fixture mutate. No root actions, navigation or RNG run. */
export function bindSceneRuntime<Q extends Record<string, any>>(
  qualities: Q,
  libraryOverrides: Record<string, unknown> = {},
) {
  const engine = new DendryEngine(new UserInterface(), {qualities:{}, data:{}, scenes:{}});
  engine.setGameLib({...gameLib, ...libraryOverrides});
  engine.state = {qualities};
  engine._setUpQualities();
  return {engine, state:engine.state, Q:engine.state.qualities as Q, G:engine.gameLib};
}
