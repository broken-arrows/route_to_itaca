// Co-located type declaration for index.js (plain JS on purpose — see the
// header comment there for why). TypeScript's module resolver picks this up
// automatically for any `import ... from '.../source/lib/index.js'`, in `ui/`
// or any future dendrynexus UI written in TypeScript; the runtime file stays
// untyped, framework-free JS so the old (bundler-less) shell can still load it
// as a plain <script>.
/** One party affiliation for a person, as returned by an allegiances function.
 *  Presentation-neutral: `colour` is a token name ("erc") or a raw hex
 *  ("#555555"); `note` (e.g. "former") is optional trailing plain text. Each UI
 *  renders this its own way — see source/lib/allegiances.js. */
export interface AllegianceEntry {
  colour: string;
  label: string;
  note?: string;
}

export interface GameLib {
  engineTick(Q: Record<string, unknown>): void;
  spaSupportInject(
    Q: Record<string, unknown>,
    family: string,
    constituency: string,
    delta: number,
    from: string,
  ): void;
  /** Q-conditional party history per glossary term id (see allegiances.js).
   *  Called by each UI's tooltip renderer. */
  allegiances: Record<string, (Q: Record<string, unknown>) => AllegianceEntry[]>;
  /** The Brief's pure row derivations — see source/lib/brief.js. Keys must
   *  match DERIVE_NAMES in ui/src/components/viz/widget-names.mjs. */
  brief: Record<string, (q: Record<string, unknown>) => unknown[]>;
  getLawsForUI(q: Record<string, unknown>): Array<{
    id: string;
    title: string;
    icon: string;
    status: 'active' | 'repealed' | 'disputed' | 'imposed' | 'struck_down';
    ticks_active: number;
    effects: Record<string, number>;
  }>;
  // NB: registerLaw / deactivateLaw are exported too but omitted here — content
  // (compiled .dry, not TS-checked) is their only caller, so a precise type
  // buys nothing. The `source/lib/` aggregation/typing shape is deferred to
  // phase 6 (see the §2.5 spec §10.1).
}
declare const gameLib: GameLib;
export default gameLib;
