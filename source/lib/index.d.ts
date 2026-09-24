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
  congresoResultsMap(q: Record<string, unknown>): {
    regions: Array<{ id: string; constituency: string; winner: string | null }>;
    panels: Record<string, { name: string; seats: number; rows: Array<{ party: string; seats: number; support: number }> }>;
  };
  localResultsMap(q: Record<string, unknown>): {
    winners: Record<string, string>;
    cities: Array<{ name: string; id: string; winner: string | null }>;
  };
  congresoPartyTour(q: Record<string, unknown>): Array<{
    id: string; viewed: boolean; highlighted: boolean; revealedByRest: boolean;
    parties: string[]; aragon: boolean;
  }>;
  buildParlamentCorruptionTransfers(
    Q: Record<string, unknown>,
    context: { carriers: Record<string, string>; support: Record<string, number>; responsibility?: Record<string, number> },
  ): Array<{ mechanism: string; from: string; to: string; amount: number }>;
  advanceParlamentDisappointment(Q: Record<string, unknown>): void;
  buildParlamentDisappointmentTransfers(
    Q: Record<string, unknown>,
    context: { province: string; demographic: string; carriers: Record<string, string>; support: Record<string, number>; scale?: number },
  ): Array<{ mechanism: string; from: string; to: string; amount: number; onSettled: (realized: number) => void }>;
  buildParlamentCompetitionTransfers(
    Q: Record<string, unknown>,
    context: { support?: Record<string, number>; scaleCatSpa?: number; dCatSpa?: number; activeFederalLeft?: string },
  ): Array<{ mechanism: string; from: string; to: string; amount: number }>;
  getFederalLeftLeadershipProfile(
    Q: Record<string, unknown>, activeFederalLeft: string,
  ): { channeling: number; conflict: number; retention: number };
  buildParlamentParticipationTransfers(
    Q: Record<string, unknown>,
    context: { province: string; demographic: string; carriers: Record<string, string>; support: Record<string, number> },
  ): Array<{ mechanism: string; from: string; to: string; amount: number; onSettled?: (realized: number) => void }>;
  /** Clear participation and disappointment memory when replacing the electorate. */
  resetParlamentParticipation(Q: Record<string, unknown>): void;
  /** Initialize after replacing the electorate; old saves initialize on first tick. */
  resetParlamentSignalBaseline(Q: Record<string, unknown>): void;
  applyParlamentTransfers(
    Q: Record<string, unknown>, province: string, demographic: string,
    matrixDeltas: Record<string, number> | number[],
    transfers: Array<{ mechanism: string; from: string; to: string; amount: number; onSettled?: (realized: number) => void }>,
    traceWeight?: number,
  ): Record<string, number>;
  getParlamentResponsibility(Q: Record<string, unknown>): Record<string, number>;
  buildParlamentResponsibilityTransfers(
    Q: Record<string, unknown>,
    context: { province: string; demographic: string; dWelfare: number; dUnemployment: number; dDissent: number; scale?: number },
  ): Array<{ mechanism: string; from: string; to: string; amount: number }>;
  governmentTooltip(
    institution: 'generalitat' | 'gobierno' | 'ajuntament',
    parties: string[],
    summary: string,
    labelHtml: string,
  ): string;
  engineTick(Q: Record<string, unknown>): void;
  /** Bounded, conserved support movement for recurring card effects. Structural
   *  party formation/succession code deliberately uses separate helpers. */
  cardSupportTransfer(
    Q: Record<string, unknown>,
    options:
      | {
          contest: 'parlament';
          to: string;
          from: string;
          amount: number;
          maxDonorFraction: 0.5;
          constituencies: string | string[];
          demographics: string | string[];
        }
      | {
          contest: 'congreso';
          to: string;
          from: string;
          amount: number;
          maxDonorFraction: 0.5;
          constituencies: string | string[];
        }
      | {
          contest: 'barcelona';
          to: string;
          from: string;
          amount: number;
          maxDonorFraction: 0.5;
        },
  ): number;
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
