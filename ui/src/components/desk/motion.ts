// Choreography constants + the sync-when-off timer helper the desk store
// builds every animated transition on top of. Values are ported verbatim
// from the approved Draw-to-Dossier prototype (see
// docs/design/reference/prototype-draw-to-dossier-NOTES.md) — do not
// retune without re-checking that spec.
export const DELAYS = {
  draw: 880,
  dossierIn: 40,
  // The dossier's open beats (NOTES motion sequence #2): scale-up, cover
  // swing, and the swing's start delay. Purely visual — no store timer
  // consumes them; they live here so OpenDossier's CSS durations route
  // through the same animMs() single source as every other desk duration.
  dossierOpen: 550,
  coverSwing: 750,
  coverSwingDelay: 280,
  resolve: 620,
  cancel: 420,
  toast: 1700,
  // Achievement unlock toast dwell time (phase 2.5 Task 8) — NOT part of the
  // Draw-to-Dossier prototype `toast` above (that one is a short key-based
  // nudge). Ported from the old shell's own achievementNotif dwell
  // (out/html/game.js's `}, 4500);`) rather than invented, so an unlock
  // reads for the same real-world duration in both UIs. Like `toast`, this
  // is information delivery, not motion: the desk store does not scale it
  // through animMs().
  achievementToast: 4500,
} as const;

// Runs `fn` after `ms` milliseconds, or synchronously right away when
// `ms <= 0` (the "animations off" case: `motion.ts` itself has no opinion on
// WHY ms is 0 — the desk store's `animMs()` decides that — this just makes
// ms<=0 a real synchronous commit instead of a same-tick-but-still-async
// `setTimeout(fn, 0)`, which is what lets tests skip fake timers entirely).
// Returns the scheduled timer handle when ms > 0, so callers whose effect
// can be RE-TRIGGERED (toast dismiss, shake reset) can clearTimeout the
// stale one; returns undefined when `fn` already ran synchronously.
export function after(ms: number, fn: () => void): ReturnType<typeof setTimeout> | undefined {
  if (ms <= 0) {
    fn();
    return undefined;
  }
  return setTimeout(fn, ms);
}
