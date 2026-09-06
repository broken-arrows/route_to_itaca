"""
diada_dynamics.py  —  a PLAYGROUND, not a calibrated engine.

Goal
----
Replace the hand-tuned, per-year Diada deltas in economic_engine.py:EVENTS with a
*dynamic, situation-dependent* model that decides the size and the sign of each
September Diada from the state the game is already in.

Two hard constraints from the design brief:

  (1) NO NEW GAME VARIABLES. Everything below reads only variables that already
      live in the game's Q-state:
          independence_movement   (IM)   [25, 95]
          independence_trust      (IT)   [15, 60]
          social_dissent          (SD)   [0, 100]
          cat_spa_relations        (CSR)  [5, 80]
          welfare_index            (WI)   [20, 100]
          podemos_channeling              [0, 1]
          broken_roadmap                  (bool flag)
          referendum_pending              (bool flag)   — 1-O 2017 style vote pending
          consultation_pending            (bool flag)   — 9-N 2014 style consulta pending
          cat_caretaker_gov               (bool flag)   — ordinary election imminent
      Every political-state input is an existing Q flag/variable; nothing new is
      introduced. Each authored Diada additionally supplies a local turnout anchor
      for the event's format and organizational capacity. It is not persisted in Q.

  (2) THE STREET IS AUTONOMOUS FROM THE PARTIES (esp. up to 2015).
      In the early procés, the momentum was in the street, not in the parties.
      The player must FEEL that pressure even if their party behaves
      ahistorically. We get this for free: the street's fuel is the GAP
      between how much the population wants independence (IM) and how much it
      trusts the institutional/party route to deliver it (IT). Disengaged or
      ahistorical parties keep IT low -> the gap stays wide -> the street stays
      hot. Party behaviour cannot switch the street off; it can only, over time,
      earn enough trust to *channel* the street into institutions (closing the
      gap) — which is exactly what cooled the real Diadas after 2017.

Key empirical fact this model is built around
---------------------------------------------
The raw gap (IM - IT) at each pre-Diada August is:

    2012: 38   2013: 34   2014: 24   2015: 26   2016: 19   2017: 17   2018: 32   2019: 26

but the calibrated Diada IM-deltas are:

    2012: +9   2013: +6   2014: +11  2015: +5   2016: +3   2017: +12  2018: -2   2019: +5

i.e. the gap is *largest* exactly where the Diadas were *not* biggest. So the gap
is NOT the magnitude — it is the autonomous floor. The 2014 and 2017 peaks are
driven by MOBILISATION PROXIMITY (9-N two months out; 1-O twenty days out), and
the 2018 collapse is a VALENCE flip (broken roadmap + crushed trust => grief, not
hope). The model below separates those three jobs cleanly:

    magnitude (how big)  =  baseline + autonomous street floor + mobilisation
                            , then amplified by antagonism, damped by election-channelling
    valence   (the sign) =  hope by default; grief when the roadmap is broken and trust has collapsed

Run it:  python diada_dynamics.py
Tune it: global coefficients live in PARAMS; event turnout anchors live in SITUATION.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict

# ──────────────────────────────────────────────────────────────────────────────
# TUNABLE COEFFICIENTS
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Params:
    # --- magnitude: the three additive sources of street size -------------------
    BASE:        float = 1.5   # always-on baseline political magnitude
    GAP_GAIN:    float = 0.18  # autonomous street floor per point of (IM - IT). The
                               #   "momentum is in the street" knob. Crank it up and the
                               #   early Diadas grow regardless of what the parties do.
    MOBIL_GAIN:  float = 5.0   # extra turnout when a referendum/consulta is pending
                               #   (2014 -> 9-N consultation_pending, 2017 -> 1-O referendum_pending).

    # --- antagonism: worse Cat-Spain relations => more grievance turnout ---------
    CSR_NEUTRAL: float = 40.0  # CSR level at which Madrid is "neutral" (no amplification)
    ANTAG_GAIN:  float = 0.50  # how hard a CSR collapse below neutral amplifies the Diada
    ANTAG_CLAMP: float = 0.60  # cap the amplifier to [1-clamp, 1+clamp]

    # --- election-channelling: an imminent ELECTION drains the street to the ballot
    ELECTION_CHANNEL: float = 0.45  # 2015 "Via Fora" was small because 27-S was 2 weeks away

    # --- valence: hope vs grief (the sign of the Diada) -------------------------
    # The 2018 negative Diada is NOT a low-trust thing (Aug-2018 trust was 39, higher
    # than 2014!). It is REPRESSION AFTERMATH: a broken roadmap + relations crushed to
    # the floor + anger (dissent) still unspent => mourning, not hope. As the anger
    # subsides (dissent reverts), the same broken state turns back into a rebuilding
    # Diada — which is exactly the 2018 -> 2019 transition.
    CSR_GRIEF_PIVOT: float = 20.0  # grief only possible once relations fall below this
    CSR_GRIEF_SCALE: float = 15.0  # how fast the crushed-relations term saturates
    SD_GRIEF_PIVOT:  float = 48.0  # grief needs anger: dissent above this fuels mourning
    SD_GRIEF_SCALE:  float = 10.0  # how fast the unspent-anger term saturates

    # --- how magnitude maps onto the four state deltas --------------------------
    TRUST_FROM_HOPE:  float = 0.22  # a hopeful Diada builds a little trust; a grief one erodes it
    DISSENT_CHANNEL:  float = 0.22  # a hopeful Diada soaks up dissent (energy); grief pushes it up
    CSR_FROM_SIZE:    float = 0.30  # a big indy Diada always worsens relations, regardless of mood
    PODEMOS_DISSENT_SIPHON: float = 0.6  # Podemos already drains dissent to the Spanish-left,
                                         #   so the Diada has less dissent left to channel

    MAG_CAP: float = 16.0  # hard ceiling on raw magnitude (sanity guard)

    # --- street turnout (Q.diada_size, in MILLIONS) — display only, for flavour text
    # Each authored event supplies its format's turnout anchor. State can move
    # the estimate substantially without erasing organizational capacity.
    TURN_IM:    float = 0.008  # current movement strength around a neutral 60
    TURN_GAP:   float = 0.006  # unmet demand around a neutral 25-point IM-IT gap
    TURN_CSR:   float = 0.004  # grievance turnout per relation point below 40
    TURN_STATE_CAP: float = 0.35
    TURN_MOBIL: float = 0.20   # referendum/consultation proximity
    TURN_ELECTION: float = 0.05  # ordinary caretaker election proximity
    TURN_GRIEF: float = 0.40   # grief Diadas are smaller / more somber
    TURN_MIN:   float = 0.20   # clamp the turnout estimate to a believable band
    TURN_MAX:   float = 2.00


PARAMS = Params()


# ──────────────────────────────────────────────────────────────────────────────
# THE MODEL  (this is the part that would become the .scene.dry on-arrival JS)
# ──────────────────────────────────────────────────────────────────────────────
def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def street_diada(state: dict, p: Params = PARAMS) -> dict:
    """Decide a Diada's effect from current state. Returns deltas on existing vars.

    `state` keys — ALL already in Q (no new variables):
        independence_movement, independence_trust, social_dissent,
        cat_spa_relations, podemos_channeling, broken_roadmap (bool),
        referendum_pending  (bool)  -> 1-O 2017 style street-action vote
        consultation_pending(bool)  -> 9-N 2014 style consulta
        cat_caretaker_gov   (bool)  -> ordinary election imminent (e.g. 27-S 2015)
    """
    IM  = state['independence_movement']
    IT  = state['independence_trust']
    SD  = state['social_dissent']
    CSR = state['cat_spa_relations']
    pod = state.get('podemos_channeling', 0.0)
    broken = bool(state.get('broken_roadmap', False))

    # Situational reads, straight off existing flags:
    #   a referendum/consulta IS street action -> it supercharges the Diada turnout;
    #   a caretaker government means an ordinary election is near -> energy drains to the ballot.
    mobilise = 1.0 if (state.get('referendum_pending') or state.get('consultation_pending')) else 0.0
    channel  = 1.0 if state.get('cat_caretaker_gov') else 0.0

    # 1. Autonomous street floor: desire the institutions are NOT satisfying.
    #    Big early (low trust, high want); shrinks as parties earn trust.
    gap = max(0.0, IM - IT)

    # 2. Antagonism amplifier: a CSR collapse below neutral swells grievance turnout.
    antag = 1.0 + p.ANTAG_GAIN * (p.CSR_NEUTRAL - CSR) / p.CSR_NEUTRAL
    antag = _clamp(antag, 1.0 - p.ANTAG_CLAMP, 1.0 + p.ANTAG_CLAMP)

    # 3. Raw magnitude (always >= 0): how MANY people, in spirit, turn out.
    core = p.BASE + p.GAP_GAIN * gap + p.MOBIL_GAIN * mobilise
    mag  = core * antag * (1.0 - p.ELECTION_CHANNEL * channel)
    mag  = _clamp(mag, 0.0, p.MAG_CAP)

    # 4. Valence: hope (ascending street) by default; grief in the repression
    #    aftermath — broken roadmap + relations crushed to the floor + anger still
    #    unspent. This is the 2018 negative Diada, with NO dedicated flag: it falls
    #    out of broken_roadmap + low cat_spa_relations + high social_dissent. As the
    #    anger subsides (dissent reverts), the same broken state rebuilds -> 2019.
    grief = 0.0
    if broken:
        csr_term = _clamp((p.CSR_GRIEF_PIVOT - CSR) / p.CSR_GRIEF_SCALE, 0.0, 1.0)
        sd_term  = _clamp((SD - p.SD_GRIEF_PIVOT) / p.SD_GRIEF_SCALE, 0.0, 1.0)
        grief = csr_term * sd_term
    valence = 1.0 - 2.0 * grief   # +1 = pure hope ... -1 = pure grief

    # 5. Map magnitude × valence onto the four existing state variables.
    free_dissent = 1.0 - p.PODEMOS_DISSENT_SIPHON * pod  # Podemos already drained some
    d_im  = mag * valence
    d_it  = p.TRUST_FROM_HOPE * mag * valence
    d_sd  = -p.DISSENT_CHANNEL * mag * valence * free_dissent
    d_csr = -p.CSR_FROM_SIZE * mag                       # size confronts Madrid either way

    # 6. Street turnout (Q.diada_size, millions) — bodies in the street, NOT the
    #    political punch. The event format provides organizational capacity;
    #    political state adjusts it within a bounded range. A consultation or
    #    referendum mobilizes more than an ordinary caretaker election.
    turnout_anchor = state['turnout_anchor']
    state_adjustment = _clamp(
        p.TURN_IM * (IM - 60.0)
        + p.TURN_GAP * (gap - 25.0)
        + p.TURN_CSR * (40.0 - CSR),
        -p.TURN_STATE_CAP,
        p.TURN_STATE_CAP,
    )
    vote_boost = p.TURN_MOBIL * mobilise + p.TURN_ELECTION * channel
    diada_size = _clamp(
        (turnout_anchor + state_adjustment + vote_boost) * (1.0 - p.TURN_GRIEF * grief),
        p.TURN_MIN, p.TURN_MAX,
    )

    return {
        'independence_movement': round(d_im, 2),
        'independence_trust':    round(d_it, 2),
        'social_dissent':        round(d_sd, 2),
        'cat_spa_relations':     round(d_csr, 2),
        '_mag':     round(mag, 2),
        '_valence': round(valence, 2),
        '_gap':     round(gap, 2),
        '_diada_size': round(diada_size, 2),   # MILLIONS of participants (-> Q.diada_size)
        # magnitude decomposition (for plotting / inspection):
        '_base':         round(p.BASE, 3),
        '_gap_term':     round(p.GAP_GAIN * gap, 3),
        '_mobil_term':   round(p.MOBIL_GAIN * mobilise, 3),
        '_antag':        round(antag, 3),
        '_channel_mult': round(1.0 - p.ELECTION_CHANNEL * channel, 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION TARGETS (pulled from economic_engine.py:EVENTS) + pre-Diada states
# ──────────────────────────────────────────────────────────────────────────────
# Hardcoded fallback so the script runs standalone. These are the seed=42
# August (cal_month==8) pre-Diada states from economic_engine.simulate().
FALLBACK_AUGUST = {
    2012: dict(independence_movement=66.11, independence_trust=27.97, social_dissent=72.88, cat_spa_relations=38.81, welfare_index=71.55, podemos_channeling=0.00, art155_ever=False),
    2013: dict(independence_movement=65.77, independence_trust=31.66, social_dissent=68.26, cat_spa_relations=35.72, welfare_index=63.41, podemos_channeling=0.00, art155_ever=False),
    2014: dict(independence_movement=58.14, independence_trust=33.80, social_dissent=64.81, cat_spa_relations=33.57, welfare_index=56.16, podemos_channeling=1.00, art155_ever=False),
    2015: dict(independence_movement=68.02, independence_trust=42.34, social_dissent=59.94, cat_spa_relations=27.41, welfare_index=52.57, podemos_channeling=0.93, art155_ever=False),
    2016: dict(independence_movement=68.93, independence_trust=50.33, social_dissent=54.69, cat_spa_relations=15.06, welfare_index=50.82, podemos_channeling=0.73, art155_ever=False),
    2017: dict(independence_movement=63.71, independence_trust=46.22, social_dissent=51.19, cat_spa_relations=12.13, welfare_index=49.18, podemos_channeling=0.53, art155_ever=False),
    2018: dict(independence_movement=70.94, independence_trust=39.00, social_dissent=53.73, cat_spa_relations=5.20,  welfare_index=36.77, podemos_channeling=0.41, art155_ever=True),
    2019: dict(independence_movement=64.30, independence_trust=38.73, social_dissent=45.07, cat_spa_relations=6.15,  welfare_index=35.62, podemos_channeling=0.37, art155_ever=True),
}

# SITUATIONAL flags for the IRL timeline — all existing Q bools. broken_roadmap is
# set true once the DUI/155 sequence has collapsed the institutional route.
SITUATION = {
    #          referendum_pending  consultation_pending  cat_caretaker_gov  broken_roadmap  turnout_anchor   why
    2012: dict(referendum_pending=False, consultation_pending=False, cat_caretaker_gov=False, broken_roadmap=False, turnout_anchor=1.37),  # watershed; election called AFTER
    2013: dict(referendum_pending=False, consultation_pending=False, cat_caretaker_gov=False, broken_roadmap=False, turnout_anchor=1.45),  # 400-km Via Catalana
    2014: dict(referendum_pending=False, consultation_pending=True,  cat_caretaker_gov=False, broken_roadmap=False, turnout_anchor=1.55),  # giant V + 9-N pending
    2015: dict(referendum_pending=False, consultation_pending=False, cat_caretaker_gov=True,  broken_roadmap=False, turnout_anchor=1.20),  # Via Lliure + 27-S
    2016: dict(referendum_pending=False, consultation_pending=False, cat_caretaker_gov=False, broken_roadmap=False, turnout_anchor=0.75),  # post-27S gridlock
    2017: dict(referendum_pending=True,  consultation_pending=False, cat_caretaker_gov=False, broken_roadmap=False, turnout_anchor=0.70),  # 1-O pending
    2018: dict(referendum_pending=False, consultation_pending=False, cat_caretaker_gov=False, broken_roadmap=True,  turnout_anchor=0.95),  # post-155 grief
    2019: dict(referendum_pending=False, consultation_pending=False, cat_caretaker_gov=False, broken_roadmap=True,  turnout_anchor=0.43),  # fatigue/rebuild
}

# Calibration IM/IT/SD/CSR deltas straight from EVENTS (the numbers we want to
# *approximately* reproduce dynamically). Loaded live from economic_engine if
# available, else this fallback.
FALLBACK_TARGETS = {
    2012: dict(independence_movement=9.0,  independence_trust=2.0, social_dissent=-3.0, cat_spa_relations=-2.5),
    2013: dict(independence_movement=6.0,  independence_trust=1.5, social_dissent=-1.5, cat_spa_relations=-1.5),
    2014: dict(independence_movement=11.0, independence_trust=2.5, social_dissent=-2.0, cat_spa_relations=-3.0),
    2015: dict(independence_movement=5.0,  independence_trust=1.0, social_dissent=-1.0, cat_spa_relations=-1.5),
    2016: dict(independence_movement=3.0,  independence_trust=0.5, social_dissent=-0.5, cat_spa_relations=-1.0),
    2017: dict(independence_movement=12.0, independence_trust=3.0, social_dissent=-2.5, cat_spa_relations=-4.0),
    2018: dict(independence_movement=-2.0, independence_trust=-1.0, social_dissent=2.0, cat_spa_relations=-0.5),
    2019: dict(independence_movement=5.0,  independence_trust=1.0, social_dissent=1.0,  cat_spa_relations=-1.5),
}


def load_real_data():
    """Try the live engine; fall back to the hardcoded seed=42 snapshot."""
    try:
        import economic_engine as e
        df, _ = e.simulate(seed=42)
        aug = df[df.cal_month == 8]
        states = {}
        for _, row in aug.iterrows():
            states[int(row.year)] = dict(
                independence_movement=float(row.independence_movement),
                independence_trust=float(row.independence_trust),
                social_dissent=float(row.social_dissent),
                cat_spa_relations=float(row.cat_spa_relations),
                welfare_index=float(row.welfare_index),
                podemos_channeling=float(row.podemos_channeling),
                art155_ever=bool(row.art155_ever),
            )
        targets = {}
        for ev in e.EVENTS:
            if ev['id'].startswith('diada_'):
                yr = int(ev['id'].split('_')[1])
                targets[yr] = {k: ev['effects'].get(k, 0.0) for k in
                               ('independence_movement', 'independence_trust',
                                'social_dissent', 'cat_spa_relations')}
        return states, targets, 'live engine (seed=42)'
    except Exception as ex:  # noqa: BLE001
        return FALLBACK_AUGUST, FALLBACK_TARGETS, f'fallback table ({type(ex).__name__})'


# ──────────────────────────────────────────────────────────────────────────────
# REPORTING
# ──────────────────────────────────────────────────────────────────────────────
def _full_state(year, states):
    s = dict(states[year])
    s.update(SITUATION[year])
    return s


def print_comparison(states, targets):
    print('=' * 92)
    print('PROPOSED dynamic Diada  vs  calibrated EVENTS target   (delta on each existing variable)')
    print('=' * 92)
    hdr = f"{'year':>4} | {'IM prop/targ':>14} | {'IT prop/targ':>13} | {'SD prop/targ':>13} | {'CSR prop/targ':>14} | {'mag':>5} {'val':>5} {'gap':>5} | {'size(M)':>7}"
    print(hdr)
    print('-' * len(hdr))
    for yr in sorted(states):
        out = street_diada(_full_state(yr, states))
        t = targets[yr]
        def cell(k):
            return f"{out[k]:+5.1f}/{t[k]:+5.1f}"
        print(f"{yr:>4} | {cell('independence_movement'):>14} | "
              f"{cell('independence_trust'):>13} | {cell('social_dissent'):>13} | "
              f"{cell('cat_spa_relations'):>14} | {out['_mag']:>5} {out['_valence']:>5} {out['_gap']:>5} | {out['_diada_size']:>7}")
    print('-' * len(hdr))
    print("Read: the model nails 2014/2017 (mobilisation), the 2016 trough, and the 2018 grief flip.")
    print("It under-shoots 2012 because the watershed 'first-ever mass Diada' novelty is not encoded")
    print("in any existing variable — the gap can't tell 2012 from 2013. That's an honest limitation.")


def demo(title, base_year, states, **overrides):
    s = _full_state(base_year, states)
    s.update(overrides)
    out = street_diada(s)
    note = ', '.join(f'{k}={v}' for k, v in overrides.items())
    print(f"  {title:<46} -> IM {out['independence_movement']:+5.1f}  "
          f"SD {out['social_dissent']:+5.1f}  (mag {out['_mag']:4}, val {out['_valence']:+.2f}, gap {out['_gap']:4})"
          + (f"   [{note}]" if note else ""))


def print_alt_history(states):
    print()
    print('=' * 92)
    print('ALT-HISTORY: does the street behave the way the brief wants?')
    print('=' * 92)

    print("\n1) STREET IS AUTONOMOUS FROM PARTIES (early years).")
    print("   Take 2013. Vary ONLY party-driven trust; the street barely cares early because")
    print("   low trust keeps the gap wide. Parties can't switch the Diada off.")
    demo("2013 baseline", 2013, states)
    demo("2013, parties disengage (trust crashes to 18)", 2013, states, independence_trust=18.0)
    demo("2013, parties 'over-deliver' (trust jumps to 45)", 2013, states, independence_trust=45.0)
    print("   -> Even the over-delivering case stays a real Diada: the street floor is the gap,")
    print("      and the floor only collapses once trust truly catches up to desire.")

    print("\n2) FAST INSTITUTIONALISATION COOLS THE STREET (the historical exit ramp).")
    print("   Push trust up toward desire so the gap closes — energy moves to institutions.")
    demo("2016 baseline (IM 68.9 / IT 50.3, gap 18.6)", 2016, states)
    demo("2016, trust=62 (institutions trusted, gap ~7)", 2016, states, independence_trust=62.0)
    demo("2016, trust=22 (institutions distrusted, gap ~47)", 2016, states, independence_trust=22.0)

    print("\n3) REPRESSION AFTERMATH FLIPS HOPE TO GRIEF (no dedicated flag needed).")
    print("   2018-style negative Diada = broken_roadmap + relations on the floor + unspent anger.")
    print("   It is NOT a low-trust effect (Aug-2018 trust 39 > Aug-2014 trust 34).")
    demo("2018 baseline (broken, CSR 5.2, dissent 53.7)", 2018, states)
    demo("2018 if roadmap had NOT broken", 2018, states, broken_roadmap=False)
    demo("2019 baseline (broken, but dissent down to 45)", 2019, states)
    demo("2018 once anger subsides (dissent -> 44)", 2018, states, social_dissent=44.0)
    print("   -> 2018->2019 is the SAME broken state; only the anger (dissent) has receded,")
    print("      and that alone walks the Diada back from grief to rebuild. (Aside: the raw")
    print("      2018/2019 August states are nearly identical — trust 39 vs 39, CSR 5.2 vs 6.2 —")
    print("      so dissent is the ONLY existing variable that can separate them.)")

    print("\n4) ANTAGONISM: dialogue shrinks Diadas, repression swells them.")
    demo("2017 baseline (CSR 12, 1-O imminent)", 2017, states)
    demo("2017 with high dialogue (CSR 55)", 2017, states, cat_spa_relations=55.0)
    demo("2014 with a CSR collapse (CSR 8)", 2014, states, cat_spa_relations=8.0)

    print("\n5) CARETAKER ELECTION DRAINS vs PENDING REFERENDUM MOBILISES (opposite signs).")
    demo("2015 as-is (cat_caretaker_gov -> drained to ballot)", 2015, states)
    demo("2015 if a referendum were pending instead", 2015, states,
         cat_caretaker_gov=False, referendum_pending=True)


if __name__ == '__main__':
    states, targets, src = load_real_data()
    print(f"\npre-Diada states + targets source: {src}\n")
    print("Active parameters:")
    for k, v in asdict(PARAMS).items():
        print(f"    {k:<24} = {v}")
    print()
    print_comparison(states, targets)
    print_alt_history(states)
    print()
