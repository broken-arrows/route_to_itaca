import { describe, expect, it } from 'vitest';
import { gameLib } from '../src/game-bindings';

// The allegiances map is game MODEL logic (source/lib/allegiances.js): which
// parties a person belonged to, branching on Q. It was ported from HTML-string
// closures in out/html/game.js to presentation-neutral {colour,label,note?}
// data. These cases lock the port's branches — the places a silent transcription
// error would hide.
describe('gameLib.allegiances', () => {
  const A = gameLib.allegiances;

  it('is wired through the game lib as a map of functions', () => {
    expect(A && typeof A).toBe('object');
    expect(typeof A.artur_mas).toBe('function');
  });

  it('returns presentation-neutral entries — no markup, colour is a token or hex', () => {
    const [erc] = A.llu_s_companys({});
    expect(erc).toEqual({ colour: 'erc', label: 'ERC' });
    // Franco keeps a raw hex (no palette token existed), same convention as glossary.
    expect(A.francisco_franco({})[0].colour).toBe('#555555');
  });

  it('adds conditional affiliations only when the Q flag is set', () => {
    expect(A.artur_mas({}).map((e) => e.label)).toEqual(['CDC', 'CiU']);
    expect(A.artur_mas({ dl_formed: true, jxsi_formed: true }).map((e) => e.label)).toEqual([
      'CDC',
      'CiU',
      'DL',
      'JxSí',
    ]);
    // mas_ousted gates JxCat/Junts off even when formed.
    expect(A.artur_mas({ jxcat_formed: true, mas_ousted: true }).map((e) => e.label)).toEqual([
      'CDC',
      'CiU',
    ]);
  });

  it('carries the "former" note on a split-off affiliation', () => {
    const list = A.carles_puigdemont({ pdcat_formed: true, pdcat_split: true });
    expect(list.find((e) => e.label === 'PDeCAT')).toEqual({
      colour: 'pdcat',
      label: 'PDeCAT',
      note: 'former',
    });
  });

  it('branches into a different label SET, not just show/hide (romeva)', () => {
    expect(A.ra_l_romeva({}).map((e) => e.label)).toEqual(['ICV-EUiA']);
    expect(
      A.ra_l_romeva({ jxsi_formed: true, jxcat_formed: true, erc_in_jxcat: true }).map(
        (e) => e.label,
      ),
    ).toEqual(['ICV-EUiA', 'JxSí', 'JxCat', 'ERC']);
  });
});
