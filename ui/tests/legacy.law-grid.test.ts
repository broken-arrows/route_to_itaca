import { describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const widgetJs = readFileSync(resolve(import.meta.dirname, '../../out/html/widgets.js'), 'utf8');

describe('old-shell law grid', () => {
  it('uses authored colours and descriptions without lifecycle status', () => {
    document.body.innerHTML = '<main><div data-widget="law-grid"></div></main>';
    const rows = [
      { id: 'a', title: 'Alpha', icon: 'img/a.svg', colour: 'gray', description: 'Gray copy' },
      { id: 'b', title: 'Beta', icon: 'img/b.svg', colour: 'orange', description: 'Orange copy' },
      { id: 'c', title: 'Gamma', icon: 'img/c.svg', colour: 'red', description: 'Red copy' },
      { id: 'd', title: 'Delta', icon: 'img/d.svg', colour: 'green', description: 'Green copy' },
      { id: 'hidden', title: 'Hidden', colour: 'green', description: 'Invisible copy' },
    ];
    (window as any).dendryUI = { dendryEngine: { gameLib: { getLawsForUI: vi.fn(() => rows) } } };
    window.eval(widgetJs);
    (window as any).mountWidgets(document.querySelector('main'), {});

    const laws = [...document.querySelectorAll('.law-grid__law')];
    expect(laws.map((law) => law.className)).toEqual([
      'law-grid__law law-grid__law--gray',
      'law-grid__law law-grid__law--orange',
      'law-grid__law law-grid__law--red',
      'law-grid__law law-grid__law--green',
    ]);
    expect(laws.map((law) => law.querySelector('.law-grid__tooltip-description')?.textContent))
      .toEqual(['Gray copy', 'Orange copy', 'Red copy', 'Green copy']);
    expect(laws[0].getAttribute('aria-label')).toBe('Alpha — Gray copy');
  });
});
