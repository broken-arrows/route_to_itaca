import { describe, expect, it } from 'vitest';
import { CaptureUI } from '../src/engine/capture-ui';

describe('CaptureUI page media lifecycle', () => {
  it('keeps a face image through same-page continuations and clears it on a new page', () => {
    const ui = new CaptureUI();

    ui.newPage();
    ui.displayContent([], 'img/events/lead.jpg');
    ui.resetTransient();
    ui.displayContent([]);
    expect(ui.faceImage).toBe('img/events/lead.jpg');

    ui.newPage();
    expect(ui.faceImage).toBeNull();
  });
});
