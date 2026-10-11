import { describe, expect, it } from 'vitest';
import { CaptureUI } from '../src/engine/capture-ui';

describe('CaptureUI page media lifecycle', () => {
  it('associates an image with the scene text appended after earlier page prose', () => {
    const ui = new CaptureUI();
    ui.displayContent(['Earlier setup', 'Difficulty']);
    ui.displayContent(['Party description'], 'img/party.png');
    ui.displayContent(['Continuation']);
    expect(ui.faceImageOffset).toBe(2);
    ui.newPage();
    ui.displayContent(['Earlier setup', 'Difficulty', 'Party description'], 'img/party.png', 2);
    expect(ui.faceImageOffset).toBe(2);
    ui.newPage();
    expect(ui.faceImageOffset).toBeNull();
  });

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
