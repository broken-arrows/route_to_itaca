/**
 * Capture the surface that Library should preserve behind its live Brief.
 *
 * The source surface already owns a Brief. Library replaces that left-hand
 * chrome with its own live Brief, so retaining the old one would create two
 * independently laid-out tab rails in the frozen presentation.
 */
export function captureLibraryUnderlay(surface: HTMLElement | null): string {
  if (!surface) return '';

  const snapshot = surface.cloneNode(true) as HTMLElement;
  snapshot.querySelectorAll('.clipboard-frame').forEach((brief) => {
    // Keep the grid cell: removing the Brief moves the Desk into its column,
    // where the live Library sheet covers it completely.
    const placeholder = document.createElement('div');
    placeholder.className = 'library-brief-placeholder';
    brief.replaceWith(placeholder);
  });
  return snapshot.innerHTML;
}
