export interface BriefTab {
  id: string;
  label: string;
  gold?: boolean;
}

/**
 * The tab rail is entirely authored by the status hub. Library is identified
 * by its semantic role and receives only the universal gold presentation.
 */
export function briefTabs(
  scenes: { id: string; title: string }[],
  libraryId: string | null,
): BriefTab[] {
  return scenes.map((scene) => ({
    id: scene.id,
    label: scene.id === libraryId ? `▤ ${scene.title.toUpperCase()}` : scene.title,
    gold: scene.id === libraryId,
  }));
}
