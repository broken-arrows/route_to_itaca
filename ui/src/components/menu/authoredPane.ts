const EMPTY_LEAD = /^(?:\s|<p>\s*<\/p>)*/i;
const LEADING_H1 = /^<h1>([\s\S]*?)<\/h1>\s*/i;

export interface AuthoredPaneContent {
  titleHtml: string | null;
  bodyHtml: string;
}

/**
 * Dendry scene metadata titles label navigation targets. They are not page
 * headings. A leading authored `= ...` block compiles to the first <h1> and
 * owns the pane's visible heading instead.
 */
export function splitAuthoredPane(html: string): AuthoredPaneContent {
  const withoutEmptyLead = html.replace(EMPTY_LEAD, '');
  const heading = LEADING_H1.exec(withoutEmptyLead);
  if (!heading) return { titleHtml: null, bodyHtml: html };
  return {
    titleHtml: heading[1],
    bodyHtml: withoutEmptyLead.slice(heading[0].length),
  };
}
