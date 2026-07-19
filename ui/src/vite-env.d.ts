/// <reference types="vite/client" />

// The dendry render hook (vendor/dendrynexus-ten/lib/ui/content/html.js:14)
// calls `window.displayText(text)` on every rendered text run, if it exists.
// Installed once at boot in main.ts. See ui/src/glossary/mark.ts for what it
// does (marks glossary terms; presentation is layered on separately by
// Prose.vue/GlossaryTerm.vue).
declare global {
  interface Window {
    displayText?: (text: string) => string;
  }
}
export {};
