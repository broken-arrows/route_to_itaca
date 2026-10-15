# UI fonts

Selected 2026-10-13 after the typography comparison:

- Mazius Display Bold: main game title, upright weight 700.
- Cooper Hewitt Medium: headings, weight 500. Supplied WOFF copied unchanged
  from the user's CooperHewitt-WebFonts-public.zip. Its internal weight is
  mislabelled; CSS registers the actual named Medium cut explicitly.
- Raleway variable upright/italic: Regular 400 reading/compact text, Semibold
  600 controls/emphasis. Existing TTF files are used without modification.

Only those faces are registered by styles/typography.css and emitted by Vite.
Each selected family has LICENSE.txt; Vite ships them in dist/licenses/.
Raleway's copyright is copied from its supplied font's name table, followed
by the standard SIL Open Font License text.

Other candidate assets remain available to the local comparison, whose font
registration is separate from the shipped UI. Original downloads are preserved
in .scratch/font-originals/; ZIP and downloaded candidates are preserved in
.scratch/font-comparison-assets/. The comparison is
.scratch/phase6c-font-prototype.html, served through Vite's /@fs/ URL.

Full typography roles and sizing are documented in
docs/design/typography_system.md. Body is 1rem/1.5, compact text .875rem/1.4.
There are no separate typewriter or newspaper fonts in the new UI.
