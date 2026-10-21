# Localization authoring and extraction

Localization changes presentation. Q retains English strings; predicates, string
comparisons and qdisplay bands use the original values. Saved pages and card labels
also retain English fallback content. Additive `state.presentation` version 1 stores
JSON recipes with captured values, including names, grammatical context and nested
display text. Both shells render those recipes with the selected content catalog.
English remains the saved fallback and the language of gameplay values.

## Named references and party forms

Game-owned naming catalogs provide explicit plain-text forms; the engine does not
infer articles, gender, pronunciation or political identity. This project's
`source/locales/{en,ca}/parties.json` uses party IDs as keys and the forms `bare`,
`definite`, `de`, `a`, and `per`. For example, PDeCAT has the Catalan forms
`PDeCAT`, `el PDeCAT`, `del PDeCAT`, `al PDeCAT`, and `pel PDeCAT`.

Declare scalar party qualities in `source/data/localization.json`:

```json
{
  "nameCatalogs": { "party": "parties.json" },
  "nameForms": { "party": ["bare", "definite", "de", "a", "per"] },
  "nameBindings": { "parlament_current_ciu": "party" }
}
```

`[+ parlament_current_ciu +]` keeps its authored syntax and native value. Capture
stores a separate named reference to the identity at that moment. Qdisplay inserts
are excluded from these bindings. Do not declare mixed prose or lists as names.
In JavaScript, supply the reference explicitly:

```js
G.setText(Q, "announcement", "{party} has decided to {title}.", {
  party: G.nameText("party", initiative.party),
  title: G.getText(Q, "npc_government_title")
});
```

The Catalan translation can be `{party:definite:cap} ha decidit {title}.`.
`{party:de}` requests the entire prepositional phrase, so do not also write `de`
before it. Append `:cap` to capitalize its first character; grammatical role alone
does not determine capitalization. Multiple occurrences can request different
forms. `{party:bare}` explicitly requests canonical display spelling. Plain
`{party}` retains the exact captured source spelling, preserving existing labels,
fallbacks, and technical identifiers in attributes.

Forms belong in translations, not English source keys. Extraction records named
parameter schemas and rejects forms on ordinary parameters and undeclared forms.
Every source sharing a translation key must support its formatted parameters.
`G.nameText(namespace, id, fallback?)` optionally supplies a separate native English
fallback; otherwise it preserves the supplied ID exactly (lookup is lowercase).

Shells load naming data alongside content. `mergeNames(catalog, namespaces)` creates
a runtime overlay; `engine.setLocale(locale, catalog, namespaces)` also accepts it
directly. Namespaces map to naming catalogs, e.g. `{party: parties}`. Publishing
copies the declared catalog files and validates their required forms. The new UI
bundles the same canonical naming files; Classic fetches their published copies.

Recipes are additive JSON in presentation version 1. Historical named references
retain their captured identity across rebrands, save/load and language switches.
Old plain-string recipes are not guessed into names. Missing names/forms or a
translation requiring a name where an old recipe has a string fall back safely to
the English source sentence. New saves containing named recipes require this
runtime; compatibility with older engine builds is not promised.

This facility does not implement qdisplay agreement or grammatical propagation
into arbitrary nested recipes. Coalition labels remain ordinary translated text.

## Generated text

The engine adds these helpers to each instance's game library. Scene actions receive
them through `G`; pass that run's Q to helpers that read or write qualities.

```js
G.setText(Q, "announcement", "{president} leads the {government}.", {
  president: Q.president,
  president_gender: Q.president_gender,
  government: G.text("caretaker government")
});
G.copyText(Q, "previous_announcement", "announcement");
G.updateText(Q, "announcement", {
  president: "Neus Munté",
  president_gender: "f"
});
```

Q.announcement is an English string. The copy retains the earlier president and
gender after the original changes. Parameters are captured, rather than evaluated
again when the language changes. Ordinary strings and numbers in parameters are
literal; use `G.text(key, params?)` for nested translatable text.

| Helper | Behavior |
| --- | --- |
| `setText(Q, field, keyOrRecipe, params?)` | Write the English materialization and retain a cloned recipe if the quality accepts the value. |
| `getText(Q, field)` | Return a cloned retained recipe, or a literal fallback for an unmarked/stale value. |
| `copyText(Q, to, from)` | Copy the English value and any valid recipe. Preserve numeric/unset sentinels. |
| `updateText(Q, field, patchOrMapper)` | Patch parameters of a text recipe, or transform a cloned recipe with a mapper. |
| `text(key, params?)` | Create a nested template with named parameters. |
| `englishText(recipe)` | Materialize captured English without using the active catalog, evaluating expressions or writing Q. Useful for native list/dedup logic. |
| `literalText(value)` / `rawText(html)` | Preserve literal display text / explicit technical HTML. |
| `joinText(parts, separator?)` | Compose captured recipes; an empty list yields an empty string. |
| `markupText(tag, attrs, content)` | Build HTML with literal technical attributes and explicitly marked translated content/attributes. |

Use game-owned helpers for domain markup. For example,
`G.governmentTooltipText(institution, parties, summary, label)` creates a recipe
whose party keys and institution class remain literal; summary and label may be
nested `G.text(...)` recipes. `G.rebrandGovernmentText(recipe, from, to)` changes
captured parameters without replacing translation keys.
Pass `true` as its fourth argument when migrating a native `replaceAll(from, to)`
step: exact mode preserves case and order instead of also replacing uppercase forms.

Ordinary assignment, including assigning the same string, invalidates the recipe.
`Object.assign`, deletion/recreation and property redefinition also invalidate it.
Native concatenation, copying and `.replaceAll()` do not carry recipes; use the
helpers when the result must translate. Quality constraints and change signals
still apply. Helpers reject qualities from another engine or an earlier run.

## Scene content and grammar

Authored prose, headings, resolved titles and choice/card labels use the shared
unit builder. Inserts become named placeholders, with distinct names for repeated
occurrences. Conditional clauses and emphasis have separate units. Their predicates
and inserts evaluate once during the ordinary engine action. qdisplay selection
uses its original input, then translates the selected wording.

`source/data/localization.json` declares game-specific presentation paths:

- `grammarBindings`: a directly inserted quality maps to one or more context
  qualities, e.g. `"president": "president_gender"`. A translation may select on
  gender even when the English sentence does not use it. Explicit JS recipes
  supply their own context parameters, as in the example above.
- `displayQualities`: unmarked literal display strings that may be catalogued;
  this does not recover a generated string's lost parameters or history.
- `registryFields`: `{registry, paths, parameters?}` explicitly identifies visible
  fields in game JSON; wildcard paths are supported. IDs, matching aliases, quality
  names and asset paths are not prose fields.
- `displaySinks`: `{helper, arguments}` declares zero-based visible helper arguments
  for advisory discovery. `widgetText` provides game-owned widget defaults.

Catalogs live at `source/locales/<locale>/content.json`. Entries are strings, null
(untranslated), or recursive selector objects. The English template is the key.

Interface catalogs live at `source/locales/<locale>/ui.json` for both shells.
The new UI bundles these same files and loads the published copies at startup;
there is no separate catalog under `ui/src/locales`. Classic loads them through
`out/html/localization.js`. Its `classic.messages` namespace uses the original
English interface labels as keys, preserving Classic terminology. Named
`{parameters}` are supported; missing messages fall back to English. These UI
entries are maintained explicitly, not discovered by `extract-strings`.

Classic's Options language selector remembers its preference in the game's
Classic settings namespace. Switching languages refreshes captured scene and
sidebar output, controls and widgets without scene actions or autosaves. Sidebar
arrival actions still run during ordinary gameplay rendering. Old saves without
captured recipes retain English fallback.

```json
{
  "{president} leads the {government}.": {
    "select": "president_gender",
    "cases": {
      "f": "La presidenta {president} encapçala el {government}.",
      "m": "El president {president} encapçala el {government}.",
      "other": "{president} encapçala el {government}."
    }
  },
  "{seats} seats": {
    "plural": "seats",
    "cases": { "=0": "Cap escó", "one": "{seats} escó", "other": "{seats} escons" }
  }
}
```

Every selector requires `other`. Plurals try exact numeric cases first, then
`Intl.PluralRules` for the selected locale. Selectors consume declared parameters
only and execute no expressions. Required placeholders must survive in each leaf,
except when consumed by a selector. Protected HTML tags/attributes cannot change.
Incompatible parameter schemas sharing a key are checked individually. Missing or
invalid catalog entries safely render the English template.

## Regeneration and review

From the project root:

```sh
npm run extract:strings
npm run extract:strings -- --dry-run
npm run extract:strings -- --prune --dry-run
npm run extract:strings -- --prune
```

The CLI also supports `--report <path>` and `--review <path>`. Regeneration adds
missing keys, preserves existing translations and orphaned keys, and reports
new/untranslated/orphaned entries. New English entries contain the template;
other locales receive null. Validation failures leave catalogs/reports untouched.
The command never executes scene actions to discover text.

`--prune` removes orphaned keys from `content.json` only; `ui.json` is maintained
separately. Preview the per-locale key list with `--prune --dry-run` (or add `--json`
for a machine-readable report). A real prune prints the keys and requires an
interactive terminal: type `PRUNE`, then `DELETE`. Any other answer or input closure
cancels the entire write. The extractor uses best-effort static analysis, so review
the list before confirming; older saves can still refer to removed keys.

Default outputs:

- `source/locales/_reports/extraction.json`: confirmed units with source/context
  schemas, advisory findings with assignment/display evidence, and analysis limits.
- `source/locales/_reports/dispositions.json`: persistent manual review decisions.

In this project the entire `_reports` directory is gitignored. Reports and review
decisions remain local; catalog entries and scene migrations remain versioned.

A disposition uses a finding's `id` as the object key and retains its evidence
`fingerprint`:

```json
{
  "version": 1,
  "findings": {
    "<finding-id>": {
      "fingerprint": "<finding-fingerprint>",
      "status": "migration-pending",
      "reason": "Generated tooltip will be migrated in the next content pass."
    }
  }
}
```

Allowed statuses are `non-translatable`, `migration-pending`, and
`analysis-deferred`; each requires a reason. A line move retains the finding's
identity. Changed assignment/display evidence marks an existing review stale.
Unreviewed findings remain `needs-review`; absent previous findings are reported.
Uncertain findings are advisory and are not automatically added as catalog keys.
Static analysis follows resolvable assignments/copies, concatenations, joins,
replacements and known display sinks. Dynamic properties, helper control flow and
unregistered paths remain explicit limitations; the report is not exhaustive.

## Refresh and save compatibility

Call `engine.setLocale(locale, catalog)` and `engine.refreshPresentation()` to
redraw the captured page and already selected outputs. Refresh performs no scene
actions, navigation, choice selection, deck probes, autosaves or RNG draws.
Desk's store also guards asynchronous catalog loads so the latest request wins.
`renderContent`, `renderLine`, `getSceneTitle`, `getCurrentTitle` and `renderChoices`
serve presentation consumers; rendering previously uncaptured authored content
can evaluate its expressions and is distinct from refreshing an existing page.

Snapshots survive accumulated pages, Library's temporary page backup, face-image
paragraph placement and actual cross-shell .dnt exchange. Ordinary save loading
retains its established on-display policy. Old saves without recipes remain
playable with English fallback; missing historical context cannot be recovered
without replaying gameplay. Full source migration and Catalan translation follow
this infrastructure milestone.
