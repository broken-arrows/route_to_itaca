<script setup lang="ts">
// Renders engine-authored HTML (dendry's `convert`/`convertLine` output — the
// same trust boundary every `v-html` in this app already used) with glossary
// terms LIVE: coloured/bold from the compiled registry, and hoverable into a
// paper-styled popover for any term that carries a `tooltip`.
//
// The terms are already marked by the time this HTML arrives — `main.ts`
// installs `window.displayText`, which dendry calls on every rendered text
// run (vendor/dendrynexus-ten/lib/ui/content/html.js:14), so `<span
// class="term" data-term="ciu">CiU</span>` is already IN `html` before this
// component ever sees it. This component's whole job is the LOOK: walk the
// raw v-html DOM for `[data-term]`, apply the colour/bold from the store's
// glossary, and drive ONE shared popover for whichever term is currently
// hovered — same "one shared, viewport-aware floating element" design as the
// old shell's tooltip (out/html/game.js, `_tipEl`), reached here via event
// delegation on the root instead of a per-trigger listener (v-html content is
// re-created wholesale on every `html` change, so delegation on the constant
// root avoids re-attaching N listeners on every prose update).
import { getCurrentInstance, h, onBeforeUnmount, onMounted, ref, render as renderVNode, watch } from 'vue';
import { useGlossary } from '../glossary/useGlossary';
import GlossaryTerm from './GlossaryTerm.vue';
import WidgetHost from './viz/WidgetHost.vue';

// This SFC has TWO root nodes (the prose div + the conditional popover), so
// Vue does NOT auto-inherit fallthrough attrs (that only happens for a
// single root) — every caller passes its own layout class (`cover-prose`,
// `option-title`, plain `prose`...), so it must be forwarded explicitly onto
// the intended element (the prose div, never the popover) below.
defineOptions({ inheritAttrs: false });

// `tag` defaults to 'div' (every current call site is fine with that); it
// exists so a caller that needs to nest this inside an inline/heading
// context (e.g. an <h2>, which may not contain a block-level <div> per the
// HTML content model) can pass tag="span" instead. Never affects behaviour,
// only which element decorate()/the event listeners attach to.
const props = withDefaults(defineProps<{ html: string; tag?: string }>(), { tag: 'div' });

const root = ref<HTMLElement | null>(null);
const { termFor, colourValue } = useGlossary();

const activeTermId = ref<string | null>(null);
const activeAnchor = ref<HTMLElement | null>(null);

function closeTip(): void {
  activeTermId.value = null;
  activeAnchor.value = null;
}

// Colours/bolds every marked term in the current DOM. An id the live
// glossary doesn't recognise (stale save, mid-edit content, a term dropped
// from the registry) is left unstyled rather than throwing — this runs on
// every prose render, so it must never take the app down.
function decorate(): void {
  const el = root.value;
  if (!el) return;
  for (const span of el.querySelectorAll<HTMLElement>('[data-term]')) {
    const term = termFor(span.dataset.term);
    if (!term) continue;
    const colour = colourValue(term.colour);
    if (colour) span.style.color = colour;
    if (term.bold) span.style.fontWeight = 'bold';
    if (term.tooltip) span.classList.add('term-hoverable');
  }
}

// Capture the app context SYNCHRONOUSLY, at setup() time — getCurrentInstance()
// only resolves inside setup()/a lifecycle-hook callback's own invocation
// window, not inside a watcher callback fired later by the reactivity
// scheduler. Reused below every time widgets are (re)mounted, exactly the
// way `app.mount()` itself stamps `vnode.appContext` before handing a vnode
// to the low-level `render()` — the mechanism that gives a manually-mounted
// WidgetHost the SAME provide/inject chain as the rest of the app (pinia).
const appContext = getCurrentInstance()?.appContext ?? null;

// Elements Prose has manually mounted a WidgetHost into — tracked so they can
// be torn down (see unmountWidgets) before v-html wholesale-replaces the DOM
// they live in, and on this component's own unmount. `render(null, el)` is
// the low-level API's unmount call; it runs cleanly even if `el` has already
// been detached from the document (v-html already replaced it), because
// unmounting operates on the vnode tree Vue is tracking, not on live DOM
// attachment.
const widgetContainers: HTMLElement[] = [];

function unmountWidgets(): void {
  for (const el of widgetContainers) renderVNode(null, el);
  widgetContainers.length = 0;
}

// Hosts `<div data-widget="name" data-props='{"k":"v"}'>` markers left by
// engine-authored HTML — the widget protocol (spec §3.5, `WidgetHost.vue`'s
// own header comment). Prose's ONLY job here is finding the markers and
// handing off the raw name/props; WidgetHost is the one and only place that
// interprets `data-props` (including resolving a `configFrom` key against
// Q) — do not duplicate that resolution here.
function mountWidgets(): void {
  const el = root.value;
  if (!el) return;
  for (const target of el.querySelectorAll<HTMLElement>('[data-widget]')) {
    const name = target.dataset.widget;
    if (!name) continue;
    let widgetProps: Record<string, unknown> = {};
    if (target.dataset.props) {
      try {
        widgetProps = JSON.parse(target.dataset.props);
      } catch (err) {
        console.warn(`widget "${name}": invalid data-props JSON`, err);
      }
    }
    const vnode = h(WidgetHost, { name, props: widgetProps });
    vnode.appContext = appContext;
    renderVNode(vnode, target);
    widgetContainers.push(target);
  }
}

function findTermEl(target: EventTarget | null): HTMLElement | null {
  if (!(target instanceof HTMLElement)) return null;
  const el = target.closest<HTMLElement>('[data-term]');
  return el && root.value?.contains(el) ? el : null;
}

function onOver(e: MouseEvent): void {
  const el = findTermEl(e.target);
  if (!el) return;
  const term = termFor(el.dataset.term);
  if (!term?.tooltip) return;
  if (activeAnchor.value === el) return; // already open for this anchor
  activeTermId.value = term.id;
  activeAnchor.value = el;
}

function onOut(e: MouseEvent): void {
  const leaving = findTermEl(e.target);
  if (!leaving || leaving !== activeAnchor.value) return;
  const to = e.relatedTarget;
  if (to instanceof Node && leaving.contains(to)) return; // moved to a child, still inside
  closeTip();
}

onMounted(() => {
  decorate();
  mountWidgets();
});
// flush:'post' runs AFTER Vue has patched the DOM for this same reactive
// update, so `root`'s v-html content is already the NEW html by the time
// decorate()/mountWidgets() query it — no manual nextTick race against
// callers awaiting setProps()/nextTick() themselves (an async callback here
// would return before decorate() ran, since Vue's scheduler doesn't await a
// watcher's returned promise).
watch(
  () => props.html,
  () => {
    closeTip(); // the old anchor node is about to be destroyed by v-html
    // Unmount BEFORE re-decorating: the old widget containers are already
    // detached (v-html replaced root's innerHTML by the time this 'post'
    // watcher runs), but render(null, el) still needs to run on them to
    // dispose their reactive effects — skipping it would leak a WidgetHost
    // instance (and whatever it mounted) per prose update.
    unmountWidgets();
    decorate();
    mountWidgets();
  },
  { flush: 'post' },
);
onBeforeUnmount(unmountWidgets);
</script>

<template>
  <component
    :is="tag"
    ref="root"
    class="prose"
    v-bind="$attrs"
    v-html="props.html"
    @mouseover="onOver"
    @mouseout="onOut"
  ></component>
  <GlossaryTerm v-if="activeTermId" :term-id="activeTermId" :anchor="activeAnchor" />
</template>

<style scoped>
.prose :deep([data-term].term-hoverable) {
  cursor: help;
  text-decoration: underline dotted currentColor 1px;
  text-underline-offset: 2px;
}
</style>
