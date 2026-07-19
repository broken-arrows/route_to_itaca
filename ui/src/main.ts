import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { i18n, syncDocument } from './i18n';
// Eagerly evaluate the game's own code so a broken `source/lib/` fails fast at
// boot rather than lazily inside the first `DendryAdapter` construction (which
// is what actually calls `installGameLib` — see engine/adapter.ts).
import './game-bindings';
import './styles/tokens.css';
import { markGlossary } from './glossary/mark';
import { useGameStore } from './stores/game';

const pinia = createPinia();
const app = createApp(App).use(pinia).use(i18n);

// The dendry render hook (vendor/.../lib/ui/content/html.js:14): if
// window.displayText exists, EVERY rendered text run passes through it. The
// old shell has always defined it; the Vue app never had — which is why the
// Desk has rendered all prose with no party colours and no glossary since
// phase 2. `useGameStore(pinia)` (the explicit-instance overload) is used
// instead of the no-arg form because this runs OUTSIDE any component's
// setup(), before app.mount() has made pinia "active" via a running
// component tree — the store itself doesn't need to exist yet:
// gameStore.glossary is a live computed that starts at [] and updates
// automatically once a game loads (see stores/game.ts), so this closure
// reads it fresh on every call.
const gameStore = useGameStore(pinia);
window.displayText = (text: string) => markGlossary(text, gameStore.glossary);

app.mount('#app');
syncDocument();
