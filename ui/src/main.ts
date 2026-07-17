import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { i18n, syncDocument } from './i18n';
// Eagerly evaluate the game's own code so a broken `source/lib/` fails fast at
// boot rather than lazily inside the first `DendryAdapter` construction (which
// is what actually calls `installGameLib` — see engine/adapter.ts).
import './game-bindings';
import './styles/tokens.css';

createApp(App).use(createPinia()).use(i18n).mount('#app');
syncDocument();
