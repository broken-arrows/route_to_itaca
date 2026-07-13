import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { i18n, syncDocument } from './i18n';
import './game-bindings'; // installs the game's runtime globals — see the file
import './styles/tokens.css';

createApp(App).use(createPinia()).use(i18n).mount('#app');
syncDocument();
