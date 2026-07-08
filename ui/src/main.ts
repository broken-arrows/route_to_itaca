import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { i18n, syncDocument } from './i18n';
import './styles/tokens.css';

createApp(App).use(createPinia()).use(i18n).mount('#app');
syncDocument();
