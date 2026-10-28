import { useI18n } from 'vue-i18n';
import { useGameStore } from '../../stores/game';
import { markGlossary } from '../../glossary/mark';

export function useSceneWidgetText() {
  const { t, te } = useI18n();
  const game = useGameStore();
  function label(source: string) {
    const key = `widgets.scene.${source.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`;
    return te(key) ? t(key) : source;
  }
  function html(source: string) {
    void game.frame;
    const escaped = label(source).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return markGlossary(escaped, game.glossary);
  }
  function change(value: number | null, digits = 0, suffix = '') {
    if (value === null) return label('New!');
    const rounded = Number(value.toFixed(digits));
    return `${rounded > 0 ? '+' : ''}${rounded.toFixed(digits)}${suffix}`;
  }
  return { label, html, change };
}
