import { onMounted, ref } from 'vue';

/** The two map outlines are game-owned assets copied into both Desk builds. */
export function useElectionSvg(file: 'local-results.svg' | 'congreso-results.svg') {
  const markup = ref('');
  const failed = ref(false);
  onMounted(async () => {
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}img/maps/${file}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      markup.value = await response.text();
    } catch (error) {
      if (import.meta.env.MODE !== 'test') console.warn(`${file}: map asset failed`, error);
      failed.value = true;
    }
  });
  return { markup, failed };
}
