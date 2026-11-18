/** Relative game assets are published beside the app. Absolute URLs retain their origin. */
export function imageUrl(path: string): string {
  return /^(?:[a-z][\w+.-]*:|\/)/i.test(path) ? path : import.meta.env.BASE_URL + path;
}
