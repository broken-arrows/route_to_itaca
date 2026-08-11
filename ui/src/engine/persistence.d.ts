declare module 'dendrynexus-ten/lib/persistence.js' {
  export const SAVE_FORMAT_VERSION: number;

  export type SaveCompatibility = 'compatible' | 'incompatible' | 'unknown';
  export interface PersistenceError {
    code: string;
    message?: string;
    cause?: unknown;
    [key: string]: unknown;
  }
  export interface SaveRecord {
    saveFormatVersion: number;
    gameVersion: string | null;
    meta: Record<string, unknown>;
    state: unknown;
  }
  export type ReadResult =
    | { status: 'missing'; slot: string }
    | { status: 'unreadable'; slot: string; error: PersistenceError }
    | { status: 'corrupt'; slot: string; raw: string; error: PersistenceError }
    | { status: 'unsupported'; slot: string; raw: string; record: SaveRecord; error: PersistenceError }
    | {
        status: 'ready';
        slot: string;
        raw: string;
        record: SaveRecord;
        compatibility: SaveCompatibility;
      };
  export interface SaveStore {
    write(
      slot: string,
      state: unknown,
      meta?: Record<string, unknown>,
    ): { ok: true; record: SaveRecord } | { ok: false; error: PersistenceError };
    read(slot: string): ReadResult;
    list(): Exclude<ReadResult, { status: 'missing' }>[];
    remove(slot: string):
      | { ok: true; existed: boolean }
      | { ok: false; error: PersistenceError };
    export(slot: string):
      | { ok: true; data: string }
      | { ok: false; error: PersistenceError };
    import(slot: string, serialized: string):
      | {
          ok: true;
          status: 'ready' | 'unsupported';
          record: SaveRecord;
          compatibility?: SaveCompatibility;
        }
      | { ok: false; error: PersistenceError };
  }
  export function createSaveStore(options: {
    storage: Storage;
    storageId: string;
    gameVersion?: string;
    now?: () => Date | string | number;
  }): SaveStore;
}
