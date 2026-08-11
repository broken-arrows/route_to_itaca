'use strict';

var SAVE_FORMAT_VERSION = 1;
var STORAGE_ID_PATTERN = /^[a-z][a-z0-9-]*$/;

function errorResult(code, error, details) {
  return Object.assign({
    code: code,
    message: error && error.message ? error.message : String(error || code),
    cause: error,
  }, details || {});
}

function parseVersion(value) {
  if (typeof value !== 'string') return null;
  var core = value.split('-', 1)[0];
  if (!/^\d+\.\d+(?:\.\d+)?$/.test(core)) return null;
  return core.split('.').map(function(part) { return Number(part); });
}

function compatibilityBetween(savedVersion, currentVersion) {
  var saved = parseVersion(savedVersion);
  var current = parseVersion(currentVersion);
  if (!saved || !current) return 'unknown';
  if (saved.length !== current.length) return 'incompatible';
  for (var i = 0; i < saved.length - 1; i++) {
    if (saved[i] !== current[i]) return 'incompatible';
  }
  return 'compatible';
}

function hasOwn(object, property) {
  return Object.prototype.hasOwnProperty.call(object, property);
}

function isEnvelope(record) {
  return !!record &&
    typeof record === 'object' &&
    !Array.isArray(record) &&
    hasOwn(record, 'saveFormatVersion') &&
    typeof record.saveFormatVersion === 'number' &&
    isFinite(record.saveFormatVersion) &&
    Math.floor(record.saveFormatVersion) === record.saveFormatVersion &&
    record.saveFormatVersion > 0 &&
    hasOwn(record, 'meta') &&
    !!record.meta &&
    typeof record.meta === 'object' &&
    !Array.isArray(record.meta) &&
    hasOwn(record, 'state');
}

function canonicalRecord(record) {
  return {
    saveFormatVersion: record.saveFormatVersion,
    gameVersion: hasOwn(record, 'gameVersion') ? record.gameVersion : null,
    meta: Object.assign({}, record.meta),
    state: record.state,
  };
}

function createSaveStore(options) {
  if (!options || !options.storage) {
    throw new TypeError('storage is required');
  }
  if (!STORAGE_ID_PATTERN.test(options.storageId || '')) {
    throw new TypeError('storageId must match [a-z][a-z0-9-]*');
  }

  var storage = options.storage;
  var storageId = options.storageId;
  var gameVersion = options.gameVersion;
  var now = options.now || function() { return new Date(); };
  var prefix = storageId + ':save:';

  function keyFor(slot) {
    if (typeof slot !== 'string' || slot.length === 0) {
      throw new TypeError('slot must be a non-empty string');
    }
    return prefix + slot;
  }

  function write(slot, state, meta) {
    var key = keyFor(slot);
    var record;
    var serialized;
    try {
      record = {
        saveFormatVersion: SAVE_FORMAT_VERSION,
        gameVersion: gameVersion == null ? null : gameVersion,
        meta: Object.assign({}, meta || {}, { savedAt: new Date(now()).toISOString() }),
        state: state,
      };
      serialized = JSON.stringify(record);
      if (!isEnvelope(JSON.parse(serialized))) {
        throw new TypeError('save envelope is not JSON-serializable');
      }
    } catch (error) {
      return { ok: false, error: errorResult('serialize-failed', error) };
    }
    try {
      storage.setItem(key, serialized);
    } catch (error) {
      return { ok: false, error: errorResult('storage-write-failed', error) };
    }
    return { ok: true, record: record };
  }

  function read(slot) {
    var key = keyFor(slot);
    var raw;
    try {
      raw = storage.getItem(key);
    } catch (error) {
      return {
        status: 'unreadable',
        slot: slot,
        error: errorResult('storage-read-failed', error),
      };
    }
    if (raw === null) {
      return { status: 'missing', slot: slot };
    }
    var parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (error) {
      return {
        status: 'corrupt',
        slot: slot,
        raw: raw,
        error: errorResult('invalid-json', error),
      };
    }
    if (!isEnvelope(parsed)) {
      return {
        status: 'corrupt',
        slot: slot,
        raw: raw,
        error: errorResult('invalid-envelope'),
      };
    }
    var record = canonicalRecord(parsed);
    if (record.saveFormatVersion !== SAVE_FORMAT_VERSION) {
      return {
        status: 'unsupported',
        slot: slot,
        raw: raw,
        record: record,
        error: errorResult('unsupported-save-format', null, {
          actual: record.saveFormatVersion,
          supported: SAVE_FORMAT_VERSION,
        }),
      };
    }
    return {
      status: 'ready',
      slot: slot,
      raw: raw,
      record: record,
      compatibility: compatibilityBetween(record.gameVersion, gameVersion),
    };
  }

  function list() {
    var slots = [];
    for (var i = 0; i < storage.length; i++) {
      var key = storage.key(i);
      if (typeof key === 'string' && key.indexOf(prefix) === 0 && key.length > prefix.length) {
        slots.push(key.slice(prefix.length));
      }
    }
    slots.sort();
    return slots.map(read).filter(function(entry) { return entry.status !== 'missing'; });
  }

  function remove(slot) {
    var key = keyFor(slot);
    try {
      var existed = storage.getItem(key) !== null;
      storage.removeItem(key);
      return { ok: true, existed: existed };
    } catch (error) {
      return { ok: false, error: errorResult('storage-remove-failed', error) };
    }
  }

  function exportSave(slot) {
    var entry = read(slot);
    if (entry.status === 'missing') {
      return { ok: false, error: errorResult('missing-save') };
    }
    if (entry.status === 'unreadable') {
      return { ok: false, error: entry.error };
    }
    return { ok: true, data: entry.raw };
  }

  function importSave(slot, serialized) {
    var key = keyFor(slot);
    var parsed;
    try {
      parsed = JSON.parse(serialized);
    } catch (error) {
      return { ok: false, error: errorResult('invalid-json', error) };
    }
    if (!isEnvelope(parsed)) {
      return { ok: false, error: errorResult('invalid-envelope') };
    }
    var record = canonicalRecord(parsed);
    var canonical;
    try {
      canonical = JSON.stringify(record);
    } catch (error) {
      return { ok: false, error: errorResult('serialize-failed', error) };
    }
    try {
      storage.setItem(key, canonical);
    } catch (error) {
      return { ok: false, error: errorResult('storage-write-failed', error) };
    }
    if (record.saveFormatVersion !== SAVE_FORMAT_VERSION) {
      return { ok: true, status: 'unsupported', record: record };
    }
    return {
      ok: true,
      status: 'ready',
      record: record,
      compatibility: compatibilityBetween(record.gameVersion, gameVersion),
    };
  }

  return {
    write: write,
    read: read,
    list: list,
    remove: remove,
    export: exportSave,
    import: importSave,
  };
}

module.exports = {
  SAVE_FORMAT_VERSION: SAVE_FORMAT_VERSION,
  createSaveStore: createSaveStore,
};
