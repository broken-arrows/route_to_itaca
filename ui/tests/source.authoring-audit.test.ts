import { createRequire } from 'node:module';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, normalize } from 'node:path';
import { promisify } from 'node:util';
import { describe, expect, it, vi } from 'vitest';

const require = createRequire(import.meta.url);
const { auditSourceFiles } = require('../../vendor/dendrynexus-ten/lib/audit.js');
const { cmd } = require('../../vendor/dendrynexus-ten/lib/cli/cmd/audit.js');
const policy = {
  deprecatedSceneProperties: {
    'is-card': 'Use role: card.',
    'is-pinned-card': 'Use role: pinned-action.',
    'is-hand': 'Use role: desk.',
    'is-deck': 'Use role: deck.',
  },
};
const file = (contents: string) => ({ name: normalize('scenes/example.scene.dry'), contents });
const audit = promisify(auditSourceFiles);

describe('source authoring audit', () => {
  it.each(['is-card', 'is-pinned-card', 'is-hand', 'is-deck'])(
    'rejects explicit %s with source location and replacement guidance', async property => {
      const result = await audit([file(`title: Example\n${property}: false\n\nBody.\n`)], policy);
      expect(result).toEqual([{
        sceneId: 'example', property,
        location: normalize('scenes/example.scene.dry') + ' line 2',
        replacement: policy.deprecatedSceneProperties[property as keyof typeof policy.deprecatedSceneProperties],
      }]);
    },
  );

  it('finds normalized aliases in subsection headers while ignoring prose and magic blocks', async () => {
    const result = await audit([file([
      'title: Example', 'role: desk', 'on-arrival: {!',
      '  var prose = "is-card: true";', '!}', '',
      'is-card: true', '', '@action', 'title: Action', 'Is_Pinned_Card: true', '', 'Body.',
    ].join('\r\n'))], policy);
    expect(result).toEqual([expect.objectContaining({
      sceneId: 'example.action', property: 'is-pinned-card',
      location: normalize('scenes/example.scene.dry') + ' line 11',
    })]);
  });

  it('accepts role-only source and skips registries and other DRY types', async () => {
    expect(await audit([
      file('title: Example\nrole: pinned-action\n\nBody.\n'),
      { name: 'data/example.json', contents: '{"is-card":true}' },
      { name: 'example.quality.dry', contents: 'is-card: true' },
    ], policy)).toEqual([]);
  });

  it('leaves source syntax available when no project policy is configured', async () => {
    expect(await audit([file('title: Example\nis-card: true\n\nBody.\n')], {})).toEqual([]);
  });

  it('propagates malformed source instead of reporting a clean audit', async () => {
    await expect(audit([file('title: Example\ntitle: Duplicate\n')], policy)).rejects.toThrow();
  });

  it('the CLI rejects a reintroduced declaration even with unchanged compiled content', async () => {
    const project = mkdtempSync(join(tmpdir(), 'rti-authoring-audit-'));
    const errors = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const logs = vi.spyOn(console, 'log').mockImplementation(() => undefined);
    try {
      mkdirSync(join(project, 'out'));
      mkdirSync(join(project, 'source'));
      writeFileSync(join(project, 'out/game.json'), JSON.stringify({ scenes: { example: { isCard: true } } }));
      writeFileSync(join(project, 'dendrynexus.audit.mjs'), 'export default ' + JSON.stringify(policy));
      const source = join(project, 'source/example.scene.dry');
      writeFileSync(source, 'title: Example\nrole: card\nis-card: true\n\nBody.\n');
      await expect(promisify(cmd.run.bind(cmd))({ project })).rejects.toThrow('Source authoring audit failed');
      expect(errors).toHaveBeenCalledWith(expect.stringContaining('Deprecated scene property "is-card"'));
      expect(errors).toHaveBeenCalledWith(expect.stringContaining('Use role: card.'));

      writeFileSync(source, 'title: Example\nrole: card\n\nBody.\n');
      await expect(promisify(cmd.run.bind(cmd))({ project })).resolves.toBeUndefined();
      expect(logs).toHaveBeenCalledWith(expect.stringContaining('Game audit clean'));
    } finally {
      errors.mockRestore(); logs.mockRestore();
      rmSync(project, { recursive: true, force: true });
    }
  });
});
