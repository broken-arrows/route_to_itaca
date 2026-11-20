'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const preparePagesArtifact = require('./prepare-pages-artifact.cjs');

const artifactName = 'github-pages-123-1';
const context = { repo: { owner: 'owner', repo: 'game' }, runId: 123 };
const artifact = (id, name = artifactName) => ({ id, name, expired: false, workflow_run: { id: 123 } });

// `listingDelay` hides the upload from the first N listings, modelling a run
// artifact list that lags behind the upload (or never shows it: Infinity).
function fixture(initial, options = {}) {
  let artifacts = [...initial];
  const deletions = [];
  const listings = [];
  const lookups = [];
  const sleeps = [];
  const warnings = [];
  const listWorkflowRunArtifacts = () => {};
  const github = {
    paginate: async (method, parameters) => {
      assert.equal(method, listWorkflowRunArtifacts);
      assert.deepEqual(parameters, { ...context.repo, run_id: 123, per_page: 100 });
      listings.push(parameters);
      const hidden = listings.length <= (options.listingDelay || 0);
      return artifacts.filter(item => !(hidden && item.id === 42));
    },
    rest: { actions: {
      listWorkflowRunArtifacts,
      getArtifact: async parameters => {
        assert.deepEqual(parameters, { ...context.repo, artifact_id: parameters.artifact_id });
        lookups.push(parameters.artifact_id);
        const found = (options.lookup || artifacts).find(item => item.id === parameters.artifact_id);
        if (!found) throw Object.assign(new Error('Not Found'), { status: 404 });
        return { data: found };
      },
      deleteArtifact: async parameters => {
        assert.deepEqual(parameters, { ...context.repo, artifact_id: parameters.artifact_id });
        deletions.push(parameters.artifact_id);
        if (options.deleteError && options.deleteError.status !== 404) throw options.deleteError;
        if (!options.leaveDuplicate) {
          artifacts = artifacts.filter(item => item.id !== parameters.artifact_id);
        }
        if (options.deleteError) throw options.deleteError;
      },
    } },
  };
  const core = { info: () => {}, warning: message => warnings.push(message) };
  return {
    run: (overrides = {}) => preparePagesArtifact({
      github, context, core, artifactName, artifactId: '42',
      sleep: async ms => { sleeps.push(ms); }, ...overrides,
    }),
    remaining: () => artifacts,
    deletions, listings, lookups, sleeps, warnings,
  };
}

test('a single uploaded artifact stays available for deployment retries', async () => {
  const f = fixture([artifact(42)]);
  await f.run();
  await f.run();
  assert.deepEqual(f.deletions, []);
  assert.deepEqual(f.remaining(), [artifact(42)]);
  assert.deepEqual(f.sleeps, []);
});

test('upload duplicates are removed by ID without deleting the uploaded artifact or other attempts', async () => {
  const initial = [artifact(41), artifact(42), artifact(43), artifact(44, 'other'), artifact(45, 'github-pages-123-2')];
  const f = fixture(initial);
  await f.run();
  assert.deepEqual(f.deletions, [41, 43]);
  assert.deepEqual(f.remaining(), [initial[1], initial[3], initial[4]]);
  assert.equal(f.listings.length, 2);
  assert.equal(f.warnings.length, 2);
});

test('a run artifact list that lags behind the upload is polled until the upload appears', async () => {
  const f = fixture([artifact(41), artifact(42)], { listingDelay: 2 });
  await f.run();
  assert.equal(f.sleeps.length, 2);
  assert.deepEqual(f.deletions, [41]);
  assert.deepEqual(f.remaining(), [artifact(42)]);
});

test('if the run list never shows a verified upload, cleanup is skipped with a diagnostic and nothing is deleted', async () => {
  const f = fixture([artifact(41), artifact(42)], { listingDelay: Infinity });
  await f.run();
  assert.deepEqual(f.deletions, []);
  assert.ok(f.sleeps.length > 0);
  assert.ok(f.warnings.some(message => /not listed.*41/s.test(message)), f.warnings.join('\n'));
});

test('a missing uploaded ID never causes cleanup to pick another artifact', async () => {
  const f = fixture([artifact(41), artifact(43)]);
  await assert.rejects(f.run(), /missing; no artifacts were deleted/);
  assert.deepEqual(f.listings, []);
  assert.deepEqual(f.deletions, []);
});

test('an ID belonging to a different name is not accepted', async () => {
  const f = fixture([artifact(41), artifact(42, 'other')]);
  await assert.rejects(f.run(), /named "other"/);
  assert.deepEqual(f.deletions, []);
});

test('an ID from a different workflow run is not accepted', async () => {
  const f = fixture([artifact(41)], { lookup: [{ ...artifact(42), workflow_run: { id: 999 } }] });
  await assert.rejects(f.run(), /belongs to run 999/);
  assert.deepEqual(f.deletions, []);
});

test('an expired uploaded artifact prevents cleanup', async () => {
  const f = fixture([artifact(41), { ...artifact(42), expired: true }]);
  await assert.rejects(f.run(), /expired/);
  assert.deepEqual(f.deletions, []);
});

test('an upload without a valid name and ID cannot delete anything', async () => {
  for (const overrides of [{ artifactId: '' }, { artifactId: 'invalid' }, { artifactName: '' }]) {
    const f = fixture([artifact(41), artifact(42)]);
    await assert.rejects(f.run(overrides), /must provide an artifact name and ID/);
    assert.deepEqual(f.lookups, []);
    assert.deepEqual(f.listings, []);
    assert.deepEqual(f.deletions, []);
  }
});

test('a failed deletion stops preparation and preserves the uploaded artifact', async () => {
  const f = fixture([artifact(41), artifact(42)], { deleteError: Object.assign(new Error('Forbidden'), { status: 403 }) });
  await assert.rejects(f.run(), /Forbidden/);
  assert.ok(f.remaining().some(item => item.id === 42));
});

test('an already deleted duplicate is accepted after verifying the remaining artifact', async () => {
  const f = fixture([artifact(41), artifact(42)], { deleteError: Object.assign(new Error('Not Found'), { status: 404 }) });
  await f.run();
  assert.deepEqual(f.remaining(), [artifact(42)]);
});

test('preparation fails if duplicate metadata remains after deletion', async () => {
  const f = fixture([artifact(41), artifact(42)], { leaveDuplicate: true });
  await assert.rejects(f.run(), /Expected only Pages artifact 42/);
  assert.ok(f.remaining().some(item => item.id === 42));
});
