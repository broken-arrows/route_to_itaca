'use strict';

// deploy-pages selects by name, so upload retries must leave exactly one match.
//
// The upload is verified by ID (GET /actions/artifacts/{id}), not by the run's
// artifact list: that list has failed to show a just-uploaded artifact on
// consecutive attempts. Cleanup only runs once the list does show the upload.
// If it never does, nothing is deleted and the deploy job (a later job, which
// fails on zero or multiple same-name matches) remains the final check.
const LIST_ATTEMPTS = 6;
const LIST_DELAY_MS = 5000;

const defaultSleep = ms => new Promise(resolve => setTimeout(resolve, ms));

module.exports = async function preparePagesArtifact({
  github, context, core, artifactName, artifactId, sleep = defaultSleep,
}) {
  const uploadedId = String(artifactId || '');
  if (!artifactName || !/^[1-9]\d*$/.test(uploadedId)) {
    throw new Error('The Pages upload must provide an artifact name and ID before cleanup.');
  }

  let uploaded;
  try {
    ({ data: uploaded } = await github.rest.actions.getArtifact({
      ...context.repo,
      artifact_id: Number(uploadedId),
    }));
  } catch (error) {
    if (error.status !== 404) throw error;
    throw new Error(`Uploaded Pages artifact ${uploadedId} (${artifactName}) is missing; no artifacts were deleted.`);
  }
  const uploadedRun = uploaded.workflow_run && uploaded.workflow_run.id;
  if (uploaded.name !== artifactName) {
    throw new Error(`Uploaded Pages artifact ${uploadedId} is named "${uploaded.name}", expected "${artifactName}"; no artifacts were deleted.`);
  }
  if (uploadedRun !== undefined && uploadedRun !== null && String(uploadedRun) !== String(context.runId)) {
    throw new Error(`Uploaded Pages artifact ${uploadedId} belongs to run ${uploadedRun}, not ${context.runId}; no artifacts were deleted.`);
  }
  if (uploaded.expired) {
    throw new Error(`Uploaded Pages artifact ${uploadedId} (${artifactName}) is expired; no artifacts were deleted.`);
  }

  const listRunArtifacts = () => github.paginate(github.rest.actions.listWorkflowRunArtifacts, {
    ...context.repo,
    run_id: context.runId,
    per_page: 100,
  });
  const describe = list => list.map(item => `${item.id} "${item.name}"`).join(', ') || 'none';

  let listed = [];
  for (let attempt = 1; attempt <= LIST_ATTEMPTS; attempt++) {
    listed = await listRunArtifacts();
    if (listed.some(item => String(item.id) === uploadedId)) break;
    if (attempt === LIST_ATTEMPTS) {
      core.warning(`Verified Pages artifact ${uploadedId} (${artifactName}) is not listed for run ${context.runId} after ${LIST_ATTEMPTS} attempts; skipping duplicate cleanup without deleting anything. Run artifacts listed: ${describe(listed)}.`);
      return;
    }
    core.info(`Pages artifact ${uploadedId} not yet listed for run ${context.runId} (attempt ${attempt}/${LIST_ATTEMPTS}; listed: ${describe(listed)}); retrying.`);
    await sleep(LIST_DELAY_MS);
  }

  const artifacts = listed.filter(item => item.name === artifactName);
  core.info(`Found ${artifacts.length} artifact(s) named ${artifactName}; preserving upload ${uploadedId}.`);
  for (const artifact of artifacts) {
    if (String(artifact.id) === uploadedId) continue;
    core.warning(`Deleting duplicate Pages artifact ${artifact.id}; preserving ${uploadedId}.`);
    try {
      await github.rest.actions.deleteArtifact({
        ...context.repo,
        artifact_id: artifact.id,
      });
    } catch (error) {
      // A retried DELETE can return 404 after the first request already succeeded.
      if (error.status !== 404) throw error;
    }
  }

  const remaining = (await listRunArtifacts()).filter(item => item.name === artifactName);
  if (remaining.length !== 1 || String(remaining[0].id) !== uploadedId) {
    throw new Error(`Expected only Pages artifact ${uploadedId} (${artifactName}) before deployment; found ${remaining.map(artifact => artifact.id).join(', ') || 'none'}.`);
  }
  core.info(`Pages artifact ${uploadedId} is ready for deployment.`);
};
