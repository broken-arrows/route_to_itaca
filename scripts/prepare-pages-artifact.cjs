'use strict';

// deploy-pages selects by name, so upload retries must leave exactly one match.
module.exports = async function preparePagesArtifact({ github, context, core, artifactName, artifactId }) {
  const uploadedId = String(artifactId || '');
  if (!artifactName || !/^[1-9]\d*$/.test(uploadedId)) {
    throw new Error('The Pages upload must provide an artifact name and ID before cleanup.');
  }

  const listArtifacts = async () => {
    const artifacts = await github.paginate(github.rest.actions.listWorkflowRunArtifacts, {
      ...context.repo,
      run_id: context.runId,
      per_page: 100,
    });
    return artifacts.filter(artifact => artifact.name === artifactName);
  };

  const artifacts = await listArtifacts();
  const uploaded = artifacts.find(artifact => String(artifact.id) === uploadedId);
  if (!uploaded || uploaded.expired) {
    throw new Error(`Uploaded Pages artifact ${uploadedId} (${artifactName}) is missing or expired; no artifacts were deleted.`);
  }

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

  const remaining = await listArtifacts();
  if (remaining.length !== 1 || String(remaining[0].id) !== uploadedId) {
    throw new Error(`Expected only Pages artifact ${uploadedId} (${artifactName}) before deployment; found ${remaining.map(artifact => artifact.id).join(', ') || 'none'}.`);
  }
  core.info(`Pages artifact ${uploadedId} is ready for deployment.`);
};
