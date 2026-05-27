.. _kubernetes_env:

=========================
Kubernetes Execution Backend
=========================

.. note::

    The Kubernetes backend is an optional feature.  Install it with::

        pip install 'rdagent[k8s]'

    This installs the ``kubernetes`` Python client.  The rest of RD-Agent
    works unchanged if the extra is not installed.

Overview
========

:class:`~rdagent.utils.env.KubernetesEnv` is a drop-in alternative to
:class:`~rdagent.utils.env.DockerEnv`.  Instead of running generated code in a
local Docker container, it submits a native Kubernetes ``Job`` and streams the
pod logs back to the agent.

It is intended for:

* **Self-hosted clusters** where Docker-in-Docker is undesirable for security,
  scheduling, or auditability reasons.
* **GPU clusters** where the Kubernetes scheduler should own GPU assignment.
* **Multi-tenant environments** where Job-level RBAC, network policies, and
  resource quotas are required.

When to use which backend
=========================

.. list-table::
    :header-rows: 1

    * - Concern
      - ``DockerEnv``
      - ``KubernetesEnv``
    * - Local developer machine
      - ✅
      - ❌ (overkill)
    * - Privileged Docker daemon required
      - Yes
      - No
    * - GPU scheduling
      - All visible GPUs
      - Native via ``nvidia.com/gpu`` (or ``amd.com/gpu``)
    * - Image builds at run time
      - ✅ (``build_from_dockerfile=True``)
      - ❌ (pre-build in CI, reference by tag)
    * - Host-path bind mounts
      - ✅
      - ❌ (use ``PersistentVolumeClaims``)
    * - Per-run isolation
      - One container, ``rm`` on exit
      - One Job, garbage-collected via ``ttlSecondsAfterFinished``

Prerequisites
=============

#. A reachable Kubernetes cluster.  RD-Agent loads in-cluster credentials when
   running inside a pod (via the mounted ServiceAccount token) and falls back
   to ``$KUBECONFIG`` / ``~/.kube/config`` otherwise.
#. A ReadWriteMany ``PersistentVolumeClaim`` shared between the rdagent
   process and the spawned Job pods.  Examples: NFS, CephFS, Longhorn (RWX),
   EFS, Azure Files.
#. A pre-built container image in a registry reachable from the cluster.
#. (Optional) ``imagePullSecrets`` if the registry requires authentication.
#. A ServiceAccount with permission to create and delete Jobs and read pod
   logs in the target namespace.

Minimal example
===============

.. code-block:: python

    from rdagent.utils.env import KubernetesConf, KubernetesEnv

    conf = KubernetesConf(
        image="registry.example.com/qlib-sandbox:1.2.3",
        mount_path="/workspace/qlib_workspace",
        default_entry="qrun conf.yaml",
        namespace="rdagent",
        workspace_pvc="rdagent-ws",
        extra_pvcs={"qlib-data": "/root/.qlib/qlib_data"},
        service_account="rdagent-runner",
        enable_gpu=True,
        mem_limit="16Gi",
    )
    env = KubernetesEnv(conf)
    env.prepare()
    result = env.run(entry="qrun conf.yaml", local_path="./my_workspace")
    print(result.stdout, result.exit_code)

Volume semantics
================

Unlike ``DockerEnv``, all persistent storage flows through PVCs:

* The **workspace PVC** (``workspace_pvc``) is mounted at ``mount_path``
  inside the Job pod.  A per-run ``subPath`` of the form
  ``<workspace_subpath_prefix>/<run-id>`` is used so concurrent runs do not
  collide.  Pre-stage workspace files into this PVC (see the
  ``K8S_WORKSPACE_PVC_LOCAL_MOUNT`` env var below).
* **Extra PVCs** are declared as ``extra_pvcs={pvc_name: mount_path}`` and
  mounted read-only by default (override with
  ``extra_volume_mode="rw"``).
* **Host-path bind mounts are rejected.**  Any entry in ``extra_volumes``
  whose key looks like a host path (starts with ``/`` or ``.``) raises a
  ``ValueError``.

If the rdagent process itself runs in a pod that has the workspace PVC
mounted, set ``K8S_WORKSPACE_PVC_LOCAL_MOUNT`` to the absolute path of that
mount so the agent can stage files in/out directly:

.. code-block:: bash

    export K8S_WORKSPACE_PVC_LOCAL_MOUNT=/var/rdagent/ws

Otherwise the agent will only warn and assume the workspace was pre-populated.

Configuration via environment variables
=======================================

All :class:`~rdagent.utils.env.KubernetesConf` fields can be set via
environment variables prefixed with ``K8S_``::

    K8S_IMAGE=registry.example.com/qlib-sandbox:1.2.3
    K8S_NAMESPACE=rdagent
    K8S_WORKSPACE_PVC=rdagent-ws
    K8S_SERVICE_ACCOUNT=rdagent-runner
    K8S_ENABLE_GPU=true
    K8S_GPU_COUNT=1
    K8S_MEM_LIMIT=16Gi
    K8S_RUNNING_TIMEOUT_PERIOD=3600

Limitations
===========

* No in-cluster image builds.  Use a CI pipeline (GitHub Actions, Tekton,
  Kaniko, etc.) to publish the image before invoking RD-Agent.
* One pod per Job.  Multi-pod parallelism within a single ``Env.run()`` is
  not supported.
* Streaming logs end when the pod terminates; if the pod is OOM-killed
  before producing any logs, only the container's exit reason
  (``OOMKilled``) and exit code are returned.
