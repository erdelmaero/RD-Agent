"""
Integration tests for :class:`rdagent.utils.env.KubernetesEnv`.

These tests submit real Jobs to a Kubernetes cluster and are skipped unless
the ``RDAGENT_K8S_TEST`` environment variable is set to a truthy value.

Prerequisites:
* A reachable cluster (in-cluster or via ``KUBECONFIG``)
* The namespace given by ``RDAGENT_K8S_NAMESPACE`` (default: ``default``)
* An RWX PersistentVolumeClaim named ``rdagent-test-ws`` (override with
  ``RDAGENT_K8S_TEST_PVC``)
* Permission to create/delete Jobs and read Pod logs in the namespace
"""

from __future__ import annotations

import os
import tempfile
import unittest


def _enabled() -> bool:
    return os.environ.get("RDAGENT_K8S_TEST", "").lower() in {"1", "true", "yes"}


@unittest.skipUnless(_enabled(), "Set RDAGENT_K8S_TEST=1 to run integration tests against a real cluster.")
class KubernetesEnvIntegrationTests(unittest.TestCase):
    """End-to-end smoke tests against a real cluster."""

    @classmethod
    def setUpClass(cls) -> None:
        from rdagent.utils.env import KubernetesConf, KubernetesEnv

        cls.conf_cls = KubernetesConf
        cls.env_cls = KubernetesEnv
        cls.namespace = os.environ.get("RDAGENT_K8S_NAMESPACE", "default")
        cls.pvc = os.environ.get("RDAGENT_K8S_TEST_PVC", "rdagent-test-ws")
        cls.image = os.environ.get("RDAGENT_K8S_TEST_IMAGE", "python:3.11-slim")

    def _make_env(self, **overrides: object) -> object:
        kwargs = dict(
            image=self.image,
            mount_path="/workspace",
            default_entry="echo ok",
            namespace=self.namespace,
            workspace_pvc=self.pvc,
            enable_gpu=False,
            running_timeout_period=180,
            job_ttl_seconds=60,
            terminal_tail_lines=10,
            save_logs_to_file=False,
        )
        kwargs.update(overrides)
        return self.env_cls(self.conf_cls(**kwargs))

    def test_echo(self) -> None:
        env = self._make_env()
        env.prepare()
        with tempfile.TemporaryDirectory() as ws:
            result = env.run(entry='echo "hello from k8s"', local_path=ws)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello from k8s", result.stdout)

    def test_non_zero_exit(self) -> None:
        env = self._make_env()
        env.prepare()
        with tempfile.TemporaryDirectory() as ws:
            result = env.run(entry="exit 7", local_path=ws)
        self.assertEqual(result.exit_code, 7)


if __name__ == "__main__":
    unittest.main()
