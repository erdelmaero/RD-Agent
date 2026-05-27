"""
Unit tests for :class:`rdagent.utils.env.KubernetesEnv`.

These tests mock the kubernetes client so they can run in CI environments
without a real cluster (or even without ``rdagent[k8s]`` installed -- the
kubernetes package is injected as a ``MagicMock`` into ``sys.modules`` for the
duration of each test).

Integration tests against a real cluster live in
``test_env_k8s_integration.py`` and are gated by the ``RDAGENT_K8S_TEST``
environment variable.
"""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock


def _install_fake_kubernetes_module() -> mock.MagicMock:
    """
    Inject a minimal ``kubernetes`` module into ``sys.modules`` so that
    importing :class:`KubernetesEnv` succeeds even when the real client is
    not installed.

    Returns the ``MagicMock`` representing the injected module so individual
    tests can assert on calls and tweak return values.
    """
    fake_k8s = mock.MagicMock(name="kubernetes")
    # rest.ApiException must be a real exception class for ``except`` clauses
    # in our code to behave correctly.
    fake_k8s.client.rest.ApiException = type("ApiException", (Exception,), {"status": 0})
    fake_k8s.config.config_exception.ConfigException = type("ConfigException", (Exception,), {})
    sys.modules["kubernetes"] = fake_k8s
    sys.modules["kubernetes.client"] = fake_k8s.client
    sys.modules["kubernetes.client.rest"] = fake_k8s.client.rest
    sys.modules["kubernetes.config"] = fake_k8s.config
    sys.modules["kubernetes.watch"] = fake_k8s.watch
    return fake_k8s


class KubernetesConfTests(unittest.TestCase):
    """Configuration-only tests -- do not require the kubernetes client."""

    def test_conf_defaults(self) -> None:
        from rdagent.utils.env import KubernetesConf

        conf = KubernetesConf(image="nginx:latest", mount_path="/workspace", default_entry="echo hi")
        self.assertEqual(conf.namespace, "default")
        self.assertEqual(conf.workspace_subpath_prefix, "rdagent-runs")
        self.assertEqual(conf.extra_pvcs, {})
        self.assertEqual(conf.gpu_resource_name, "nvidia.com/gpu")
        self.assertTrue(conf.enable_gpu)
        self.assertEqual(conf.image_pull_policy, "IfNotPresent")

    def test_conf_env_prefix(self) -> None:
        """``K8S_*`` env vars must populate fields via pydantic settings."""
        from rdagent.utils.env import KubernetesConf

        with mock.patch.dict(
            "os.environ",
            {
                "K8S_IMAGE": "myimg:1",
                "K8S_MOUNT_PATH": "/ws",
                "K8S_DEFAULT_ENTRY": "echo k8s",
                "K8S_NAMESPACE": "edgelabs",
                "K8S_WORKSPACE_PVC": "rdagent-ws",
            },
            clear=False,
        ):
            conf = KubernetesConf()
            self.assertEqual(conf.image, "myimg:1")
            self.assertEqual(conf.namespace, "edgelabs")
            self.assertEqual(conf.workspace_pvc, "rdagent-ws")


class KubernetesEnvLazyImportTests(unittest.TestCase):
    """``KubernetesEnv`` must raise a clear error when ``kubernetes`` is missing."""

    def test_helpful_import_error(self) -> None:
        # Force the import helper to fail as if ``kubernetes`` were missing.
        with mock.patch("rdagent.utils.env._import_kubernetes") as mock_import:
            mock_import.side_effect = ImportError(
                "The 'kubernetes' package is required for KubernetesEnv. "
                "Install the optional extras with: pip install 'rdagent[k8s]'"
            )
            from rdagent.utils.env import KubernetesConf, KubernetesEnv

            with self.assertRaises(ImportError) as cm:
                KubernetesEnv(KubernetesConf(image="x", mount_path="/ws", default_entry="echo"))
            self.assertIn("rdagent[k8s]", str(cm.exception))


class KubernetesEnvLogicTests(unittest.TestCase):
    """Behavioral tests using a faked kubernetes module."""

    def setUp(self) -> None:
        self.fake_k8s = _install_fake_kubernetes_module()
        # Re-import to pick up the fake module.
        from rdagent.utils.env import KubernetesConf, KubernetesEnv  # noqa: F401

        self.KubernetesConf = KubernetesConf
        self.KubernetesEnv = KubernetesEnv

    def tearDown(self) -> None:
        for mod in ("kubernetes", "kubernetes.client", "kubernetes.client.rest", "kubernetes.config", "kubernetes.watch"):
            sys.modules.pop(mod, None)

    def _make_env(self, **overrides: object) -> object:
        defaults = dict(
            image="reg.example.com/img:1",
            mount_path="/workspace",
            default_entry="bash run.sh",
            namespace="edgelabs",
            workspace_pvc="rdagent-ws",
        )
        defaults.update(overrides)
        return self.KubernetesEnv(self.KubernetesConf(**defaults))

    def test_prepare_rejects_dockerfile_build(self) -> None:
        env = self._make_env()
        # prepare() checks getattr(self.conf, "build_from_dockerfile", False).
        # Patch it on the conf instance's __dict__ directly (bypasses pydantic).
        object.__setattr__(env.conf, "build_from_dockerfile", True)
        with self.assertRaises(NotImplementedError):
            env.prepare()

    def test_prepare_requires_image(self) -> None:
        env = self._make_env()
        env.conf.image = ""
        with self.assertRaises(ValueError):
            env.prepare()

    def test_translate_volumes_basic(self) -> None:
        env = self._make_env(
            extra_pvcs={"qlib-data": "/qlib-data"},
            shm_size="2Gi",
        )
        volumes, mounts, subpath = env._translate_volumes(  # type: ignore[attr-defined]
            run_id="abc123",
            local_path="/tmp/ws",
            running_extra_volume={},
        )
        # Expect: workspace + extra-pvc-0 + dshm
        self.assertEqual(len(volumes), 3)
        self.assertEqual(len(mounts), 3)
        self.assertTrue(subpath.startswith("rdagent-runs/"))
        self.assertIn("abc123", subpath)
        # The API model constructors are MagicMocks; verify the kw used.
        self.fake_k8s.client.V1PersistentVolumeClaimVolumeSource.assert_any_call(claim_name="rdagent-ws")
        self.fake_k8s.client.V1PersistentVolumeClaimVolumeSource.assert_any_call(claim_name="qlib-data")
        self.fake_k8s.client.V1EmptyDirVolumeSource.assert_called_once_with(medium="Memory", size_limit="2Gi")

    def test_translate_volumes_rejects_host_paths(self) -> None:
        env = self._make_env()
        env.conf.extra_volumes = {"/host/data": "/data"}
        with self.assertRaises(ValueError) as cm:
            env._translate_volumes(run_id="x", local_path="/tmp/ws", running_extra_volume={})  # type: ignore[attr-defined]
        self.assertIn("host-path bind mounts", str(cm.exception))
        self.assertIn("extra_pvcs", str(cm.exception))

    def test_build_resources_with_gpu(self) -> None:
        env = self._make_env(
            mem_limit="8Gi",
            cpu_count=4,
            cpu_request="1",
            enable_gpu=True,
            gpu_count=2,
            gpu_resource_name="nvidia.com/gpu",
        )
        env._build_resources()  # type: ignore[attr-defined]
        self.fake_k8s.client.V1ResourceRequirements.assert_called_once()
        kwargs = self.fake_k8s.client.V1ResourceRequirements.call_args.kwargs
        self.assertEqual(kwargs["limits"]["memory"], "8Gi")
        self.assertEqual(kwargs["limits"]["cpu"], "4")
        self.assertEqual(kwargs["limits"]["nvidia.com/gpu"], "2")
        self.assertEqual(kwargs["requests"]["cpu"], "1")

    def test_build_job_spec_basic(self) -> None:
        env = self._make_env()
        env._build_job_spec(  # type: ignore[attr-defined]
            run_id="r1",
            entry="echo hi",
            env={"FOO": "bar"},
            volumes=[mock.MagicMock(name="vol")],
            volume_mounts=[SimpleNamespace(mount_path="/workspace", name="workspace")],
        )
        # The Job was created with our expected name.
        self.fake_k8s.client.V1ObjectMeta.assert_any_call(
            name="rdagent-r1",
            namespace="edgelabs",
            labels={"app.kubernetes.io/managed-by": "rdagent", "rdagent.run-id": "r1"},
        )
        # bash -c wrapping must be applied.
        container_kwargs = self.fake_k8s.client.V1Container.call_args.kwargs
        self.assertEqual(container_kwargs["command"], ["bash", "-c", "echo hi"])
        self.assertEqual(container_kwargs["image_pull_policy"], "IfNotPresent")

    def test_cleanup_job_tolerates_404(self) -> None:
        env = self._make_env()
        api_exc_cls = self.fake_k8s.client.rest.ApiException
        exc_404 = api_exc_cls("not found")
        exc_404.status = 404
        # Make delete_namespaced_job raise the 404
        batch_mock = mock.MagicMock()
        batch_mock.delete_namespaced_job.side_effect = exc_404
        env._api_clients["batch"] = batch_mock  # type: ignore[attr-defined]

        # Should not raise.
        env._cleanup_job("rdagent-x")  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
