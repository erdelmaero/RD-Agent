"""
The motivation of the utils is for environment management

Tries to create uniform environment for the agent to run;
- All the code and data is expected included in one folder
"""

# TODO: move the scenario specific docker env into other folders.

import contextlib
import json
import os
import pickle
import select
import shutil
import subprocess
import time
import uuid
import zipfile
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Mapping,
    TypeVar,
    cast,
)

import docker  # type: ignore[import-untyped]
import docker.models  # type: ignore[import-untyped]
import docker.models.containers  # type: ignore[import-untyped]
import docker.types  # type: ignore[import-untyped]
from pydantic import model_validator
from pydantic_settings import SettingsConfigDict
from rdagent.core.conf import ExtendedBaseSettings
from rdagent.core.experiment import RD_AGENT_SETTINGS
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils import filter_redundant_text
from rdagent.utils.agent.tpl import T
from rdagent.utils.fmt import shrink_text
from rdagent.utils.workflow import wait_retry
from rich import print
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

if TYPE_CHECKING:
    # The ``kubernetes`` package is an optional dependency (extras: ``rdagent[k8s]``).
    # Importing it at module top-level would force every rdagent user to install it,
    # so we only reference it for static type checking.  At runtime, ``KubernetesEnv``
    # imports it lazily inside methods so this module remains importable without it.
    pass  # type: ignore[import-untyped]

CacheKeyFunc = Callable[[str | Path], list[list[str]]]


def extract_dir_name_from_path_config(path_str: str) -> str:
    """
    Extract the first directory component from a relative path string.

    This is used to get the basename from path configurations like "./workspace_input/"
    to use in chmod exclusion patterns.

    Args:
        path_str: A path string, typically from T() template configuration

    Returns:
        The first directory component, or empty string if not a relative path

    Examples:
        "./workspace_input/" -> "workspace_input"
        "./assets/" -> "assets"
        "/absolute/path" -> ""
    """
    p = Path(path_str)
    if not p.is_absolute() and p.parts:
        return p.parts[0]
    return ""


def cleanup_container(container: docker.models.containers.Container | None, context: str = "") -> None:  # type: ignore[no-any-unimported]
    """
    Shared helper function to clean up a Docker container.
    Always stops the container before removing it.

    Parameters
    ----------
    container : docker container object or None
        The container to clean up, or None if no container to clean up
    context : str
        Additional context for logging (e.g., "health check", "GPU test")
    """
    if container is not None:
        try:
            # Always stop first - stop() doesn't raise error if already stopped
            container.stop()
            container.remove()
        except Exception as cleanup_error:
            # Log cleanup error but don't mask the original exception
            context_str = f" {context}" if context else ""
            logger.warning(f"Failed to cleanup{context_str} container {container.id}: {cleanup_error}")


# Normalize all bind paths in volumes to absolute paths using the workspace (working_dir).
def normalize_volumes(vols: dict[str, str | dict[str, str]], working_dir: str) -> dict:
    abs_vols: dict[str, str | dict[str, str]] = {}

    def to_abs(path: str) -> str:
        # Converts a relative path to an absolute path using the workspace (working_dir).
        return os.path.abspath(os.path.join(working_dir, path)) if not os.path.isabs(path) else path

    for lp, vinfo in vols.items():
        # Support both:
        # 1. {'host_path': {'bind': 'container_path', ...}}
        # 2. {'host_path': 'container_path'}
        if isinstance(vinfo, dict):
            # abs_vols = cast(dict[str, dict[str, str]], abs_vols)
            vinfo = vinfo.copy()
            vinfo["bind"] = to_abs(vinfo["bind"])
            abs_vols[lp] = vinfo
        else:
            # abs_vols = cast(dict[str, str], abs_vols)
            abs_vols[lp] = to_abs(vinfo)
    return abs_vols


def pull_image_with_progress(image: str) -> None:
    client = docker.APIClient(base_url="unix://var/run/docker.sock")
    pull_logs = client.pull(image, stream=True, decode=True)
    progress_bars = {}

    for log in pull_logs:
        if "id" in log and log.get("progressDetail"):
            layer_id = log["id"]
            progress_detail = log["progressDetail"]
            current = progress_detail.get("current", 0)
            total = progress_detail.get("total", 0)

            if total:
                if layer_id not in progress_bars:
                    progress_bars[layer_id] = tqdm(total=total, desc=f"Layer {layer_id}", unit="B", unit_scale=True)
                progress_bars[layer_id].n = current
                progress_bars[layer_id].refresh()

        elif "status" in log:
            print(log["status"])

    for pb in progress_bars.values():
        pb.close()


class EnvConf(ExtendedBaseSettings):
    default_entry: str
    env_dict: dict = {}
    extra_volumes: dict = {}
    running_timeout_period: int | None = 3600  # 10 minutes

    """it is a function to calculating hash keys"""

    def get_workspace_content_for_hash(self, local_path: str | Path) -> list[list[str]]:
        """Get content of key files in workspace for cache hash calculation.

        Scans .py, .csv, and .yaml files.
        """
        # we must add the information of data (beyond code) into the key.
        # Otherwise, all commands operating on data will become invalid (e.g. rm -r submission.csv)
        # So we recursively walk in the folder and add the sorted relative filename list as part of the key.
        # data_key = []
        # for path in Path(local_path).rglob("*"):
        #     p = str(path.relative_to(Path(local_path)))
        #     if p.startswith("__pycache__"):
        #         continue
        #     data_key.append(p)
        # data_key = sorted(data_key)
        local_path = Path(local_path)
        return [
            [str(path.relative_to(local_path)), path.read_text()]
            for path in sorted(
                list(local_path.rglob("*.py")) + list(local_path.rglob("*.csv")) + list(local_path.rglob("*.yaml")),
            )
        ]

    redirect_stdout_to_file: bool = False
    # helper settings to support transparent;
    enable_cache: bool = True
    retry_count: int = 5  # retry count for the docker run
    retry_wait_seconds: int = 10  # retry wait seconds for the docker run
    exclude_chmod_paths: list[str] = []  # List of directory names to exclude from chmod operation

    model_config = SettingsConfigDict(
        # TODO: add prefix ....
        env_parse_none_str="None",  # Nthis is the key to accept `RUNNING_TIMEOUT_PERIOD=None`
    )


ASpecificEnvConf = TypeVar("ASpecificEnvConf", bound=EnvConf)


@dataclass
class EnvResult:
    """
    The result of running the environment.
    It contains the stdout, the exit code, and the running time in seconds.
    """

    full_stdout: str
    exit_code: int
    running_time: float
    stored_full_stdout_to_truncated_stdout: dict[str, str]

    def __init__(self, stdout: str, exit_code: int, running_time: float):
        self.full_stdout = stdout
        self.exit_code = exit_code
        self.running_time = running_time
        self.stored_full_stdout_to_truncated_stdout = {}

    def update_stdout(self, stdout: str) -> None:
        self.full_stdout = stdout

    @property
    def stdout(self) -> str:
        if self.full_stdout not in self.stored_full_stdout_to_truncated_stdout:
            truncated: str = self._get_truncated_stdout(self.full_stdout)
            self.stored_full_stdout_to_truncated_stdout[self.full_stdout] = truncated
        return self.stored_full_stdout_to_truncated_stdout[self.full_stdout]

    def hash_full_stdout(self, full_stdout: str) -> str:
        return md5_hash(full_stdout)

    @cache_with_pickle(hash_full_stdout)
    def _get_truncated_stdout(self, full_stdout: str) -> str:
        return shrink_text(
            filter_redundant_text(full_stdout),
            context_lines=RD_AGENT_SETTINGS.stdout_context_len,
            line_len=RD_AGENT_SETTINGS.stdout_line_len,
        )


class Env(Generic[ASpecificEnvConf]):
    """
    We use BaseModel as the setting due to the features it provides
    - It provides base typing and checking features.
    - loading and dumping the information will be easier: for example, we can use package like `pydantic-yaml`
    """

    conf: ASpecificEnvConf  # different env have different conf.

    def __init__(self, conf: ASpecificEnvConf):
        self.conf = conf

    def zip_a_folder_into_a_file(self, folder_path: str, zip_file_path: str) -> None:
        """
        Zip a folder into a file, use zipfile instead of subprocess
        """
        with zipfile.ZipFile(zip_file_path, "w") as z:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    z.write(
                        os.path.join(root, file),
                        os.path.relpath(os.path.join(root, file), folder_path),
                    )

    def unzip_a_file_into_a_folder(
        self,
        zip_file_path: str,
        folder_path: str,
        files_to_extract: list[str] | None = None,
    ) -> None:
        """
        Unzip a file into a folder, use zipfile instead of subprocess
        """
        if files_to_extract is None:
            # Clear folder_path before extracting
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)

        with zipfile.ZipFile(zip_file_path, "r") as z:
            if files_to_extract is not None:
                for file_name in files_to_extract:
                    try:
                        z.extract(file_name, folder_path)
                    except KeyError:
                        logger.warning(f"File {file_name} not found in cache zip.")
            else:
                z.extractall(folder_path)

    @abstractmethod
    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Prepare for the environment based on it's configure
        """

    def check_output(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        cache_key_extra_func: CacheKeyFunc | None = None,
        cache_files_to_extract: list[str] | None = None,
    ) -> str:
        result = self.run(
            entry=entry,
            local_path=local_path,
            env=env,
            running_extra_volume=running_extra_volume,
            cache_key_extra_func=cache_key_extra_func,
            cache_files_to_extract=cache_files_to_extract,
        )
        return result.stdout

    def __run_with_retry(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
    ) -> EnvResult:
        for retry_index in range(self.conf.retry_count + 1):
            try:
                start = time.time()
                log_output, return_code = self._run(
                    entry,
                    local_path,
                    env,
                    running_extra_volume=running_extra_volume,
                )
                end = time.time()
                logger.info(f"Running time: {end - start} seconds")
                if self.conf.running_timeout_period is not None and end - start + 1 >= self.conf.running_timeout_period:
                    logger.warning(
                        f"The running time exceeds {self.conf.running_timeout_period} seconds, so the process is killed.",
                    )
                    log_output += f"\n\nThe running time exceeds {self.conf.running_timeout_period} seconds, so the process is killed."
                return EnvResult(log_output, return_code, end - start)
            except Exception as e:
                if retry_index == self.conf.retry_count:
                    raise
                logger.warning(
                    f"Error while running the container: {e}, current try index: {retry_index + 1}, {self.conf.retry_count - retry_index - 1} retries left.",
                )
                time.sleep(self.conf.retry_wait_seconds)
        raise RuntimeError  # for passing CI

    def run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        cache_key_extra_func: CacheKeyFunc | None = None,
        cache_files_to_extract: list[str] | None = None,
    ) -> EnvResult:
        """
        Run the folder under the environment and return the stdout, exit code, and running time.

        Parameters
        ----------
        entry : str | None
            We may we the entry point when we run it.
            For example, we may have different entries when we run and summarize the project.
        local_path : str | None
            the local path (to project, mainly for code) will be mounted into the docker
            Here are some examples for a None local path
            - for example, run docker for updating the data in the extra_volumes.
            - simply run the image. The results are produced by output or network
        env : dict | None
            Run the code with your specific environment.
        running_extra_volume : Mapping
            Extra volumes to mount during execution.
        cache_key_extra_func : CacheKeyFunc | None
            Optional function to calculate extra information for cache key calculation
        cache_files_to_extract : list[str] | None
            Optional list of files to extract from cache zip. If None, extract all.

        Returns
        -------
            EnvResult: An object containing the stdout, the exit code, and the running time in seconds.
        """
        _env = self.conf.env_dict.copy()
        if env:
            _env.update(env)
        env = _env

        if entry is None:
            entry = self.conf.default_entry

        if "|" in entry:
            logger.warning(
                "You are using a command with a shell pipeline (i.e., '|'). "
                "The exit code ($exit_code) will reflect the result of "
                "the last command in the pipeline.",
            )

        # Exclude configured directories from chmod operation to prevent modifying
        # read-only or specially configured directories that may produce warnings.
        def _get_chmod_cmd(workspace_path: str) -> str:
            find_cmd = f"find {workspace_path} -mindepth 1 -maxdepth 1"

            # Use configurable exclude paths from DockerConf
            for name in self.conf.exclude_chmod_paths:
                if name:  # Skip empty names
                    find_cmd += f" ! -name {name}"

            chmod_cmd = f"{find_cmd} -exec chmod -R 777 {{}} +"
            return chmod_cmd

        if self.conf.redirect_stdout_to_file:
            log_file_name = md5_hash(entry)[:8] + ".log"
            log_file = Path(local_path) / f"{log_file_name}"
            log_file_relative_path = log_file.relative_to(Path(local_path))
            entry = f"{entry} > {log_file_relative_path} 2>&1"

        if self.conf.running_timeout_period is None:
            timeout_cmd = entry
        else:
            timeout_cmd = f"timeout --kill-after=10 {self.conf.running_timeout_period} {entry}"
        entry_add_timeout = (
            f"/bin/sh -c '"  # start of the sh command
            + f"{timeout_cmd}; entry_exit_code=$?; "
            + (
                f"{_get_chmod_cmd(self.conf.mount_path)}; "
                # We don't have to change the permission of the cache and input folder to remove it
                # + f"if [ -d {self.conf.mount_path}/cache ]; then chmod 777 {self.conf.mount_path}/cache; fi; " +
                #     f"if [ -d {self.conf.mount_path}/input ]; then chmod 777 {self.conf.mount_path}/input; fi; "
                if isinstance(self.conf, DockerConf)
                else ""
            )
            + "exit $entry_exit_code"
            + "'"  # end of the sh command
        )

        if self.conf.enable_cache:
            result = self.cached_run(
                entry_add_timeout,
                local_path,
                env,
                running_extra_volume,
                cache_key_extra_func,
                cache_files_to_extract,
            )
        else:
            result = self.__run_with_retry(
                entry_add_timeout,
                local_path,
                env,
                running_extra_volume,
            )
        if self.conf.redirect_stdout_to_file:
            stdout = log_file.read_text(errors="replace")
            log_file.unlink(missing_ok=True)
            result.update_stdout(stdout)
        if str(Path(local_path).resolve()) in result.stdout:
            result.update_stdout(result.stdout.replace(str(Path(local_path).resolve()), "<WORKSPACE_PATH>"))

        return result

    def cached_run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        cache_key_extra_func: CacheKeyFunc | None = None,
        cache_files_to_extract: list[str] | None = None,
    ) -> EnvResult:
        """
        Run the folder under the environment.
        Will cache the output and the folder diff for next round of running.
        Use the python codes and the parameters(entry, running_extra_volume) as key to hash the input.
        """
        target_folder = Path(RD_AGENT_SETTINGS.pickle_cache_folder_path_str) / f"utils.env.run"
        target_folder.mkdir(parents=True, exist_ok=True)

        if cache_key_extra_func is not None:
            cache_key_extra = cache_key_extra_func(local_path)
        else:
            cache_key_extra = self.conf.get_workspace_content_for_hash(local_path)

        key = md5_hash(
            json.dumps(cache_key_extra)
            + json.dumps({"entry": entry, "running_extra_volume": dict(running_extra_volume)})
            + json.dumps({"extra_volumes": self.conf.extra_volumes}),
            # + json.dumps(data_key)
        )
        if Path(target_folder / f"{key}.pkl").exists() and Path(target_folder / f"{key}.zip").exists():
            with open(target_folder / f"{key}.pkl", "rb") as f:
                ret = pickle.load(f)
            self.unzip_a_file_into_a_folder(str(target_folder / f"{key}.zip"), local_path, cache_files_to_extract)
        else:
            ret = self.__run_with_retry(entry, local_path, env, running_extra_volume)
            with open(target_folder / f"{key}.pkl", "wb") as f:
                pickle.dump(ret, f)
            self.zip_a_folder_into_a_file(local_path, str(target_folder / f"{key}.zip"))
        return cast(EnvResult, ret)

    @abstractmethod
    def _run(
        self,
        entry: str | None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[str, int]:
        """
        Execute the specified entry point within the given environment and local path.

        Parameters
        ----------
        entry : str | None
            The entry point to execute. If None, defaults to the configured entry.
        local_path : str
            The local directory path where the execution should occur.
        env : dict | None
            Environment variables to set during execution.
        kwargs : dict
            Additional keyword arguments for execution customization.

        Returns
        -------
        tuple[str, int]
            A tuple containing the standard output and the exit code.
        """
        pass

    def dump_python_code_run_and_get_results(
        self,
        code: str,
        dump_file_names: list[str],
        local_path: str,
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        code_dump_file_py_name: str | None = None,
    ) -> tuple[str, list]:
        """
        Dump the code into the local path and run the code.
        """
        random_file_name = f"{uuid.uuid4()}.py" if code_dump_file_py_name is None else f"{code_dump_file_py_name}.py"
        with open(os.path.join(local_path, random_file_name), "w") as f:
            f.write(code)
        entry = f"python {random_file_name}"
        log_output = self.check_output(entry, local_path, env, running_extra_volume=dict(running_extra_volume))
        results = []
        os.remove(os.path.join(local_path, random_file_name))
        for name in dump_file_names:
            if os.path.exists(os.path.join(local_path, f"{name}")):
                results.append(pickle.load(open(os.path.join(local_path, f"{name}"), "rb")))
                os.remove(os.path.join(local_path, f"{name}"))
            else:
                return log_output, []
        return log_output, results

    def refresh_env(self) -> None:
        """Refresh the environment, e.g., pull the latest docker image. rebuild the conda env."""
        pass


# class EnvWithCache
#

## Local Environment -----


class LocalConf(EnvConf):
    bin_path: str = ""
    """path like <path1>:<path2>:<path3>, which will be prepend to bin path."""

    retry_count: int = 0  # retry count for; run `retry_count + 1` times
    live_output: bool = True


ASpecificLocalConf = TypeVar("ASpecificLocalConf", bound=LocalConf)


class LocalEnv(Env[ASpecificLocalConf]):
    """
    Sometimes local environment may be more convenient for testing
    """

    def prepare(self) -> None: ...

    def _run(
        self,
        entry: str | None = None,
        local_path: str | None = None,
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: dict,
    ) -> tuple[str, int]:
        # Handle volume links
        volumes = {}
        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumes[lp] = rp["bind"] if isinstance(rp, dict) else rp
            cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            volumes[cache_path] = T("scenarios.data_science.share:scen.cache_path").r()
        for lp, rp in running_extra_volume.items():
            volumes[lp] = rp

        assert local_path is not None, "local_path should not be None"
        volumes = normalize_volumes(volumes, local_path)

        @contextlib.contextmanager
        def _symlink_ctx(vol_map: Mapping[str, str]) -> Generator[None, None, None]:
            created_links: list[Path] = []
            try:
                for real, link in vol_map.items():
                    link_path = Path(link)
                    real_path = Path(real)
                    if not link_path.parent.exists():
                        link_path.parent.mkdir(parents=True, exist_ok=True)
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()
                    link_path.symlink_to(real_path)
                    created_links.append(link_path)
                yield
            finally:
                for p in created_links:
                    try:
                        if p.is_symlink() or p.exists():
                            p.unlink()
                    except FileNotFoundError:
                        pass

        with _symlink_ctx(volumes):
            # Setup environment
            if env is None:
                env = {}

            # Auto-propagate CUDA_VISIBLE_DEVICES for proper GPU isolation
            if "CUDA_VISIBLE_DEVICES" in os.environ and "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]

            path = [
                *self.conf.bin_path.split(":"),
                "/bin/",
                "/usr/bin/",
                *env.get("PATH", "").split(":"),
            ]
            env["PATH"] = ":".join(path)

            if entry is None:
                entry = self.conf.default_entry

            print(Rule("[bold green]LocalEnv Logs Begin[/bold green]", style="dark_orange"))
            table = Table(title="Run Info", show_header=False)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="bold magenta")
            table.add_row("Entry", entry)
            table.add_row("Local Path", local_path or "")
            table.add_row("Env", "\n".join(f"{k}:{v}" for k, v in env.items()))
            table.add_row("Volumes", "\n".join(f"{k}:\n  {v}" for k, v in volumes.items()))
            print(table)

            cwd = Path(local_path).resolve() if local_path else None
            env = {k: str(v) if isinstance(v, int) else v for k, v in env.items()}

            process = subprocess.Popen(
                entry,
                cwd=cwd,
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Setup polling
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("The subprocess did not correctly create stdout/stderr pipes")

            if self.conf.live_output:
                stdout_fd = process.stdout.fileno()
                stderr_fd = process.stderr.fileno()

                poller = select.poll()
                poller.register(stdout_fd, select.POLLIN)
                poller.register(stderr_fd, select.POLLIN)

                combined_output = ""
                while True:
                    if process.poll() is not None:
                        break
                    events = poller.poll(100)
                    for fd, event in events:
                        if event & select.POLLIN:
                            if fd == stdout_fd:
                                while True:
                                    output = process.stdout.readline()
                                    if output == "":
                                        break
                                    Console().print(output.strip(), markup=False)
                                    combined_output += output
                            elif fd == stderr_fd:
                                while True:
                                    error = process.stderr.readline()
                                    if error == "":
                                        break
                                    Console().print(error.strip(), markup=False)
                                    combined_output += error

                # Capture any final output
                remaining_output, remaining_error = process.communicate()
                if remaining_output:
                    Console().print(remaining_output.strip(), markup=False)
                    combined_output += remaining_output
                if remaining_error:
                    Console().print(remaining_error.strip(), markup=False)
                    combined_output += remaining_error
            else:
                # Sacrifice real-time output to avoid possible standard I/O hangs
                out, err = process.communicate()
                Console().print(out, end="", markup=False)
                Console().print(err, end="", markup=False)
                combined_output = out + err

            return_code = process.returncode
            print(Rule("[bold green]LocalEnv Logs End[/bold green]", style="dark_orange"))

            return combined_output, return_code


class CondaConf(LocalConf):
    conda_env_name: str
    default_entry: str = "python main.py"

    @model_validator(mode="after")
    def change_bin_path(self, **data: Any) -> "CondaConf":
        self._update_bin_path()
        return self

    def _update_bin_path(self) -> None:
        """Update bin_path by querying the conda environment's PATH.

        This is called during initialization and can be called again after prepare()
        to ensure bin_path is set correctly even if the conda env was just created.
        """
        conda_path_result = subprocess.run(
            f"conda run -n {self.conda_env_name} --no-capture-output env | grep '^PATH='",
            capture_output=True,
            text=True,
            shell=True,
        )
        self.bin_path = conda_path_result.stdout.strip().split("=")[1] if conda_path_result.returncode == 0 else ""


class MLECondaConf(CondaConf):
    enable_cache: bool = False  # aligning with the docker settings.


## Docker Environment -----
class DockerConf(EnvConf):
    build_from_dockerfile: bool = False
    dockerfile_folder_path: Path | None = (
        None  # the path to the dockerfile optional path provided when build_from_dockerfile is False
    )
    image: str  # the image you want to build
    mount_path: str  # the path in the docker image to mount the folder
    default_entry: str  # the entry point of the image

    extra_volumes: dict = {}
    """It accept a dict of volumes, which can be either
    {<host_path>: <container_path>} or
    {<host_path>: {"bind": <container_path>, "mode": <mode, ro/rw/default is extra_volume_mode>}}
    """
    extra_volume_mode: str = "ro"  # by default. only the mount_path should be writable, others are changed to read-only

    exclude_chmod_paths: list[str] = []
    """List of directory names to exclude from chmod -R 777 operation.
    This prevents modifying permissions of read-only or specially configured directories."""

    # Declarative configuration for auto-populating exclude_chmod_paths from share.yaml
    # Subclasses can override these to specify which config keys to read
    _scenario_name: str | None = None  # e.g., "data_science", "finetune"
    _exclude_path_keys: list[str] = []  # e.g., ["input_path", "cache_path"]

    # Sometime, we need maintain some extra data for the workspace.
    # And the extra data may be shared and the downloading can be time consuming.
    # So we just want to download it once.
    network: str | None = "bridge"  # the network mode for the docker
    shm_size: str | None = None
    enable_gpu: bool = True  # because we will automatically disable GPU if not available. So we enable it by default.
    mem_limit: str | None = "48g"  # Add memory limit attribute
    cpu_count: int | None = None  # Add CPU limit attribute

    running_timeout_period: int | None = 3600  # 1 hour

    enable_cache: bool = True  # enable the cache mechanism

    retry_count: int = 5  # retry count for the docker run
    retry_wait_seconds: int = 10  # retry wait seconds for the docker run
    save_logs_to_file: bool = True
    terminal_tail_lines: int = 20

    @model_validator(mode="after")
    def populate_exclude_chmod_paths(self) -> "DockerConf":
        """
        Automatically populate exclude_chmod_paths from share.yaml configuration.

        This method reads path configurations from scenarios/<scenario_name>/share.yaml
        based on _scenario_name and _exclude_path_keys class attributes.
        """
        if not self.exclude_chmod_paths and self._scenario_name and self._exclude_path_keys:
            # Extract directory names from scenario configuration
            self.exclude_chmod_paths = [
                name
                for key in self._exclude_path_keys
                if (
                    name := extract_dir_name_from_path_config(
                        T(f"scenarios.{self._scenario_name}.share:scen.{key}").r(),
                    )
                )
            ]
        return self


class QlibCondaConf(CondaConf):
    conda_env_name: str = "rdagent4qlib"
    enable_cache: bool = False
    default_entry: str = "qrun conf.yaml"
    # extra_volumes: dict = {str(Path("~/.qlib/").expanduser().resolve().absolute()): "/root/.qlib/"}


class QlibCondaEnv(LocalEnv[QlibCondaConf]):
    def prepare(self) -> None:
        """Prepare the conda environment if not already created."""
        try:
            envs = subprocess.run("conda env list", capture_output=True, text=True, shell=True)
            if self.conf.conda_env_name not in envs.stdout:
                print(f"[yellow]Conda env '{self.conf.conda_env_name}' not found, creating...[/yellow]")
                subprocess.check_call(
                    f"conda create -y -n {self.conf.conda_env_name} python=3.10",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install --upgrade pip cython",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install git+https://github.com/microsoft/qlib.git@2fb9380b342556ddb50a4b24e4fe8655d548b2b8",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install catboost xgboost tables torch",
                    shell=True,
                )

        except Exception as e:
            print(f"[red]Failed to prepare conda env: {e}[/red]")


# ========== Conda Environment Configuration Loader ==========
# Config files location: rdagent/scenarios/finetune/env/conda/

FT_CONDA_CONFIG_DIR = Path(__file__).parent.parent / "scenarios" / "finetune" / "env" / "conda"

# Track which conda environments have been prepared in this process
# This avoids redundant pip install checks that produce verbose output
_CONDA_ENV_PREPARED: set[str] = set()


def _sync_conda_cache_with_real_envs() -> None:
    """Ensure the prepared cache includes environments that already exist on disk."""
    try:
        result = subprocess.run(
            "conda env list",
            capture_output=True,
            text=True,
            shell=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - best-effort helper
        logger.warning(f"Failed to inspect conda env list: {exc}")
        return

    env_names: set[str] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Lines look like: "base                  *  /opt/conda"
        first_column = line.split()[0]
        name = first_column.replace("*", "").strip()
        if name:
            env_names.add(name)

    _CONDA_ENV_PREPARED.update(env_names)


def _prepare_conda_env(env_name: str, requirements_file: Path, python_version: str = "3.10") -> None:
    """Prepare conda environment with dependencies from requirements.txt.

    Creates the env if it doesn't exist, then installs dependencies.
    Uses a process-level cache to avoid redundant preparation in the same run.

    Args:
        env_name: Conda environment name
        requirements_file: Path to requirements.txt file
        python_version: Python version for the environment
    """
    # 1. Create conda environment if not exists
    result = subprocess.run(f"conda env list | grep -q '^{env_name} '", shell=True)
    if result.returncode != 0:
        print(f"[yellow]Creating conda env '{env_name}' (Python {python_version})...[/yellow]")
        subprocess.check_call(f"conda create -y -n {env_name} python={python_version}", shell=True)
        subprocess.check_call(f"conda run -n {env_name} pip install --upgrade pip", shell=True)

    print(f"[yellow]Installing dependencies from {requirements_file.name}...[/yellow]")
    subprocess.check_call(f"conda run -n {env_name} pip install -r {requirements_file}", shell=True)
    print(f"[green]Conda env '{env_name}' ready[/green]")

    _CONDA_ENV_PREPARED.add(env_name)


# ========== FT (LLaMA Factory) Conda Environment ==========
class FTCondaConf(CondaConf):
    """Conda configuration for LLM fine-tuning environment."""

    model_config = SettingsConfigDict(env_prefix="FT_CONDA_")

    conda_env_name: str = "llm_finetune"
    default_entry: str = "llamafactory-cli version"
    enable_cache: bool = False


class FTCondaEnv(LocalEnv[FTCondaConf]):
    """LLaMA Factory Conda Environment with auto-dependency installation.

    Requirements: rdagent/scenarios/finetune/conda/llm_finetune_requirements.txt
    Docker equivalent: rdagent/scenarios/finetune/docker/llm_finetune_docker/Dockerfile
    """

    def prepare(self) -> None:
        try:
            # Skip if already prepared
            _sync_conda_cache_with_real_envs()
            if self.conf.conda_env_name in _CONDA_ENV_PREPARED:
                return

            # Step 1: Install base dependencies (torch, llamafactory, etc.)
            req_file = FT_CONDA_CONFIG_DIR / "llm_finetune_requirements.txt"
            _prepare_conda_env(self.conf.conda_env_name, req_file)

            # Step 2: Install flash-attn (requires torch first, uses --no-build-isolation)
            # --no-cache-dir: avoid cross-filesystem hardlink error when /tmp and ~/.cache/pip are on different mounts
            # Note: flash-attn>=2.8 is required for B200 (sm_100) support
            print("[yellow]Installing flash-attn (compiling, may take a few minutes)...[/yellow]")
            subprocess.check_call(
                f"conda run -n {self.conf.conda_env_name} pip install 'flash-attn>=2.8' --no-build-isolation --no-cache-dir",
                shell=True,
            )

            # Re-update bin_path after prepare() in case the conda env was just created
            if not self.conf.bin_path:
                self.conf._update_bin_path()
        except Exception as e:
            print(f"[red]Failed to prepare LLaMA Factory conda env: {e}[/red]")


# ========== Benchmark (OpenCompass) Conda Environment ==========
class BenchmarkCondaConf(CondaConf):
    """Conda configuration for OpenCompass benchmark evaluation."""

    model_config = SettingsConfigDict(env_prefix="BENCHMARK_CONDA_")

    conda_env_name: str = "opencompass"
    default_entry: str = "opencompass --help"
    enable_cache: bool = False
    env_dict: dict = {"COMPASS_DATA_CACHE": "/benchmarks/opencompass_data"}


class BenchmarkCondaEnv(LocalEnv[BenchmarkCondaConf]):
    """OpenCompass Conda Environment with auto-dependency installation.

    Requirements: rdagent/scenarios/finetune/conda/opencompass_requirements.txt
    Docker equivalent: rdagent/scenarios/finetune/docker/opencompass/Dockerfile
    """

    def prepare(self) -> None:
        try:
            # Skip if already prepared
            _sync_conda_cache_with_real_envs()
            if self.conf.conda_env_name in _CONDA_ENV_PREPARED:
                return
            req_file = FT_CONDA_CONFIG_DIR / "opencompass_requirements.txt"
            _prepare_conda_env(self.conf.conda_env_name, req_file)
            # Re-update bin_path after prepare() in case the conda env was just created
            if not self.conf.bin_path:
                self.conf._update_bin_path()
        except Exception as e:
            print(f"[red]Failed to prepare OpenCompass conda env: {e}[/red]")


class QlibDockerConf(DockerConf):
    model_config = SettingsConfigDict(
        env_prefix="QLIB_DOCKER_",
        env_parse_none_str="None",  # Nthis is the key to accept `RUNNING_TIMEOUT_PERIOD=None`
    )

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "qlib" / "docker"
    image: str = "local_qlib:latest"
    mount_path: str = "/workspace/qlib_workspace/"
    default_entry: str = "qrun conf.yaml"
    extra_volumes: dict = {
        str(Path("~/.qlib/").expanduser().resolve().absolute()): {
            "bind": "/root/.qlib/",
            "mode": "rw",
        },
    }
    shm_size: str | None = "16g"
    enable_gpu: bool = True
    enable_cache: bool = False
    save_logs_to_file: bool = True  # Explicitly inherit from DockerConf for compatibility


class KGDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="KG_DOCKER_")

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "kaggle_docker"
    image: str = "local_kg:latest"
    # image: str = "gcr.io/kaggle-gpu-images/python:latest"
    mount_path: str = "/workspace/kg_workspace/"
    default_entry: str = "python train.py"
    # extra_volumes: dict = {
    #     # TODO connect to the place where the data is stored
    #     Path("git_ignore_folder/data").resolve(): "/root/.data/"
    # }

    running_timeout_period: int | None = 600
    mem_limit: str | None = (
        "48g"  # Add memory limit attribute # new-york-city-taxi-fare-prediction may need more memory
    )


class DSDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="DS_DOCKER_")

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "DS_docker"
    image: str = "local_ds:latest"
    mount_path: str = "/kaggle/workspace"
    default_entry: str = "python main.py"

    running_timeout_period: int | None = 600
    mem_limit: str | None = (
        "48g"  # Add memory limit attribute # new-york-city-taxi-fare-prediction may need more memory
    )

    # Declarative configuration: automatically loads from scenarios/data_science/share.yaml
    _scenario_name: str = "data_science"
    _exclude_path_keys: list[str] = ["input_path", "cache_path"]


class MLEBDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="MLEB_DOCKER_")

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "mle_bench_docker"
    image: str = "local_mle:latest"
    # image: str = "gcr.io/kaggle-gpu-images/python:latest"
    mount_path: str = "/workspace/data_folder/"
    default_entry: str = "mlebench prepare --all"
    # extra_volumes: dict = {
    #     # TODO connect to the place where the data is stored
    #     Path("git_ignore_folder/data").resolve(): "/root/.data/"
    # }
    mem_limit: str | None = (
        "48g"  # Add memory limit attribute # new-york-city-taxi-fare-prediction may need more memory
    )
    enable_cache: bool = False


class FTDockerConf(DockerConf):
    model_config = SettingsConfigDict(env_prefix="FT_DOCKER_")

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = (
        Path(__file__).parent.parent / "scenarios" / "finetune" / "env" / "docker" / "llm_finetune"
    )
    image: str = "local_llm_finetune:latest"
    mount_path: str = "/workspace/"
    default_entry: str = "llamafactory-cli version"

    running_timeout_period: int | None = 36000  # 10 hours for training
    mem_limit: str | None = "48g"  # Large memory for LLM training
    shm_size: str | None = "16g"  # Shared memory for multi-GPU training
    enable_gpu: bool = True  # Enable GPU for LLM training
    enable_cache: bool = False  # Disable cache to avoid conflicts during training, True for debug

    # Override log output control for FT training
    save_logs_to_file: bool = True
    terminal_tail_lines: int = 20

    # Declarative configuration: automatically loads from scenarios/finetune/share.yaml
    _scenario_name: str = "finetune"
    _exclude_path_keys: list[str] = ["assets_path"]

    network: str | None = "host"  # Use host network for finetune access to litellm proxy

    def get_workspace_content_for_hash(self, local_path: str | Path) -> list[list[str]]:
        """Include dataset_info.json in cache key calculation."""
        content = super().get_workspace_content_for_hash(local_path)
        local_path = Path(local_path)
        # Add dataset_info.json if it exists
        # NOTE: data.json is excluded because it is a generated file
        for path in local_path.rglob("dataset_info.json"):
            content.append([str(path.relative_to(local_path)), path.read_text()])

        # Sort again to ensure deterministic order (though super is sorted, appended one might not be)
        content.sort(key=lambda x: x[0])
        return content


class BenchmarkDockerConf(DockerConf):
    """Docker configuration for OpenCompass benchmark evaluation."""

    model_config = SettingsConfigDict(env_prefix="BENCHMARK_DOCKER_")

    build_from_dockerfile: bool = True
    dockerfile_folder_path: Path = (
        Path(__file__).parent.parent / "scenarios" / "finetune" / "env" / "docker" / "opencompass"
    )
    image: str = "rdagent-opencompass:latest"
    mount_path: str = "/workspace/"
    default_entry: str = "opencompass --help"

    running_timeout_period: int | None = 3600  # 1 hour default for benchmarks
    mem_limit: str | None = "32g"  # Moderate memory for inference
    shm_size: str | None = "8g"  # Shared memory for model loading
    enable_gpu: bool = True  # Enable GPU for fast inference
    enable_cache: bool = False  # Disable cache for reproducibility

    # Benchmark-specific log settings
    save_logs_to_file: bool = True
    terminal_tail_lines: int = 50  # Show more lines for benchmark progress

    network: str | None = "host"  # Use host network for benchmark access to litellm proxy
    env_dict: dict = {"COMPASS_DATA_CACHE": "/benchmarks/opencompass_data"}


# physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/FIDDLE_mimic3
class DockerEnv(Env[DockerConf]):
    # TODO: Save the output into a specific file

    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Download image if it doesn't exist
        """
        client = docker.from_env()
        if (
            self.conf.build_from_dockerfile
            and self.conf.dockerfile_folder_path is not None
            and self.conf.dockerfile_folder_path.exists()
        ):
            logger.info(f"Building the image from dockerfile: {self.conf.dockerfile_folder_path}")
            resp_stream = client.api.build(
                path=str(self.conf.dockerfile_folder_path),
                tag=self.conf.image,
                network_mode=self.conf.network,
            )
            if isinstance(resp_stream, str):
                logger.info(resp_stream)
            with Progress(SpinnerColumn(), TextColumn("{task.description}")) as p:
                task = p.add_task("[cyan]Building image...")
                for part in resp_stream:
                    lines = part.decode("utf-8").split("\r\n")
                    for line in lines:
                        if line.strip():
                            status_dict = json.loads(line)
                            if "error" in status_dict:
                                p.update(
                                    task,
                                    description=f"[red]error: {status_dict['error']}",
                                )
                                raise docker.errors.BuildError(status_dict["error"], "")
                            if "stream" in status_dict:
                                p.update(task, description=status_dict["stream"])
            logger.info(f"Finished building the image from dockerfile: {self.conf.dockerfile_folder_path}")
        try:
            client.images.get(self.conf.image)
        except docker.errors.ImageNotFound:
            image_pull = client.api.pull(self.conf.image, stream=True, decode=True)
            current_status = ""
            layer_set = set()
            completed_layers = 0
            with Progress(TextColumn("{task.description}"), TextColumn("{task.fields[progress]}")) as sp:
                main_task = sp.add_task("[cyan]Pulling image...", progress="")
                status_task = sp.add_task("[bright_magenta]layer status", progress="")
                for line in image_pull:
                    if "error" in line:
                        sp.update(
                            status_task,
                            description=f"[red]error",
                            progress=line["error"],
                        )
                        raise docker.errors.APIError(line["error"])

                    layer_id = line["id"]
                    status = line["status"]
                    p_text = line.get("progress", None)

                    if layer_id not in layer_set:
                        layer_set.add(layer_id)

                    if p_text:
                        current_status = p_text

                    if status == "Pull complete" or status == "Already exists":
                        completed_layers += 1

                    sp.update(
                        main_task,
                        progress=f"[green]{completed_layers}[white]/{len(layer_set)} layers completed",
                    )
                    sp.update(
                        status_task,
                        description=f"[bright_magenta]layer {layer_id} [yellow]{status}",
                        progress=current_status,
                    )
        except docker.errors.APIError as e:
            raise RuntimeError(f"Error while pulling the image: {e}")

    def _gpu_kwargs(self, client: docker.DockerClient) -> dict:  # type: ignore[no-any-unimported]
        """get gpu kwargs based on its availability.

        Supports GPU selection via CUDA_VISIBLE_DEVICES environment variable.
        If set, only the specified GPUs will be available in the container.
        Example: CUDA_VISIBLE_DEVICES=0,1 will only expose GPU 0 and 1.
        """
        if not self.conf.enable_gpu:
            return {}

        # Check if specific GPUs are requested via CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            # Use device_ids to specify exact GPUs (cannot use count with device_ids)
            device_ids = [gpu.strip() for gpu in cuda_visible.split(",") if gpu.strip()]
            gpu_kwargs = {
                "device_requests": [docker.types.DeviceRequest(device_ids=device_ids, capabilities=[["gpu"]])],
            }
            logger.info(f"GPU selection: using specific GPUs {device_ids}")
        else:
            # Default: use all available GPUs
            gpu_kwargs = {
                "device_requests": [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
            }

        def get_image(image_name: str) -> None:
            try:
                client.images.get(image_name)
            except docker.errors.ImageNotFound:
                pull_image_with_progress(image_name)

        @wait_retry(5, 10)
        def _f() -> dict:
            container = None
            try:
                get_image(self.conf.image)
                container = client.containers.run(self.conf.image, "nvidia-smi", detach=True, **gpu_kwargs)
                # Wait for container to complete
                container.wait()
                logger.info("GPU Devices are available.")
            except docker.errors.APIError:
                return {}
            finally:
                cleanup_container(container, context="GPU test")
            return gpu_kwargs

        return _f()

    def _generate_log_header(self, entry: str | None = None) -> str:
        """
        Generate a header for log files with execution info.

        Args:
            entry: Command entry that was executed

        Returns:
            Formatted header string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = "=" * 80 + "\n"
        header += f"Docker Execution Log\n"
        header += f"Timestamp: {timestamp}\n"
        header += f"Image: {self.conf.image}\n"
        if entry:
            header += f"Command: {entry}\n"
        header += "=" * 80 + "\n\n"
        return header

    def _process_container_logs(self, logs: Iterable[bytes], local_path: str = ".", entry: str | None = None) -> str:
        """
        Process Docker container logs with optional tail mode.

        This method can be controlled via configuration:
        - save_logs_to_file: Save full logs to timestamped files in logs/ subdirectory
        - terminal_tail_lines: Show only last N lines in terminal (0 = show all)

        Args:
            logs: Docker container log stream
            local_path: Path to workspace for saving log files
            entry: Command entry that was executed (for logging header)

        Returns:
            Complete log output as string
        """
        log_output = ""

        # Determine if we should use tail mode
        use_tail_mode = self.conf.terminal_tail_lines > 0
        save_to_file = self.conf.save_logs_to_file

        # Set up log file with timestamp if needed
        log_file_path = None
        if save_to_file and local_path:
            workspace = Path(local_path)

            # Create logs subdirectory
            logs_dir = workspace / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = logs_dir / f"docker_execution_{timestamp}.log"

            # Write header with execution info
            header = self._generate_log_header(entry)
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(header)

            # Also create/update a symlink to the latest log for convenience
            latest_link = logs_dir / "docker_execution_latest.log"

            print(f"[cyan]Full logs will be saved to: {log_file_path.absolute()}[/cyan]")

        # Process logs with tail mode
        if use_tail_mode:
            log_buffer: deque[str] = deque(maxlen=self.conf.terminal_tail_lines)

            def format_tail_display() -> Text:
                text = Text()
                text.append(
                    f"[Showing last {len(log_buffer)}/{self.conf.terminal_tail_lines} lines",
                    style="dim",
                )
                if log_file_path:
                    text.append(f" | Full log: {log_file_path.name}]\n", style="dim cyan")
                else:
                    text.append("]\n", style="dim")
                text.append("-" * 80 + "\n", style="dim")
                for line in log_buffer:
                    text.append(line + "\n")
                return text

            with Live(format_tail_display(), refresh_per_second=2, console=Console()) as live:
                for log in logs:
                    decoded_log = log.strip().decode()
                    log_output += decoded_log + "\n"
                    log_buffer.append(decoded_log)

                    if log_file_path:
                        with open(log_file_path, "a", encoding="utf-8") as f:
                            f.write(decoded_log + "\n")

                    live.update(format_tail_display())
        else:
            # Default behavior: show all logs
            for log in logs:
                decoded_log = log.strip().decode()
                Console().print(decoded_log, markup=False)
                log_output += decoded_log + "\n"

                if log_file_path:
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(decoded_log + "\n")

        # Show log file location and create latest symlink
        if log_file_path and log_file_path.exists():
            print(f"[green]Full execution log saved to: {log_file_path.absolute()}[/green]")

            # Create or update symlink to latest log
            latest_link = log_file_path.parent / "docker_execution_latest.log"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            try:
                latest_link.symlink_to(log_file_path.name)
                print(f"[dim]Latest log symlink: logs/{latest_link.name} -> {log_file_path.name}[/dim]")
            except Exception:
                # Symlinks might not work on all systems (e.g., Windows without admin)
                pass

        return log_output

    def _run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[str, int]:
        if env is None:
            env = {}
        env["PYTHONWARNINGS"] = "ignore"
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        env["PYTHONUNBUFFERED"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer fork warning in multi-process training
        client = docker.from_env()

        volumes = {}
        if local_path is not None:
            local_path = os.path.abspath(local_path)
            volumes[local_path] = {"bind": self.conf.mount_path, "mode": "rw"}

        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumes[lp] = rp if isinstance(rp, dict) else {"bind": rp, "mode": self.conf.extra_volume_mode}
            cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            volumes[cache_path] = {
                "bind": T("scenarios.data_science.share:scen.cache_path").r(),
                "mode": "rw",
            }
        for lp, rp in running_extra_volume.items():
            volumes[lp] = rp if isinstance(rp, dict) else {"bind": rp, "mode": self.conf.extra_volume_mode}

        volumes = normalize_volumes(cast(dict[str, str | dict[str, str]], volumes), self.conf.mount_path)

        log_output = ""
        container: docker.models.containers.Container | None = None  # type: ignore[no-any-unimported]

        try:
            container = client.containers.run(
                image=self.conf.image,
                command=entry,
                volumes=volumes,
                environment=env,
                detach=True,
                working_dir=self.conf.mount_path,
                # auto_remove=True, # remove too fast might cause the logs not to be get
                network=self.conf.network,
                shm_size=self.conf.shm_size,
                mem_limit=self.conf.mem_limit,  # Set memory limit
                cpu_count=self.conf.cpu_count,  # Set CPU limit
                **self._gpu_kwargs(client),
            )
            assert container is not None  # Ensure container was created successfully
            logs = container.logs(stream=True)
            print(Rule("[bold green]Docker Logs Begin[/bold green]", style="dark_orange"))
            table = Table(title="Run Info", show_header=False)
            table.add_column("Key", style="bold cyan")
            table.add_column("Value", style="bold magenta")
            table.add_row("Image", self.conf.image)
            table.add_row("Container ID", container.id)
            table.add_row("Container Name", container.name)
            table.add_row("Entry", entry)
            table.add_row("Env", "\n".join(f"{k}:{v}" for k, v in env.items()))
            table.add_row("Volumes", "\n".join(f"{k}:\n  {v}" for k, v in volumes.items()))
            print(table)

            # Process logs (supports tail mode if configured)
            log_output = self._process_container_logs(logs, local_path, entry=entry)

            exit_status = container.wait()["StatusCode"]
            print(Rule("[bold green]Docker Logs End[/bold green]", style="dark_orange"))
            return log_output, exit_status
        except docker.errors.ContainerError as e:
            raise RuntimeError(f"Error while running the container: {e}")
        except docker.errors.ImageNotFound:
            raise RuntimeError("Docker image not found.")
        except docker.errors.APIError as e:
            raise RuntimeError(f"Error while running the container: {e}")
        finally:
            cleanup_container(container)

    def refresh_env(self) -> None:
        """Remove the Docker image associated with this environment."""
        client = docker.from_env()
        try:
            # Remove the specific image
            client.images.remove(image=self.conf.image, force=True)
            logger.info(f"Removed Docker image: {self.conf.image}")

            client.images.prune()
            client.api.prune_builds()
            logger.info(f"Successfully removed Docker image: {self.conf.image}")
        except docker.errors.ImageNotFound:
            logger.warning(f"Docker image not found, cannot remove: {self.conf.image}")
        except docker.errors.APIError as e:
            logger.error(f"Error while removing Docker image: {e}")
        self.prepare()


class QTDockerEnv(DockerEnv):
    """Qlib Torch Docker"""

    def __init__(self, conf: DockerConf = QlibDockerConf()):
        super().__init__(conf)

    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Download image & data if it doesn't exist
        """
        super().prepare()
        qlib_data_path = next(iter(self.conf.extra_volumes.keys()))
        if not (Path(qlib_data_path) / "qlib_data" / "cn_data").exists():
            logger.info("We are downloading!")
            cmd = "python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --interval 1d --delete_old False"
            self.check_output(entry=cmd)
        else:
            logger.info("Data already exists. Download skipped.")


class KGDockerEnv(DockerEnv):
    """Kaggle Competition Docker"""

    def __init__(self, competition: str | None = None, conf: DockerConf = KGDockerConf()):
        super().__init__(conf)


class MLEBDockerEnv(DockerEnv):
    """MLEBench Docker"""

    def __init__(self, conf: DockerConf = MLEBDockerConf()):
        super().__init__(conf)


class FTDockerEnv(DockerEnv):
    """
    LLM Fine-tuning Docker Environment with improved log output control.

    FTDockerConf enables:
    - save_logs_to_file: True (saves full logs to workspace/docker_execution.log)
    - terminal_tail_lines: 20 (only shows last 20 lines in terminal)

    To customize, set environment variables:
        export FT_DOCKER_terminal_tail_lines=50  # show last 50 lines
        export FT_DOCKER_save_logs_to_file=false # disable log file
    """

    def __init__(self, conf: DockerConf = FTDockerConf()):
        super().__init__(conf)


class BenchmarkDockerEnv(DockerEnv):
    """
    OpenCompass Benchmark Docker Environment.

    Uses BenchmarkDockerConf for evaluation-specific settings:
    - Moderate memory/GPU allocation for inference
    - Longer terminal output (50 lines) to track benchmark progress
    - Automatic Dockerfile building from scenarios/finetune/docker/opencompass

    To customize, set environment variables:
        export BENCHMARK_DOCKER_running_timeout_period=7200  # 2 hours
        export BENCHMARK_DOCKER_terminal_tail_lines=100  # show last 100 lines
    """

    def __init__(self, conf: DockerConf = BenchmarkDockerConf()):
        super().__init__(conf)


## Kubernetes Environment -----
#
# ``KubernetesEnv`` executes generated code as native Kubernetes Jobs instead of
# Docker containers.  It is intended for self-hosted clusters where running
# Docker-in-Docker is undesirable for security or scheduling reasons.
#
# Design constraints:
#   * The ``kubernetes`` Python client is an OPTIONAL dependency (extras: ``k8s``).
#     Importing this module without it must succeed; only constructing
#     ``KubernetesEnv`` requires the client to be installed.
#   * In-cluster builds are NOT supported in this initial version.  Pre-build
#     images via CI and reference them by registry tag.
#   * Volumes use PersistentVolumeClaims, not host-path bind mounts.  Workspaces
#     are isolated per run using a ``subPath`` under a single RWX PVC.

_K8S_IMPORT_ERROR_HINT = (
    "The 'kubernetes' package is required for KubernetesEnv. "
    "Install the optional extras with: pip install 'rdagent[k8s]'"
)


def _import_kubernetes() -> Any:
    """
    Lazily import the kubernetes package.

    Raises:
        ImportError: If the ``kubernetes`` package is not installed, with a
            message pointing the user to the ``rdagent[k8s]`` extras.
    """
    try:
        import kubernetes  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised in tests via mocking
        raise ImportError(_K8S_IMPORT_ERROR_HINT) from exc
    return kubernetes


class KubernetesConf(EnvConf):
    """
    Configuration for :class:`KubernetesEnv`.

    Mirrors the runtime-relevant fields of :class:`DockerConf` (image, resource
    limits, GPU toggle, timeouts) while adding Kubernetes-specific fields for
    Job placement, RBAC, persistent volumes, and image pull secrets.

    Volume handling differs from ``DockerConf``: host-path bind mounts are not
    supported.  All persistent data must be declared via :attr:`extra_pvcs`,
    which maps a PersistentVolumeClaim name to a mount path inside the
    workload pod.  ``extra_volumes`` (inherited from :class:`EnvConf`) is
    accepted only when its values are ``{pvc_name: mount_path}`` strings; any
    host-path key (matching ``/``) will raise a ``ValueError`` at run time.
    """

    # --- image ---
    image: str = ""  # the container image to run; must be pullable from the registry
    mount_path: str = "/workspace"  # path inside the container for the workspace folder
    default_entry: str = "bash run.sh"  # default command (matches DockerConf semantics)

    # --- workspace volume ---
    namespace: str = "default"
    """Kubernetes namespace in which Jobs will be created."""

    workspace_pvc: str = ""
    """Name of the RWX PersistentVolumeClaim used to share the workspace folder
    between the agent process and the spawned Job pods.  Must be created
    out-of-band (e.g. via Helm or kustomize) before running.

    Empty string disables workspace sharing — useful for trivial commands like
    ``nvidia-smi`` that do not need a workspace mount."""

    workspace_subpath_prefix: str = "rdagent-runs"
    """Sub-directory within ``workspace_pvc`` under which per-run sub-paths are
    created.  Useful when the PVC is shared with other workloads."""

    # --- extra mounts ---
    extra_pvcs: dict[str, str] = {}
    """Mapping of additional PersistentVolumeClaim names to mount paths.

    Example::

        extra_pvcs = {"qlib-data": "/qlib-data"}

    All extra PVCs are mounted read-only by default to mirror
    ``DockerConf.extra_volume_mode``."""

    extra_volume_mode: str = "ro"
    """Mount mode for entries in :attr:`extra_pvcs` ('ro' or 'rw')."""

    # --- pod scheduling / security ---
    service_account: str | None = None
    """ServiceAccount name for the Job pods.  When ``None``, the namespace
    default is used."""

    node_selector: dict[str, str] = {}
    """Optional ``spec.nodeSelector`` for the Job pods."""

    tolerations: list[dict] = []
    """Optional ``spec.tolerations`` for the Job pods.  Each entry follows the
    Kubernetes API shape (``{key, operator, value, effect, tolerationSeconds}``)."""

    image_pull_secrets: list[str] = []
    """List of ``imagePullSecrets`` references for private registries."""

    pod_security_context: dict | None = None
    """Optional ``spec.securityContext`` for the Job pods."""

    # --- resources ---
    network: str | None = None
    """Pod-level network configuration (currently informational only — Kubernetes
    Job pods always use the cluster network).  Accepted for parity with
    ``DockerConf``."""

    shm_size: str | None = None
    """If set, an additional ``emptyDir`` volume with ``medium=Memory`` is mounted
    at ``/dev/shm`` with the given size limit."""

    enable_gpu: bool = True
    """If ``True``, requests :attr:`gpu_resource_name` resources for the Job.
    GPU count defaults to 1; override with the ``K8S_GPU_COUNT`` env var or by
    subclassing."""

    gpu_resource_name: str = "nvidia.com/gpu"
    """Resource name for GPU scheduling.  Override for AMD (``amd.com/gpu``) or
    other accelerators."""

    gpu_count: int = 1
    """Number of GPUs to request when :attr:`enable_gpu` is ``True``."""

    mem_limit: str | None = "48Gi"
    """Pod memory limit (e.g. ``"48Gi"``, ``"4096Mi"``).  Set ``None`` to omit."""

    mem_request: str | None = None
    """Pod memory request.  Defaults to :attr:`mem_limit` when ``None``."""

    cpu_count: int | None = None
    """CPU limit (whole cores).  Set ``None`` to omit."""

    cpu_request: str | None = None
    """CPU request (e.g. ``"500m"``).  When ``None``, no request is set and
    Kubernetes' default request equals the limit."""

    # --- lifecycle ---
    running_timeout_period: int | None = 3600  # 1 hour, mirrors DockerConf
    """Sets ``spec.activeDeadlineSeconds`` on the Job.  ``None`` disables the
    deadline (not recommended)."""

    job_ttl_seconds: int = 3600
    """``spec.ttlSecondsAfterFinished`` — how long completed Jobs linger before
    Kubernetes garbage-collects them."""

    image_pull_policy: str = "IfNotPresent"
    """``imagePullPolicy`` for the Job container.  Use ``Always`` to force
    re-pulls when the registry tag is mutable (e.g. ``:latest``)."""

    # --- behavior ---
    save_logs_to_file: bool = True
    terminal_tail_lines: int = 20

    enable_cache: bool = True
    retry_count: int = 5
    retry_wait_seconds: int = 10

    # --- auth ---
    kubeconfig_path: str | None = None
    """Explicit kubeconfig path.  When ``None`` the standard discovery rules
    apply: in-cluster config first (when running inside a pod), then
    ``$KUBECONFIG``, then ``~/.kube/config``."""

    in_cluster: bool | None = None
    """When ``True``, force in-cluster config (requires running inside a pod
    with a mounted ServiceAccount token).  When ``False``, force kubeconfig
    loading.  ``None`` (default) auto-detects."""

    model_config = SettingsConfigDict(
        env_prefix="K8S_",
        env_parse_none_str="None",
    )


class KubernetesEnv(Env[KubernetesConf]):
    """
    Execute generated code as Kubernetes Jobs.

    This environment is a drop-in alternative to :class:`DockerEnv` for clusters
    where Docker-in-Docker is unavailable or undesirable.  It is best suited for
    deployments where the rdagent process itself runs inside the same
    Kubernetes cluster and shares an RWX PersistentVolumeClaim with the spawned
    Job pods.

    See ``docs/scens/kubernetes_env.rst`` for prerequisites and configuration
    examples.
    """

    # Mark this attribute so callers can detect whether the backend is K8s.
    _backend_kind: str = "kubernetes"

    def __init__(self, conf: KubernetesConf) -> None:
        super().__init__(conf)
        # Trigger the ImportError early with the actionable hint instead of
        # waiting until ``prepare()``/``_run()`` is called.
        _import_kubernetes()
        self._api_clients: dict[str, Any] = {}

    # ------------------------------------------------------------------ API

    def prepare(self, *args: Any, **kwargs: Any) -> None:
        """
        Validate that the configured image can be used.

        Unlike :meth:`DockerEnv.prepare`, this method never builds images.
        Pre-build all images in CI and reference them by registry tag.

        Raises:
            NotImplementedError: If the configuration requests building from a
                Dockerfile (which is not supported by ``KubernetesEnv`` —
                a Kaniko/Buildah-based variant could add this in the future).
            ValueError: If required configuration fields are missing.
        """
        if getattr(self.conf, "build_from_dockerfile", False):
            raise NotImplementedError(
                "KubernetesEnv does not support 'build_from_dockerfile'. "
                "Pre-build the image in CI and set 'image' to its registry tag.",
            )
        if not self.conf.image:
            raise ValueError("KubernetesConf.image must be set.")
        # Touch the API client to surface auth/connectivity issues early.
        self._get_batch_api()
        logger.info(f"KubernetesEnv ready (namespace={self.conf.namespace}, image={self.conf.image})")

    def refresh_env(self) -> None:
        """
        No-op for the Kubernetes backend.

        With ``imagePullPolicy: Always`` (or by changing the image tag) the
        kubelet pulls a fresh image on the next Job submission, so no extra
        action is required here.  Provided for interface parity with
        :class:`DockerEnv`.
        """
        logger.info(
            "KubernetesEnv.refresh_env() is a no-op; "
            "configure 'image_pull_policy=\"Always\"' or change the image tag to force a re-pull.",
        )

    # ----------------------------------------------------------- internals

    def _get_api_client(self) -> Any:
        """Load Kubernetes auth (in-cluster first, then kubeconfig) once and cache."""
        if "api" in self._api_clients:
            return self._api_clients["api"]
        kubernetes = _import_kubernetes()
        cfg = kubernetes.config
        try:
            if self.conf.in_cluster is True:
                cfg.load_incluster_config()
            elif self.conf.in_cluster is False:
                cfg.load_kube_config(config_file=self.conf.kubeconfig_path)
            else:
                try:
                    cfg.load_incluster_config()
                except cfg.config_exception.ConfigException:
                    cfg.load_kube_config(config_file=self.conf.kubeconfig_path)
        except Exception as exc:  # pragma: no cover - handled by integration tests
            raise RuntimeError(f"Failed to load Kubernetes configuration: {exc}") from exc

        # Workaround for kubernetes-client v36 which mis-sets the default API key
        # name when load_incluster_config is used (see kubernetes-client/python#2475).
        configuration = kubernetes.client.Configuration.get_default_copy()
        if "authorization" in configuration.api_key:
            configuration.api_key["BearerToken"] = configuration.api_key["authorization"]
        api_client = kubernetes.client.ApiClient(configuration)
        self._api_clients["api"] = api_client
        return api_client

    def _get_batch_api(self) -> Any:
        kubernetes = _import_kubernetes()
        if "batch" not in self._api_clients:
            self._api_clients["batch"] = kubernetes.client.BatchV1Api(self._get_api_client())
        return self._api_clients["batch"]

    def _get_core_api(self) -> Any:
        kubernetes = _import_kubernetes()
        if "core" not in self._api_clients:
            self._api_clients["core"] = kubernetes.client.CoreV1Api(self._get_api_client())
        return self._api_clients["core"]

    @staticmethod
    def _generate_run_id() -> str:
        """Short unique identifier suitable for embedding in Kubernetes object names."""
        return uuid.uuid4().hex[:12]

    def _job_name(self, run_id: str) -> str:
        """Build a DNS-1123-compliant Job name from the run id."""
        return f"rdagent-{run_id}"

    def _translate_volumes(
        self,
        run_id: str,
        local_path: str | None,
        running_extra_volume: Mapping,
    ) -> tuple[list[Any], list[Any], str | None]:
        """
        Translate Docker-style volume specs into Kubernetes Volume/VolumeMount lists.

        Returns:
            Tuple of (volumes, volume_mounts, workspace_subpath) where
            ``workspace_subpath`` is the per-run sub-directory within
            :attr:`KubernetesConf.workspace_pvc` (or ``None`` when no workspace
            mount is configured).

        Raises:
            ValueError: If :attr:`KubernetesConf.extra_volumes` or
                ``running_extra_volume`` contains a host-path bind mount.
        """
        kubernetes = _import_kubernetes()
        V = kubernetes.client

        volumes: list[Any] = []
        mounts: list[Any] = []
        workspace_subpath: str | None = None

        # 1. Workspace PVC + subPath
        if local_path is not None and self.conf.workspace_pvc:
            workspace_subpath = f"{self.conf.workspace_subpath_prefix}/{run_id}"
            volumes.append(
                V.V1Volume(
                    name="workspace",
                    persistent_volume_claim=V.V1PersistentVolumeClaimVolumeSource(
                        claim_name=self.conf.workspace_pvc,
                    ),
                ),
            )
            mounts.append(
                V.V1VolumeMount(
                    name="workspace",
                    mount_path=self.conf.mount_path,
                    sub_path=workspace_subpath,
                ),
            )

        # 2. extra_pvcs — declared mapping of PVC name → mount path
        ro = self.conf.extra_volume_mode == "ro"
        for idx, (pvc_name, mount_path) in enumerate(self.conf.extra_pvcs.items()):
            vol_name = f"extra-pvc-{idx}"
            volumes.append(
                V.V1Volume(
                    name=vol_name,
                    persistent_volume_claim=V.V1PersistentVolumeClaimVolumeSource(claim_name=pvc_name),
                ),
            )
            mounts.append(
                V.V1VolumeMount(
                    name=vol_name,
                    mount_path=mount_path,
                    read_only=ro,
                ),
            )

        # 3. Reject Docker-style host bind mounts from extra_volumes / running_extra_volume.
        for source, _spec in {**self.conf.extra_volumes, **running_extra_volume}.items():
            if isinstance(source, str) and (source.startswith("/") or source.startswith(".")):
                raise ValueError(
                    "KubernetesEnv does not support host-path bind mounts. "
                    f"Got host path '{source}'. Declare a PersistentVolumeClaim "
                    "via KubernetesConf.extra_pvcs={'<pvc-name>': '<mount-path>'} instead.",
                )

        # 4. Optional /dev/shm sized via shm_size.
        if self.conf.shm_size:
            volumes.append(
                V.V1Volume(
                    name="dshm",
                    empty_dir=V.V1EmptyDirVolumeSource(medium="Memory", size_limit=self.conf.shm_size),
                ),
            )
            mounts.append(V.V1VolumeMount(name="dshm", mount_path="/dev/shm"))

        return volumes, mounts, workspace_subpath

    def _build_resources(self) -> Any:
        """Build a ``V1ResourceRequirements`` object from the configured limits."""
        kubernetes = _import_kubernetes()
        V = kubernetes.client

        limits: dict[str, str] = {}
        requests: dict[str, str] = {}
        if self.conf.mem_limit:
            limits["memory"] = self.conf.mem_limit
            requests["memory"] = self.conf.mem_request or self.conf.mem_limit
        if self.conf.cpu_count is not None:
            limits["cpu"] = str(self.conf.cpu_count)
            if self.conf.cpu_request:
                requests["cpu"] = self.conf.cpu_request
        if self.conf.enable_gpu and self.conf.gpu_count > 0:
            limits[self.conf.gpu_resource_name] = str(self.conf.gpu_count)
        if not limits and not requests:
            return None
        return V.V1ResourceRequirements(limits=limits or None, requests=requests or None)

    def _build_job_spec(
        self,
        run_id: str,
        entry: str | None,
        env: dict,
        volumes: list[Any],
        volume_mounts: list[Any],
    ) -> Any:
        """Programmatically build a ``V1Job`` for the given run."""
        kubernetes = _import_kubernetes()
        V = kubernetes.client

        entry_str = entry if entry is not None else self.conf.default_entry
        # We deliberately do NOT wrap with ``timeout`` like DockerEnv does;
        # Kubernetes' ``activeDeadlineSeconds`` enforces the timeout instead.
        container_command = ["bash", "-c", entry_str]

        container = V.V1Container(
            name="workload",
            image=self.conf.image,
            image_pull_policy=self.conf.image_pull_policy,
            command=container_command,
            env=[V.V1EnvVar(name=k, value=str(v)) for k, v in env.items()],
            volume_mounts=volume_mounts or None,
            resources=self._build_resources(),
            working_dir=self.conf.mount_path
            if any(m.mount_path == self.conf.mount_path for m in volume_mounts)
            else None,
        )

        pod_spec_kwargs: dict[str, Any] = {
            "restart_policy": "Never",
            "containers": [container],
        }
        if volumes:
            pod_spec_kwargs["volumes"] = volumes
        if self.conf.service_account:
            pod_spec_kwargs["service_account_name"] = self.conf.service_account
        if self.conf.node_selector:
            pod_spec_kwargs["node_selector"] = self.conf.node_selector
        if self.conf.tolerations:
            pod_spec_kwargs["tolerations"] = [V.V1Toleration(**t) for t in self.conf.tolerations]
        if self.conf.image_pull_secrets:
            pod_spec_kwargs["image_pull_secrets"] = [
                V.V1LocalObjectReference(name=n) for n in self.conf.image_pull_secrets
            ]
        if self.conf.pod_security_context:
            pod_spec_kwargs["security_context"] = V.V1PodSecurityContext(**self.conf.pod_security_context)

        pod_template = V.V1PodTemplateSpec(
            metadata=V.V1ObjectMeta(
                labels={"app.kubernetes.io/managed-by": "rdagent", "rdagent.run-id": run_id},
            ),
            spec=V.V1PodSpec(**pod_spec_kwargs),
        )

        job_spec_kwargs: dict[str, Any] = {
            "template": pod_template,
            "backoff_limit": 0,
            "ttl_seconds_after_finished": self.conf.job_ttl_seconds,
        }
        if self.conf.running_timeout_period is not None:
            job_spec_kwargs["active_deadline_seconds"] = self.conf.running_timeout_period

        return V.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=V.V1ObjectMeta(
                name=self._job_name(run_id),
                namespace=self.conf.namespace,
                labels={"app.kubernetes.io/managed-by": "rdagent", "rdagent.run-id": run_id},
            ),
            spec=V.V1JobSpec(**job_spec_kwargs),
        )

    def _stage_workspace_in(self, local_path: str | None, workspace_subpath: str | None) -> None:
        """
        Copy the contents of ``local_path`` into the workspace PVC subPath.

        This assumes the agent process itself runs in a pod that has the same
        :attr:`KubernetesConf.workspace_pvc` mounted at a known path -- by
        convention, ``/<workspace_pvc>``.  If that mount is not present (for
        example when developing locally without the PVC), this is a no-op and
        the workspace is expected to already contain the code.
        """
        if local_path is None or workspace_subpath is None:
            return
        # The PVC is typically mounted at /<pvc-name> in the agent pod.
        # Users who mount it elsewhere should set K8S_WORKSPACE_PVC_LOCAL_MOUNT.
        local_mount_root = os.environ.get("K8S_WORKSPACE_PVC_LOCAL_MOUNT", f"/{self.conf.workspace_pvc}")
        target_dir = Path(local_mount_root) / workspace_subpath
        if not Path(local_mount_root).is_dir():
            logger.warning(
                f"Workspace PVC mount not found at '{local_mount_root}'. "
                "Set K8S_WORKSPACE_PVC_LOCAL_MOUNT to the path where the PVC is mounted "
                "in the rdagent pod, or pre-stage the workspace yourself.",
            )
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in Path(local_path).iterdir():
            dest = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    def _stage_workspace_out(self, local_path: str | None, workspace_subpath: str | None) -> None:
        """Copy results back from the workspace PVC subPath into ``local_path``."""
        if local_path is None or workspace_subpath is None:
            return
        local_mount_root = os.environ.get("K8S_WORKSPACE_PVC_LOCAL_MOUNT", f"/{self.conf.workspace_pvc}")
        source_dir = Path(local_mount_root) / workspace_subpath
        if not source_dir.is_dir():
            logger.warning(f"No workspace results found at '{source_dir}'.")
            return
        for item in source_dir.iterdir():
            dest = Path(local_path) / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    def _wait_for_pod(self, job_name: str, timeout: int = 300) -> str:
        """Wait until exactly one pod for the given job is observable and return its name."""
        core = self._get_core_api()
        deadline = time.time() + timeout
        while time.time() < deadline:
            pods = core.list_namespaced_pod(
                namespace=self.conf.namespace,
                label_selector=f"job-name={job_name}",
            )
            if pods.items:
                return pods.items[0].metadata.name
            time.sleep(1.0)
        raise TimeoutError(f"Pod for Job '{job_name}' did not appear within {timeout}s.")

    def _stream_pod_logs(self, pod_name: str, local_path: str | None, entry: str | None) -> str:
        """Stream pod logs until the container terminates and return the full text."""
        kubernetes = _import_kubernetes()
        core = self._get_core_api()

        log_output = ""
        use_tail = self.conf.terminal_tail_lines > 0
        tail_buffer: deque[str] = deque(maxlen=self.conf.terminal_tail_lines)
        log_file_path: Path | None = None
        if self.conf.save_logs_to_file and local_path:
            logs_dir = Path(local_path) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = logs_dir / f"k8s_execution_{timestamp}.log"
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"Kubernetes Execution Log\nTimestamp: {timestamp}\n")
                f.write(f"Image: {self.conf.image}\nNamespace: {self.conf.namespace}\nPod: {pod_name}\n")
                if entry:
                    f.write(f"Command: {entry}\n")
                f.write("=" * 80 + "\n\n")

        # Wait for the pod to actually be Running before streaming.
        deadline = time.time() + max(60, (self.conf.running_timeout_period or 300))
        while time.time() < deadline:
            pod = core.read_namespaced_pod(name=pod_name, namespace=self.conf.namespace)
            phase = pod.status.phase
            if phase in ("Running", "Succeeded", "Failed"):
                break
            time.sleep(1.0)

        w = kubernetes.watch.Watch()
        try:
            for line in w.stream(
                core.read_namespaced_pod_log,
                name=pod_name,
                namespace=self.conf.namespace,
                container="workload",
                follow=True,
                _request_timeout=self.conf.running_timeout_period,
            ):
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                log_output += line + "\n"
                if use_tail:
                    tail_buffer.append(line)
                else:
                    Console().print(line, markup=False)
                if log_file_path:
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
        except Exception as exc:
            logger.warning(f"Log streaming for pod '{pod_name}' ended early: {exc}")
        finally:
            w.stop()

        if use_tail and tail_buffer:
            print(
                f"[dim]Showing last {len(tail_buffer)}/{self.conf.terminal_tail_lines} lines:[/dim]",
            )
            for line in tail_buffer:
                Console().print(line, markup=False)
        if log_file_path:
            print(f"[green]Full execution log saved to: {log_file_path.absolute()}[/green]")
        return log_output

    def _wait_for_job(self, job_name: str) -> tuple[int, str]:
        """Poll Job status until completion and return (exit_code, reason)."""
        batch = self._get_batch_api()
        core = self._get_core_api()
        deadline = time.time() + self.conf.running_timeout_period + 60 if self.conf.running_timeout_period else None
        while True:
            if deadline and time.time() > deadline:
                raise TimeoutError(f"Job '{job_name}' did not finish within timeout.")
            job = batch.read_namespaced_job(name=job_name, namespace=self.conf.namespace)
            status = job.status
            if status.succeeded:
                return 0, "Succeeded"
            if status.failed:
                # Read terminated container exit code from the pod for accuracy.
                pods = core.list_namespaced_pod(
                    namespace=self.conf.namespace,
                    label_selector=f"job-name={job_name}",
                )
                exit_code = 1
                reason = "Failed"
                for pod in pods.items:
                    for cs in pod.status.container_statuses or []:
                        if cs.state and cs.state.terminated:
                            exit_code = cs.state.terminated.exit_code or 1
                            reason = cs.state.terminated.reason or reason
                            break
                return exit_code, reason
            time.sleep(2.0)

    def _cleanup_job(self, job_name: str) -> None:
        """Delete the Job and its pods.  Errors (404, etc.) are logged and swallowed."""
        kubernetes = _import_kubernetes()
        try:
            self._get_batch_api().delete_namespaced_job(
                name=job_name,
                namespace=self.conf.namespace,
                body=kubernetes.client.V1DeleteOptions(propagation_policy="Background"),
            )
        except kubernetes.client.rest.ApiException as exc:
            if exc.status != 404:
                logger.warning(f"Failed to delete Job '{job_name}': {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Unexpected error deleting Job '{job_name}': {exc}")

    # ---------------------------------------------------------------- _run

    def _run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[str, int]:
        """
        Submit a Kubernetes Job, stream its logs, and return (stdout, exit_code).

        Mirrors :meth:`DockerEnv._run` semantics so the rest of rdagent can use
        either backend interchangeably.
        """
        if env is None:
            env = {}
        env.setdefault("PYTHONWARNINGS", "ignore")
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        run_id = self._generate_run_id()
        volumes, mounts, workspace_subpath = self._translate_volumes(run_id, local_path, running_extra_volume)
        self._stage_workspace_in(local_path if local_path else None, workspace_subpath)

        job = self._build_job_spec(run_id, entry, env, volumes, mounts)
        job_name = job.metadata.name

        batch = self._get_batch_api()
        try:
            batch.create_namespaced_job(namespace=self.conf.namespace, body=job)
        except Exception as exc:
            raise RuntimeError(f"Failed to create Kubernetes Job '{job_name}': {exc}") from exc

        print(Rule(f"[bold green]K8s Job '{job_name}' submitted[/bold green]", style="dark_orange"))
        table = Table(title="Run Info", show_header=False)
        table.add_column("Key", style="bold cyan")
        table.add_column("Value", style="bold magenta")
        table.add_row("Image", self.conf.image)
        table.add_row("Namespace", self.conf.namespace)
        table.add_row("Job", job_name)
        table.add_row("Entry", entry or self.conf.default_entry)
        table.add_row("Env", "\n".join(f"{k}:{v}" for k, v in env.items()))
        table.add_row("Volumes", "\n".join(f"{m.name}:{m.mount_path}" for m in mounts))
        print(table)

        log_output = ""
        exit_code = 1
        try:
            pod_name = self._wait_for_pod(job_name)
            log_output = self._stream_pod_logs(pod_name, local_path if local_path else None, entry)
            exit_code, reason = self._wait_for_job(job_name)
            print(Rule(f"[bold green]K8s Job '{job_name}' finished: {reason}[/bold green]", style="dark_orange"))
        finally:
            self._stage_workspace_out(local_path if local_path else None, workspace_subpath)
            self._cleanup_job(job_name)

        return log_output, exit_code
