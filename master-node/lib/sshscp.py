import shlex
import logging
import paramiko
from paramiko import SSHClient
from scp import SCPClient

log = logging.getLogger(__name__)

# Per-run artifacts a Cooja run leaves inside the container. build/cooja holds
# the mtype*.cooja firmware libraries compiled for each simulation; their names
# are random per run and nothing references them afterwards, so they would
# accumulate forever in the container writable layer if not removed.
GENERATED_ARTIFACTS = ("build/cooja", "COOJA.testlog", "COOJA.log")

def create_ssh_client(
            hostname: str, 
            port: int, 
            username: str, 
            password: str
        ) -> SSHClient:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port=port, username=username, password=password)
    return client

def send_files_scp(
        client: SSHClient, 
        local_path: str, 
        remote_path: str, 
        source_files: list[str], 
        target_files: list[str]
    ) -> None:
    if len(source_files) != len(target_files):
        log.info("The number of source files is not equal number of targets.")
        return
    with SCPClient(client.get_transport()) as scp:
        for src, dest in zip(source_files, target_files):
            local_file_path = local_path + "/" + src
            remote_file_path = remote_path + "/" + dest
            log.info(f"Sending {local_file_path} to {remote_file_path}")
            scp.put(local_file_path, remote_file_path)

def build_cleanup_command(remote_dir: str, remote_files: list[str] | None = None) -> str:
    # remote_files names come from GridFS documents; refuse anything that
    # could escape remote_dir so rm never touches files shipped in the image.
    targets = list(GENERATED_ARTIFACTS)
    for name in remote_files or []:
        if name.startswith("/") or ".." in name:
            log.warning(f"Skipping unsafe cleanup target: {name}")
            continue
        targets.append(name)
    quoted = " ".join(shlex.quote(t) for t in targets)
    return f"cd {shlex.quote(remote_dir)} && rm -rf -- {quoted}"

def cleanup_remote_files(
        client: SSHClient,
        remote_dir: str,
        remote_files: list[str] | None = None,
        timeout: float = 60.0,
    ) -> bool:
    """
    Removes per-run simulation files from the Cooja container: the generated
    build artifacts (GENERATED_ARTIFACTS) plus the uploaded remote_files.
    Best-effort: failures are logged as warnings and reported via the return
    value so cleanup never aborts the simulation flow.
    """
    command = build_cleanup_command(remote_dir, remote_files)
    try:
        _, stdout, stderr = client.exec_command(command, timeout=timeout)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            err = stderr.read().decode("utf-8", errors="ignore").strip()
            log.warning(f"Remote cleanup exited with status {exit_status}: {err}")
            return False
        log.info(f"Remote cleanup done in {remote_dir}")
        return True
    except Exception as ex:
        log.warning(f"Remote cleanup failed: {ex}")
        return False