import paramiko
from scp import SCPClient

def create_ssh_client(hostname, port, username, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port=port, username=username, password=password)
    return client

def send_files_scp(client, local_path, remote_path, source_files, target_files):
    if len(source_files) != len(target_files):
        print("The number of source files is not equal number of targets.")
        return
    with SCPClient(client.get_transport()) as scp:
        for src, dest in zip(source_files, target_files):
            local_file_path = local_path + "/" + src
            remote_file_path = remote_path + "/" + dest
            print(f"Sending {local_file_path} to {remote_file_path}")
            scp.put(local_file_path, remote_file_path)  