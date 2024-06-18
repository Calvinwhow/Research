import os
import paramiko
import getpass
import shutil

class LocalTransfer:
    """
    LocalTransfer class provides a convenient way to transfer files within the local filesystem.

    Methods:
        transfer_files_in_list(local_files, dest_path):
            Transfers a list of local files to the specified destination path.

        transfer_files_in_dict(dict_files, base_dest_path):
            Transfers files specified in a dictionary to new subdirectories in a given base directory.
            
    Example usage:
    dict_files_to_transfer = {
        "outcomes": ["outcome_data_1.csv", "outcome_data_2.csv"],
        "covariates": ["covariate_data_1.csv"],
        "voxelwise": ["voxelwise_data_1.csv", "voxelwise_data_2.csv"]
    }
    local_base_path = "/path/to/destination/folder"

    local_transfer = LocalTransfer()
    local_transfer.transfer_files_in_dict(dict_files_to_transfer, local_base_path)

    """

    @staticmethod
    def transfer_files_in_list(local_files, dest_path):
        """
        Transfers a list of local files to the specified destination path.

        Parameters:
            local_files (list): A list of file paths on the local computer to transfer.
            dest_path (str): The destination path on the local filesystem where the files will be transferred.

        Raises:
            FileNotFoundError: If a local file does not exist.
            OSError: If there is an issue creating the destination directory or copying files.
        """
        try:
            # Create the destination path if it doesn't exist
            os.makedirs(dest_path, exist_ok=True)

            # Transfer each local file to the destination path
            for local_file in local_files:
                if not os.path.isfile(local_file):
                    raise FileNotFoundError(f"The file {local_file} does not exist.")
                dest_file_path = os.path.join(dest_path, os.path.basename(local_file))
                shutil.copy2(local_file, dest_file_path)
                print(f'Successfully transferred {local_file} to {dest_file_path}.')

            print("Files transferred successfully.")

        except Exception as e:
            print(f"Error occurred: {e}")

    @staticmethod
    def transfer_files_in_dict(dict_files, base_dest_path):
        """
        Transfers files specified in a dictionary to new subdirectories in a given base directory.

        Parameters:
            dict_files (dict): A dictionary where the key is the subdirectory name, and the value is a list of file paths to transfer.
            base_dest_path (str): The base path on the local filesystem where the new subdirectories will be created.

        Raises:
            FileNotFoundError: If a local file does not exist.
            OSError: If there is an issue creating the destination directory or copying files.
        """
        try:
            for subdir, local_files in dict_files.items():
                # Create the destination subdirectory if it doesn't exist
                dest_subdir_path = os.path.join(base_dest_path, subdir)
                os.makedirs(dest_subdir_path, exist_ok=True)

                # Transfer each local file to the new subdirectory
                for local_file in local_files:
                    if not os.path.isfile(local_file):
                        raise FileNotFoundError(f"The file {local_file} does not exist.")
                    dest_file_path = os.path.join(dest_subdir_path, os.path.basename(local_file))
                    shutil.copy2(local_file, dest_file_path)
                    print(f'Successfully transferred {local_file} to {dest_file_path}.')

            print("All files transferred successfully.")

        except Exception as e:
            print(f"Error occurred: {e}")

class ScpTransfer:
    """
    ScpTransfer class provides a convenient way to transfer files to a remote server
    using SCP (Secure Copy).

    Parameters:
        hostname (str): The hostname or IP address of the remote server.
        username (str): The username to use for authentication.
        password (str, optional): The password for password-based authentication. Default is None.
        ssh_key (str, optional): The path to the SSH private key file for key-based authentication. Default is None.

    Attributes:
        hostname (str): The hostname or IP address of the remote server.
        username (str): The username to use for authentication.
        password (str or None): The password for password-based authentication. None if using key-based authentication.
        ssh_key (str or None): The path to the SSH private key file for key-based authentication. None if using password-based authentication.

    Methods:
        transfer_files(local_files, remote_path):
            Transfers a list of local files to the specified path on the remote server using SCP.

    Example:
        # Example usage
        dict_files_to_transfer = {
            "outcomes": ["outcome_data_1.csv", "outcome_data_2.csv"],
            "covariates": ["covariate_data_1.csv"],
            "voxelwise": ["voxelwise_data_1.csv", "voxelwise_data_2.csv"]
        }
        remote_base_path = "/path/on/remote/server"
        remote_hostname = "example.com"
        remote_username = "your_username"
        remote_password = "your_password"  # Or use remote_ssh_key="path/to/your/ssh_key" if using an SSH key

        scp_transfer = ScpTransfer(remote_hostname, remote_username, remote_password)
        scp_transfer.transfer_files_in_dict(dict_files_to_transfer, remote_base_path)
    """

    def __init__(self, hostname, username, ssh_key=None):
        """
        Initialize the ScpTransfer object.

        Parameters:
            hostname (str): The hostname or IP address of the remote server.
            username (str): The username to use for authentication.
            password (str, optional): The password for password-based authentication. Default is None.
            ssh_key (str, optional): The path to the SSH private key file for key-based authentication. Default is None.
        """
        self.hostname = hostname
        self.username = username
        if ssh_key is None:
            self.password = getpass.getpass(prompt='Input your ssh password here: ')
        self.ssh_key = ssh_key

    def _connect_to_server(self):
        """
        Connect to the remote server using SSH and return the SSHClient object.

        Returns:
            paramiko.SSHClient: The SSHClient object used for the SSH connection.
        """
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        if self.ssh_key is not None:
            client.connect(self.hostname, username=self.username, key_filename=self.ssh_key)
        else:
            client.connect(self.hostname, username=self.username, password=self.password)

        return client

    def transfer_files_in_list(self, local_files, remote_path):
        """
        Transfers a list of local files to the specified path on the remote server using SCP.

        Parameters:
            local_files (list): A list of file paths on the local computer to transfer.
            remote_path (str): The destination path on the remote server where the files will be transferred.
        """
        try:
            client = self._connect_to_server()

            # Create the destination path on the server if it doesn't exist
            stdin, stdout, stderr = client.exec_command(f'mkdir -p {remote_path}')
            stdout.channel.recv_exit_status()

            # Transfer each local file to the server
            sftp = client.open_sftp()
            for local_file in local_files:
                local_file_path = os.path.abspath(local_file)
                remote_file_path = os.path.join(remote_path, os.path.basename(local_file))
                sftp.put(local_file_path, remote_file_path)

            sftp.close()
            print("Files transferred successfully.")

        except paramiko.AuthenticationException:
            print("Authentication failed. Please check your username, password, or SSH key.")
        except paramiko.SSHException as e:
            print(f"SSH error occurred: {e}")
        except Exception as e:
            print(f"Error occurred: {e}")

        finally:
            client.close()
            
    def transfer_files_in_dict(self, dict_files, base_remote_path):
        """
        Transfers files specified in a dictionary to new subdirectories in a given base directory on the remote server.

        Parameters:
            dict_files (dict): A dictionary where the key is the subdirectory name, and the value is a list of file paths to transfer.
            base_remote_path (str): The base path on the remote server where the new subdirectories will be created.
        """
        client = self._connect_to_server()

        sftp = client.open_sftp()
        
        dict_files_remote = {}
        for subdir, local_files in dict_files.items():
            # Create the destination subdirectory on the server
            remote_subdir_path = os.path.join(base_remote_path, subdir)
            stdin, stdout, stderr = client.exec_command(f'mkdir -p {remote_subdir_path}')
            stdout.channel.recv_exit_status()

            remote_files = []
            # Transfer each local file to the new subdirectory on the server
            for local_file in local_files:
                local_file_path = os.path.abspath(local_file)
                remote_file_path = os.path.join(remote_subdir_path, os.path.basename(local_file))
                sftp.put(local_file_path, remote_file_path)
                print(f'Successfully transferred {local_file_path} to {remote_file_path}.')
                remote_files.append(remote_file_path)
                
            dict_files_remote[subdir] = remote_files
        sftp.close()
        client.close()
        return dict_files_remote