"""
Module: Job Scheduler

This module provides classes to handle job submission to different servers with various job schedulers.

Classes:
Server: A class to represent a generic server.
Job: A class to represent a generic job.
JobSubmitter: A class to handle the job submission.
LSFServer: A subclass of Server specific to the LSF job scheduler.
LSFJob: A subclass of Job specific to the LSF job scheduler.

Example:
from jobscheduler import LSFServer, LSFJob, JobSubmitter

Initialize server
server = LSFServer("erisone.partners.org", "cu135")

Initialize job
job = LSFJob(
job_name="f_test_bm",
user_email="choward12@bwh.harvard.edu",
cpus=1,
output_dir="/PHShome/cu135/terminal_outputs",
error_dir="/PHShome/cu135/error_outputs",
queue="big-multi",
script_path="/PHShome/cu135/python_scripts/launch_f_test_palm.py",
options="-o /PHShome/cu135/permutation_tests/f_test/age_by_stim_pd_dbs/results",
Gb_requested=4,
wait_time=0
)

Initialize job submitter
job_submitter = JobSubmitter(server, job)

Submit jobs
job_submitter.submit_jobs(10)

This example will log into the server 'erisone.partners.org' as user 'cu135', and it will submit 10 jobs named 'f_test_bm' to the queue 'big-multi'. Each job will run the Python script at '/PHShome/cu135/python_scripts/launch_f_test_palm.py' with the options specified, outputting to '/PHShome/cu135/terminal_outputs' and '/PHShome/cu135/error_outputs' for terminal and error logs respectively. The job requests 1 CPU and 4GB of memory.
"""

import os
import getpass
import paramiko

class Server:
    """
    A class to represent a server.
    """

    def __init__(self, server_name, username):
        self.server_name = server_name
        self.username = username
        self.password = getpass.getpass(prompt="Please type password: ")

class Job:
    """
    A class to represent a job.
    """

    def __init__(self, job_name, user_email, cpus, output_dir, error_dir, queue, script_path, options, wait_time):
        self.job_name = job_name
        self.cpus = cpus
        self.user_email = user_email
        self.output_dir = output_dir
        self.error_dir = error_dir
        self.queue = queue
        self.script_path = script_path
        self.options = options
        self.work_dir = os.path.dirname(script_path)
        self.wait_time = wait_time


class JobSubmitter:
    """
    A class to handle the job submission.
    """

    def __init__(self, server, job):
        self.server = server
        self.job = job
        self.submitted = False

    def submit_jobs(self, print_job=True):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(self.server.server_name, username=self.server.username, password=self.server.password)
            
            job_command = self.job.construct_command()
            stdin, stdout, stderr = ssh.exec_command(job_command)
            if print_job:
                print("Job command: ", job_command)
            # Capture and print any output
            for line in stdout:
                print(line.strip('\n'))
            
            ssh.close()
            self.submitted = True
        except Exception as e:
            print(f"Failed to submit jobs due to error: {e}")
            self.submitted = False
        return print('Job submitted successfully: ', self.submitted)

class LSFServer(Server):
    """
    A subclass of Server specific to the LSF job scheduler.
    """
    def ssh(self):
        """
        Returns the SSH command prefix for this server.
        
        Args:
            password (str): The password to login to the server.
        """
        ssh_prefix = ["sshpass", "-p", self.password, "ssh", f"{self.username}@{self.server_name}"]
        return ssh_prefix


class LSFJob(Job):
    """
    A subclass of Job specific to the LSF job scheduler.

    Inherits all attributes and methods from the Job class.

    Overridden Methods:
        construct_command(n_jobs): Returns the job command specific to the LSF job scheduler.
    """
        
    def __init__(self, gb_requested, n_jobs, environment_activation_string=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mem_limit = gb_requested * 1000  # Convert to ~MB (actually *1024)
        self.resource_req = f"rusage[mem={self.mem_limit}] span[ptile={self.cpus}]" 
        self.n_jobs = n_jobs
        self.env_activate = environment_activation_string
            
    def construct_command(self):
        """
        Returns the job command specific to the LSF job scheduler.
        """
        activation_string = f"source ~/.bashrc && {self.env_activate} && " if self.env_activate else ""
        if self.n_jobs:  # If n_jobs is defined
            job_command = f"bsub -q {self.queue} -n {self.cpus} -R '{self.resource_req}' -M {self.mem_limit} -o {self.output_dir}/{self.job_name}_%I.txt -J '{self.job_name}[1-{self.n_jobs}]' -u {self.user_email} -B -N -cwd {self.work_dir} {activation_string} python {self.script_path} {self.options if self.options else ''}"
        else:  # If n_jobs is not defined
            job_command = f"bsub -q {self.queue} -n {self.cpus} -R '{self.resource_req}' -M {self.mem_limit} -o {self.output_dir}/{self.job_name}.txt -J '{self.job_name}' -u {self.user_email} -B -N -cwd {self.work_dir} {activation_string} python {self.script_path} {self.options if self.options else ''}"
        return job_command







