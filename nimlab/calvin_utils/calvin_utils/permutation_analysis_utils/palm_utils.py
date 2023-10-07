from __future__ import print_function
import pandas as pd
import numpy as np
import patsy
import re
import os
import io
import os
import shutil
import os.path
from time import time
import subprocess
import getpass
from termcolor import cprint
from nilearn import image
from nimlab import datasets as ds
from nimlab import configuration as config
import pandas as pd
from IPython.core.getipython import get_ipython
try:
    from pathlib import Path
except:
    from pathlib2 import Path

class PalmPreparation:
    """
    A class to generate design matrices and process NIFTI file paths.
    """
    
    def __init__(self, mask="mni_icbm152", out_dir=None):
        """
        Initialize the PalmPreparation class.
        
        Parameters:
        - mask: String, name of the default mask. Defaults to "mni_icbm152".
        - out_dir: String, path to the output directory. If None, defaults to the current directory.
        """
        self.DEFAULT_MASK = mask if mask else "mni_icbm152"
        self.out_dir = out_dir if out_dir else os.getcwd()
            
        if self.out_dir:
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

    
    def extract_subject_id(self, file_path):
        """
        Extract subject ID from a given file path.

        Parameters:
        - file_path: String, path to the NIFTI file.

        Returns:
        - String representing the extracted subject ID.
        """
        match = re.search(r'(subject\d+|sub-\d+)', file_path)
        if match:
            return ''.join(filter(str.isdigit, match.group()))
        else:
            file_name = file_path.split('/')[-1]
            match = re.search(r'\d+', file_name)
            return match.group() if match else os.path.basename(file_name)
    
    def process_nifti_paths(self, csv_file_path):
        """
        Process a CSV file containing NIFTI paths to extract subject IDs.

        Parameters:
        - csv_file_path: String, path to the CSV file containing NIFTI paths.

        Returns:
        - List of NIFTI paths and a DataFrame containing NIFTI paths and associated subject IDs.
        """
        self.nifti_paths_df = pd.read_csv(csv_file_path, header=None, names=['nifti_path'])
        self.nifti_paths_df['subject_id'] = self.nifti_paths_df['nifti_path'].apply(self.extract_subject_id)
        self.ordered_nifti_list = self.nifti_paths_df['nifti_path'].tolist()
        return self.nifti_paths_df,  self.ordered_nifti_list

    def create_design_matrix(self, formula_vars=None, data_df=None, subject_id_column=None, subject_ids=None):
        """
        Create a design matrix based on given formula variables and data.

        Parameters:
        - formula_vars: List of strings, each representing a variable in the design formula.
        - data_df: DataFrame, data to create the design matrix from.
        - subject_id_column: String, column in data_df containing the subject ID.
        - subject_ids: List of subject IDs, used if formula_vars is None.

        Returns:
        - DataFrame representing the design matrix.
        """
        if formula_vars is None:
            if subject_ids is not None:
                design_matrix = pd.DataFrame({'Intercept': np.ones(len(subject_ids))}, index=subject_ids)
                design_matrix.index.rename('subject_id', inplace=True)
                return design_matrix
            else:
                raise ValueError("No subject IDs provided. Cannot create design matrix with just intercept.")
        
        formula = ' + '.join(formula_vars)
        design_matrix = patsy.dmatrix(formula, data_df, return_type='dataframe')
        design_matrix.index = data_df[subject_id_column]
        design_matrix.index.rename('subject_id', inplace=True)
        
        self.design_matrix=design_matrix
        return self.design_matrix

    def generate_basic_contrast_matrix(self, design_matrix):
        """
        Generate a basic contrast matrix based on the design matrix and display it for potential user modification.

        Parameters:
        - design_matrix: DataFrame, the design matrix.

        Returns:
        - 2D NumPy array representing the default contrast matrix.
        """
        contrast_matrix = np.eye(len(design_matrix.columns), len(design_matrix.columns))
        for i in range(1, len(contrast_matrix)):
            contrast_matrix[i, 0] = -1
        contrast_df = pd.DataFrame(data=contrast_matrix, columns=design_matrix.columns)
        
        print("This is a basic contrast matrix set up to evaluate the significance of each variable.")
        print("Copy it into a cell below and edit it for more control over your analysis.")
        print(contrast_df)
        
        return contrast_matrix

    def save_contrast_matrix_to_csv(self, design_matrix, contrast_matrix, file_name='contrast_matrix.csv'):
        """
        Save the contrast matrix to a CSV file.

        Parameters:
        - contrast_matrix: 2D NumPy array, the contrast matrix.
        - out_dir: String, the output directory.
        - file_name: String, the name of the CSV file to save.

        Returns:
        - None
        """
        contrast_df = pd.DataFrame(data=contrast_matrix, columns=design_matrix.columns)
        contrast_df.to_csv(os.path.join(self.out_dir, file_name), index=False)
        print('Contrast matrix saved to: ', os.path.join(self.out_dir, file_name))
        return contrast_df
        
    def example_exchangability_block(self):
        # under work
        ### This is just an example, you will have to edit to adapt to your data, 
        ### but it should be integers, starting with 1,2,3....

        # coding_key = {"Prosopagnosia_w_Yeo1000": 1,
        #              "Corbetta_Lesions": 1,
        #              "DBS_dataset": 2
        #              }

        # eb_matrix = pd.DataFrame()
        # eb_matrix = clean_df['dataset'].replace(coding_key)
        # display(eb_matrix)
        return None
    
    def build_cluster_submit_string(
        self,
        directory,
        cluster_name,
        username,
        cmd,
        dryrun=False,
        queue="",
        cores="",
        memory="",
        job_name="",
        job_time="",
        num_nodes="",
        num_tasks="",
        x11_forwarding="",
        service_class="",
        cluster_email="",
        debug=False,
        extra="",
    ):
        """Build a job submission script to a cluster and returns a command string.

        Args:
            directory (str): Path to output job script to.
            cluster_name (str): Name of cluster to submit job to.
            username (str): Username for ssh login to cluster.
            cmd (str): Command string to run on cluster.
            dryrun (bool): If True, does not submit to cluster and prints out submit command. Defaults to False.
            queue (str): Job queue/partition to submit to.
            cores (str): Number of requested cores.
            memory (str): Amount of requested memory (in MB).
            job_name (str): Job name.
            job_time (str): Total runtime limit.
            num_nodes (str): Number of nodes to run on. Typically will be 1.
            num_tasks (str): Number of tasks to run. Typically will be 1.
            x11_forwarding (str): Options for X11 forwarding. On LSF will activate -XF flag. On Slurm will enable options for x11 forwarding such as -xf=batch.
            service_class (str): Service class to run on.
            cluster_email (str): Email of user for job status notifications.
            debug (bool): If True, adds debug scripts to job script. Defaults to False.
            extra (str): If not empty, adds extra scripts to job script.

        Raises:
            ValueError: Unrecognized cluster
        """

        if cluster_name in config.clusters.keys():
            cluster_config = config.clusters[cluster_name]
            job_script_path = os.path.abspath(
                os.path.join(directory, f"job_script_{int(time())}.sh")
            )
            sshpass_prefix = [
                "sshpass",
                "-p",
                getpass.getpass(prompt="Please type password"),
                "ssh",
                username + "@" + cluster_config["hostname"],
            ]

            if dryrun:
                print(f"{cluster_config['submit-command']} {job_script_path}")

            else:
                sshpass_prefix.append("'echo")
                sshpass_prefix.append('"' + cluster_config["submit-command"])
                sshpass_prefix.append(job_script_path + '";')
                sshpass_prefix.append(cluster_config["submit-command"])
                sshpass_prefix.append(job_script_path + "'")

            job_script = ["#!/bin/bash"]
            if job_name:
                job_script.append(cluster_config["job-name"].replace("&", job_name))
            if queue:
                job_script.append(cluster_config["queue-name"].replace("&", queue))
            if num_nodes:
                job_script.append(cluster_config["num-nodes"].replace("&", num_nodes))
            if num_tasks:
                job_script.append(cluster_config["num-tasks"].replace("&", num_tasks))
            if cores:
                job_script.append(cluster_config["cores"].replace("&", cores))
            if memory:
                job_script.append(cluster_config["memory"].replace("&", memory))
            job_script.append(cluster_config["standard-output"])
            job_script.append(cluster_config["error-output"])
            if job_time:
                job_script.append(cluster_config["job-time"].replace("&", job_time))
            if x11_forwarding:
                job_script.append(
                    cluster_config["x11-forwarding"].replace("&", x11_forwarding)
                )
            if service_class:
                job_script.append(
                    cluster_config["service-class"].replace("&", service_class)
                )
            if cluster_email:
                job_script.append(
                    cluster_config["mail-address"].replace("&", cluster_email)
                )
                job_script.append(cluster_config["mail-type"])
            job_script.append("")
            job_script.append(cluster_config["environment-setup"])
            if debug:
                job_script.append("")
                job_script.append(cluster_config["debug"])
            if extra:
                job_script.append("")
                job_script.append(extra)
            job_script.append("")
            job_script.append(" ".join(cmd))
            with open(job_script_path, "w") as f:
                for item in job_script:
                    f.write("%s\n" % item)

            return sshpass_prefix

        else:
            raise ValueError(
                f"Cluster option '{cluster_name}' not recognized! Available options are {list(config.clusters.keys())}"
            )
            
    def text2vest(self, input_file, output_file):
        config.verify_software(["fsl_path"])
        process = subprocess.Popen(
            [f"{config.software['fsl_path']}/bin/Text2Vest", input_file, output_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, stderr = process.communicate()
        if stdout:
            raise Exception("Error in text2vest: " + stdout)
            
    def calvins_call_palm(
        self, input_imgs, design_matrix, contrast_matrix, working_directory=None, output_directory=None, iterations=10000, voxelwise_evs=None,
        eb=None, mask="", save_1p=True, logp=False, tfce=False, ise_flag=False,
        two_tailed_flag=True, corrcon_flag=False, fdr_flag=False, accel="", cluster_name="",
        username="", cluster_email="", queue="normal", cores=1, memory=6000,
        dryrun=False, job_name="", job_time="", num_nodes="", num_tasks="", x11_forwarding="",
        service_class="", debug=False,extra="",
        ):
        
        """Call FSL PALM https://github.com/andersonwinkler/PALM

        Parameters
        ----------
        input_imgs : list of Nifti-like objects
            Input images to give to PALM. Typically network maps
        design_matrix : pd.DataFrame
            Design matrix
        contrast_matrix : pd.DataFrame
            Contrast matrix
        working_directory : str
            Path to working directory where PALM config files are written
        output_directory : str
            Path to output directory
        iterations : int
            Number of permutations to run
        voxelwise_evs : list of tuples
            Each tuple contains: 
                file: The file with one voxelwise EV.
                evpos: The column number (position) of this EV in the design matrix.
        eb : pd.DataFrame, optional
            Dataframe specifying exchangeability block membership. Defaults to None
        mask : str
            Path to mask file. Defaults to "MNI152_T1_2mm_brain_mask_dil" provided
            by nimlab.datasets
        save_1p : bool
            Save p values as 1 - p. Defaults to True.
        logp : bool
            -logp
            Save the output p-values as -log10(p). Defaults to False
        tfce : bool
            Generate tfce output. Defaults to False.
        ise_flag : bool
            Generate ISE output. Defaults to False
        two_tailed_flag : bool
            Run as two tailed test. Defaults to True.
        corrcon_flag : bool
            Multiple comparisons correction across contrasts. Defaults to False.
        fdr_flag : bool
            Generate FDR output. Defaults to False.
        accel : str
            Acceleration method. Defaults to none.
        cluster_submit : str
            Specify cluster to submit job to, if any. Current options are "erisone, dryrun, eristwo".
        cluster_user: str
            Username on cluster
        Returns
        -------
        None

        """
        # prep folders
        working_directory = os.path.join(self.out_dir, "palm_config")
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)
            
        output_directory = os.path.join(self.out_dir, "palm_results")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            
            
        config.verify_software(["palm_path"])
        # Concatenate input files
        print("concatenating input...")
        concat_file = os.path.abspath(working_directory) + "/concat.nii"
        image.concat_imgs(input_imgs).to_filename(concat_file)
        output_directory
        # Create and convert design and contrast matrices
        design_matrix_file = working_directory + "/design.mat"
        contrast_matrix_file = working_directory + "/contrast.con"
        design_matrix.to_csv(
            working_directory + "/design.tsv", header=False, index=False, sep="\t"
        )
        contrast_matrix.to_csv(
            working_directory + "/contrast.tsv", header=False, index=False, sep="\t"
        )
        self.text2vest(working_directory + "/design.tsv", design_matrix_file)
        self.text2vest(working_directory + "/contrast.tsv", contrast_matrix_file)
        if mask == "":
            mask_file = working_directory + "/" + self.DEFAULT_MASK + ".nii"
            ds.get_img(self.DEFAULT_MASK).to_filename(mask_file)
            mask = mask_file

        # Create exchangeability blocks
        if eb is not None:
            eb_file = working_directory + "/eb.csv"
            eb.to_csv(eb_file, header=False, index=False)

        # Required argument
        palm_cmd = [
            f"{config.software['palm_path']}/palm",
            "-i",
            os.path.abspath(concat_file),
            "-o",
            os.path.abspath(output_directory) + "/",
            "-d",
            os.path.abspath(design_matrix_file),
            "-t",
            os.path.abspath(contrast_matrix_file),
            "-n",
            str(iterations),
            "-m",
            os.path.abspath(mask),
        ]

        # Optional arguments
        if eb is not None:
            palm_cmd += ["-eb", os.path.abspath(eb_file), "-vg", "auto"]
        if save_1p:
            palm_cmd += ["-save1-p"]
        if logp:
            palm_cmd += ["-logp"]
        if tfce:
            palm_cmd += ["-T"]
        if ise_flag:
            palm_cmd += ["-ise"]
        if two_tailed_flag:
            palm_cmd += ["-twotail"]
        if corrcon_flag:
            palm_cmd += ["-corrcon"]
        if fdr_flag:
            palm_cmd += ["-fdr"]
        if accel:
            palm_cmd += ["-accel", accel]
        # Add -evperdat argument for voxelwise EVs
        if voxelwise_evs is not None:
            for voxelwise_ev in voxelwise_evs:
                file, evpos = voxelwise_ev
                palm_cmd += ["-evperdat", file, str(evpos)]
        
        print("Calling PALM with following command:")
        print(" ".join(palm_cmd))
        start = time()
        if cluster_name == "":
            cmd = palm_cmd
        else:
            cmd = self.build_cluster_submit_string(
                directory=output_directory,
                cluster_name=cluster_name,
                username=username,
                cmd=palm_cmd,
                dryrun=dryrun,
                queue=queue,
                cores=cores,
                memory=memory,
                job_name=job_name,
                job_time=job_time,
                num_nodes=num_nodes,
                num_tasks=num_tasks,
                x11_forwarding=x11_forwarding,
                service_class=service_class,
                cluster_email=cluster_email,
                debug=debug,
                extra=extra,
            )

        if not dryrun:
            ipython = get_ipython()
            ipython.system(" ".join(cmd))
        end = time()

        print("\n")

        print("Time elapsed: " + str(round(end - start)) + " seconds")