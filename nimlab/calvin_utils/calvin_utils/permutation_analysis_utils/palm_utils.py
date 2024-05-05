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
import nibabel as nib
from nilearn import image
from nimlab import datasets as ds
from nimlab import configuration as config
import pandas as pd
from IPython.core.getipython import get_ipython
try:
    from pathlib import Path
except:
    from pathlib2 import Path

import pandas as pd
import os
import subprocess
import numpy as np
import patsy
from calvin_utils.file_utils.dataframe_utilities import preprocess_colnames_for_regression

from nilearn import image
from nimlab import datasets as nimds
import nibabel as nib
import numpy as np

class CalvinPalm:
    """
    Class for handling PALM analysis workflow.
    """
    def __init__(self, input_csv_path, output_dir, sheet=None):
        """
        Initialize the CalvinPalm class with input and output paths.

        Parameters:
        - input_csv_path: str, path to the input CSV file
        - output_dir: str, path to the output directory
        """
        self.input_csv_path = input_csv_path
        self.sheet = sheet
        self.output_dir = output_dir
        self.df = None
        self.design_matrix = None
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def read_data(self):
        """
        Reads data from a CSV file and preprocesses the column names for regression.

        Returns:
        - DataFrame: Preprocessed data read from the CSV file.
        """
        if self.sheet is not None:
            try:
                df = pd.read_excel(self.input_csv_path, sheet_name=self.sheet)
            except Exception as e:
                print(f'Error: {e}')
        else:
            df = pd.read_csv(self.input_csv_path)
        self.df = preprocess_colnames_for_regression(df)
        return self.df
    
    def handle_interactions(self, design_matrix, formula_vars):
        """
        Manually handle interaction terms in the DataFrame.
        """
        if formula_vars is None:
            return design_matrix
        
        for var in formula_vars:
            if '*' in var or ':' in var:
                interacting_vars = var.replace('*', ':').split(':')
                new_col = design_matrix[interacting_vars].prod(axis=1)
                design_matrix[var] = new_col
        return design_matrix

    def create_design_matrix(self, formula_vars=None, data_df=None, intercept=True):
        """
        Create a design matrix based on given formula variables and data.

        Parameters:
        - formula_vars: List of strings, each representing a variable in the design formula.
        - data_df: DataFrame, data to create the design matrix from.
        - intercept: Boolean, whether to add an intercept term if 'Intercept' is not in formula_vars.

        Returns:
        - DataFrame representing the design matrix.
        """
        if formula_vars is None or len(formula_vars) == 0:
            design_matrix = pd.DataFrame({'Intercept': np.ones(len(data_df.index))}, index=data_df.index)
        else:
            formula = ' + '.join(formula_vars)
            design_matrix = patsy.dmatrix(formula, data_df, return_type='dataframe')

        # Handle intercept
        if intercept and 'Intercept' not in map(str.capitalize, formula_vars):
            design_matrix['Intercept'] = 1.0
        if intercept==False:
            try:
                self.design_matrix.pop('Intercept')
            except:
                print('Could not remove Intercept.')

        # Handle interactions
        design_matrix = self.handle_interactions(design_matrix, formula_vars)

        # Set index
        if 'ID' in data_df.columns:
            design_matrix.index = data_df['ID']
        else:
            design_matrix.index = data_df.index
        design_matrix.index.rename('id', inplace=True)

        # Preprocess column names (Assuming this is a function you have)
        self.design_matrix = preprocess_colnames_for_regression(design_matrix)


        return self.design_matrix

    def set_dependent_variable(self, dep_var_column):
        """
        Set the dependent variable for the model.

        Parameters:
        - dep_var_column: String, the column name in the DataFrame that contains the dependent variable (paths to NIFTI files).

        Returns:
        - Series representing the dependent variable.
        """
        if dep_var_column not in self.df.columns:
            raise ValueError(f"{dep_var_column} is not a valid column in the DataFrame.")

        self.dep_var_series = self.df[dep_var_column]
        return self.dep_var_series
            
    def generate_voxelwise_cov_4d_nifti(self, X, voxelwise_variable, absval=False):
        """
        Generate a 4D Nifti file for each component of the formula.

        Parameters:
        - X (df): the design matrix
        - voxelwise_variable (str): the voxelwise variable in the design matrix
        - absval: Boolean, whether to take the absolute value of the 4D image.

        Returns:
        - String, the path to the generated 4D Nifti file.
        """
        # Prepare directory
        save_dir = os.path.join(self.output_dir, '4d_niftis')
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare the voxelwise 4D nifti
        nifti_imgs = [image.load_img(file_path) for file_path in X[voxelwise_variable]]
        data_4d = image.concat_imgs(nifti_imgs)
            
        if absval:
            data_4d = image.math_img("np.abs(img)", img=data_4d)

        output_path = os.path.join(save_dir, f'{voxelwise_variable}.nii')
        data_4d.to_filename(output_path)

        return output_path
    
    def generate_univariate_4d_niftis(self, X, voxelwise_variable, absval=False):
        """
        Generate a 4D Nifti file for each covariate excluding the voxelwise variable,
        with each 3D volume in the 4D Nifti filled with a scalar covariate value.

        Parameters:
        - X (DataFrame): The design matrix with covariates in columns and patients in rows.
        - voxelwise_variable (str): The column name of the voxelwise images in the DataFrame.
        - absval (bool): Whether to take the absolute value of the generated 4D Nifti image.

        Returns:
        - List of paths to the generated 4D Nifti files.
        """
        # Prepare directory
        save_dir = os.path.join(self.output_dir, '4d_niftis')
        os.makedirs(save_dir, exist_ok=True)
        
        # Load a reference 3D Nifti image to get the shape
        reference_img_path = X.loc[0, voxelwise_variable]  # Assuming the first row can be used as reference
        reference_img = image.load_img(reference_img_path)

        # Prepare a list to store paths of the generated Nifti files
        nifti_paths = []

        # Loop over each covariate in the design matrix
        for covariate in X.columns:
            if covariate == voxelwise_variable:
                continue  # Skip the voxelwise variable

            # Initialize an empty 4D array with the same spatial dimensions as the reference image,
            # but with depth equal to the number of patients
            num_patients = X.shape[0]
            covariate_data_4d = np.zeros(reference_img.shape + (num_patients,))

            # Fill each slice of the 4D array with the covariate value for each patient
            for idx, value in enumerate(X[covariate].values):
                covariate_data_4d[..., idx] = value  # Fill the entire volume slice with the covariate value

            # Create a 4D Nifti image from the filled data
            covariate_nifti = image.new_img_like(reference_img, covariate_data_4d)

            # Optionally take the absolute value of the 4D image
            if absval:
                covariate_nifti = image.math_img("np.abs(img)", img=covariate_nifti)

            # Generate the output path and save the 4D Nifti file
            output_path = os.path.join(save_dir, f'{covariate}.nii')
            covariate_nifti.to_filename(output_path)

            # Add the path to the list of generated Nifti files
            nifti_paths.append(output_path)

        return nifti_paths

    def generate_4d_dependent_variable_nifti(self, Y, X, voxelwise_variable, absval=False):
        """
        Generate a 4D Nifti file for each covariate excluding the voxelwise variable,
        with each 3D volume in the 4D Nifti filled with a scalar covariate value.

        Parameters:
        - Y (DataFrame): The outcome matrix with patients in rows. 
        - X (DataFrame): The design matrix with covariates in columns and patients in rows.
        - voxelwise_variable (str): The column name of the voxelwise images in the DataFrame.
        - absval (bool): Whether to take the absolute value of the generated 4D Nifti image.

        Returns:
        - List of paths to the generated 4D Nifti files.
        """
        # Prepare directory
        save_dir = os.path.join(self.output_dir, '4d_niftis')
        os.makedirs(save_dir, exist_ok=True)
        
        # Load a reference 3D Nifti image to get the shape
        reference_img_path = X.loc[0, voxelwise_variable]  # Assuming the first row can be used as reference
        reference_img = image.load_img(reference_img_path)

        # Prepare a list to store paths of the generated Nifti files
        nifti_paths = []

        # Loop over each covariate in the design matrix
        for covariate in Y.columns:
            if covariate == voxelwise_variable:
                continue  # Skip the voxelwise variable

            # Initialize an empty 4D array with the same spatial dimensions as the reference image,
            # but with depth equal to the number of patients
            num_patients = X.shape[0]
            covariate_data_4d = np.zeros(reference_img.shape + (num_patients,))

            # Fill each slice of the 4D array with the covariate value for each patient
            for idx, value in enumerate(Y[covariate].values):
                covariate_data_4d[..., idx] = value  # Fill the entire volume slice with the covariate value

            # Create a 4D Nifti image from the filled data
            covariate_nifti = image.new_img_like(reference_img, covariate_data_4d)

            # Optionally take the absolute value of the 4D image
            if absval:
                covariate_nifti = image.math_img("np.abs(img)", img=covariate_nifti)

            # Generate the output path and save the 4D Nifti file
            output_path = os.path.join(save_dir, f'{covariate}.nii')
            covariate_nifti.to_filename(output_path)

            # Add the path to the list of generated Nifti files
            nifti_paths.append(output_path)

        return nifti_paths

    def generate_basic_contrast_matrix(self, design_matrix, compare_to_intercept=False):
        """
        Generate a basic contrast matrix based on the design matrix and display it for potential user modification.

        Parameters:
        - design_matrix: DataFrame, the design matrix.

        Returns:
        - 2D NumPy array representing the default contrast matrix.
        """
        contrast_matrix = np.eye(len(design_matrix.columns), len(design_matrix.columns), dtype=int)
        if compare_to_intercept:
            for i in range(1, len(contrast_matrix)):
                contrast_matrix[i, 0] = -1
                
        contrast_df = pd.DataFrame(data=contrast_matrix, columns=design_matrix.columns)
        
        # Convert DataFrame to a numpy array and then to a list of lists for clean printing
        contrast_matrix_list = contrast_df.values.tolist()
        print('Here is a basic contrast matrix set up to evaluate the significance of each variable.')
        print('Here is an example of what your contrast matrix looks like as a dataframe: ')
        display(contrast_df)
        
        print("Below is the same contrast matrix, but as an array.")
        print("Copy it into a cell below and edit it for more control over your analysis.")
        print("[")
        for row in contrast_matrix_list:
            print("    " + str(row) + ",")
        print("]")
        return contrast_matrix


    def finalize_contrast_matrix(self, design_matrix, contrast_matrix):
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
        return contrast_df


class CalvinPalmSubmitter:
    def calvins_call_palm(self, dv_nifti_file_path, design_matrix, contrast_matrix, 
                        working_directory=None, output_directory=None, iterations=10000,
                        accel="tail", voxelwise_evs=None, eb=None, mask="", save_1p=True, 
                        logp=False, tfce=False, ise_flag=False, two_tailed_flag=True,
                        corrcon_flag=False, fdr_flag=False, cluster_name="",
                        username="", cluster_email="", queue="normal", cores="1", 
                        memory="6000", dryrun=False, job_name="", job_time="", 
                        num_nodes="", num_tasks="", x11_forwarding="", 
                        service_class="", debug=False, extra=""):
        """
        Calls FSL's PALM for permutation-based analysis on a set of neuroimaging data.

        Parameters:
        - dv_nifti_file_path: str, path to the dependent variable 4D NIFTI file.
        - design_matrix: pd.DataFrame, the design matrix for the analysis.
        - contrast_matrix: pd.DataFrame, the contrast matrix for the analysis.
        ... (rest of the parameters)
        
        Returns:
        - None
        """
        # Setup directories
        working_directory = os.path.join(output_directory, "palm_config")
        os.makedirs(working_directory, exist_ok=True)
        self.output_directory = os.path.join(output_directory, "palm_results")
        os.makedirs(self.output_directory, exist_ok=True)

        # Verify software
        config.verify_software(["palm_path"])

        # Prepare design and contrast matrices
        design_matrix_file, contrast_matrix_file = self.prepare_matrices(
            design_matrix, contrast_matrix, working_directory)

        # Prepare mask
        mask = self.prepare_mask(mask, working_directory)

        # Prepare exchangeability blocks
        eb_file = self.prepare_eb(eb, working_directory)

        # Build and run PALM command
        palm_cmd = self.build_palm_cmd(
            concat_file=dv_nifti_file_path, 
            design_matrix_file=design_matrix_file, 
            contrast_matrix_file=contrast_matrix_file,
            mask=mask, 
            eb_file=eb_file, 
            iterations=iterations, 
            accel=accel, 
            save_1p=save_1p, 
            logp=logp, 
            tfce=tfce, 
            ise_flag=ise_flag, 
            two_tailed_flag=two_tailed_flag, 
            corrcon_flag=corrcon_flag, 
            fdr_flag=fdr_flag
        )
        self.run_palm_cmd(
            palm_cmd=palm_cmd, 
            output_directory=output_directory, 
            cluster_name=cluster_name, 
            username=username, 
            cluster_email=cluster_email, 
            queue=queue, 
            cores=cores, 
            memory=memory, 
            dryrun=dryrun, 
            job_name=job_name, 
            job_time=job_time, 
            num_nodes=num_nodes, 
            num_tasks=num_tasks, 
            x11_forwarding=x11_forwarding, 
            service_class=service_class, 
            debug=debug, 
            extra=extra
)

    def text2vest(self, input_file, output_file):
        """
        Converts a text file to a FSL VEST file using FSL's Text2Vest utility.

        Parameters:
        - input_file: str, the path to the input text file.
        - output_file: str, the path to the output VEST file.

        Returns:
        - None. An exception is raised if an error occurs.
        """
        # Verify if FSL software is available
        config.verify_software(["fsl_path"])

        # Run Text2Vest using subprocess
        process = subprocess.Popen(
            [f"{config.software['fsl_path']}/bin/Text2Vest", input_file, output_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Capture stdout and stderr
        stdout, stderr = process.communicate()

        # If stdout is not empty, raise an exception
        if stdout:
            raise Exception("Error in text2vest: " + stdout.decode("utf-8"))


    def prepare_matrices(self, design_matrix, contrast_matrix, working_directory):
        """
        Prepares the design and contrast matrices and converts them into FSL VEST format.

        Parameters:
        - design_matrix: pd.DataFrame, the design matrix to be used in the analysis.
        - contrast_matrix: pd.DataFrame, the contrast matrix to be used in the analysis.
        - working_directory: str, the directory where the converted matrices will be saved.

        Returns:
        - Tuple containing paths to the design and contrast matrices in FSL VEST format.
        """
        # Define file paths
        design_matrix_file = f"{working_directory}/design.mat"
        contrast_matrix_file = f"{working_directory}/contrast.con"

        # Save matrices to TSV files
        design_matrix.to_csv(f"{working_directory}/design.tsv", header=False, index=False, sep="\t")
        contrast_matrix.to_csv(f"{working_directory}/contrast.tsv", header=False, index=False, sep="\t")

        # Convert TSV files to FSL VEST files using text2vest
        self.text2vest(f"{working_directory}/design.tsv", design_matrix_file)
        self.text2vest(f"{working_directory}/contrast.tsv", contrast_matrix_file)

        return design_matrix_file, contrast_matrix_file


    def prepare_mask(self, mask, working_directory):
        """
        Prepare the mask for PALM analysis. If no mask is provided, it uses a default mask.

        Parameters:
        - mask: str, the path to the mask file. If empty, a default mask will be used.
        - working_directory: str, the directory where the default mask file will be saved if used.

        Returns:
        - str: the path to the mask file to be used in PALM analysis.
        """
        if mask == "":
            mask_file = os.path.join(working_directory, f"MNI152_T1_2mm_brain_mask_dil.nii")
            ds.get_img("MNI152_T1_2mm_brain_mask_dil").to_filename(mask_file)
            return mask_file
        else:
            mask_file = os.path.join(working_directory, f"{mask}.nii")
            ds.get_img(mask).to_filename(mask_file)
            return mask_file
        
    def prepare_eb(self, eb, working_directory):
        """
        Prepares and saves the exchangeability blocks (eb) to a CSV file.

        Parameters:
        - eb: pd.DataFrame or None, the exchangeability blocks to be used in the analysis.
        - working_directory: str, the directory where the eb file will be saved.

        Returns:
        - str or None: The file path to the saved eb file, or None if eb is not provided.
        """
        eb_file = None
        if eb is not None:
            eb_file = f"{working_directory}/eb.csv"
            eb.to_csv(eb_file, header=False, index=False)
        return eb_file

    def build_palm_cmd(self, concat_file, design_matrix_file, contrast_matrix_file,
                    mask, eb_file, iterations, accel, save_1p, logp, tfce, ise_flag, 
                    two_tailed_flag, corrcon_flag, fdr_flag):
        """
        Builds the command line command for running PALM based on the provided parameters.

        Parameters:
        - concat_file: str, path to the 4D Nifti file that contains all the input images.
        - design_matrix_file: str, path to the design matrix file.
        - contrast_matrix_file: str, path to the contrast matrix file.
        - mask: str, path to the mask file.
        - eb_file: str, path to the exchangeability blocks file.
        - iterations: int, number of iterations for permutation testing.
        - accel: str, acceleration method.
        - save_1p: bool, whether to save 1-p values.
        - logp: bool, whether to log-transform p-values.
        - tfce: bool, whether to use TFCE.
        - ise_flag: bool, whether to perform image-based smoothing.
        - two_tailed_flag: bool, whether to perform a two-tailed test.
        - corrcon_flag: bool, whether to correct for contrasts.
        - fdr_flag: bool, whether to use FDR correction.
        
        Returns:
        - list: A list of command line arguments for running PALM.
        """
        palm_cmd = [
            f"{config.software['palm_path']}/palm",
            "-i", os.path.abspath(concat_file),
            "-o", os.path.abspath(self.output_directory) + "/",
            "-d", os.path.abspath(design_matrix_file),
            "-t", os.path.abspath(contrast_matrix_file),
            "-n", str(iterations),
            "-m", os.path.abspath(mask)
        ]
        
        if eb_file is not None:
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
        
        return palm_cmd

    def run_palm_cmd(self, palm_cmd, output_directory, cluster_name, username, cluster_email, 
                    queue, cores, memory, dryrun, job_name, job_time, num_nodes, 
                    num_tasks, x11_forwarding, service_class, debug, extra):
        """
        Executes the PALM command and handles cluster submission if needed.

        Parameters:
        - palm_cmd: list, the PALM command to run.
        - output_directory: str, the directory where output should be saved.
        - cluster_name, username, cluster_email, queue, cores, memory, job_name, job_time, 
        num_nodes, num_tasks, x11_forwarding, service_class, debug, extra: Various cluster 
        and job parameters for cluster submission.

        - dryrun: bool, whether to actually run the command or just print it.

        Returns:
        - None
        """
        from time import time
        print("Calling PALM with the following command:")
        print(" ".join(palm_cmd))
        
        start = time()
        if cluster_name == "":
            cmd = palm_cmd
        else:
            cmd = self.build_cluster_submit_string(cmd=palm_cmd, directory=output_directory, cluster_name=cluster_name, 
                                                   username=username, cluster_email=cluster_email, 
                                                   queue=queue, cores=cores, memory=memory, 
                                                   job_name=job_name, job_time=job_time, num_nodes=num_nodes,
                                                   num_tasks=num_tasks,
                                                   x11_forwarding=x11_forwarding, service_class=service_class, debug=debug, 
                                                   extra=extra, dryrun=False  # Assuming you want to keep the default value for dryrun
                                                   )
        if not dryrun:
            ipython = get_ipython()
            ipython.system(" ".join(cmd))
            
        end = time()
        print("\n")
        print(f"Time elapsed: {round(end - start)} seconds")
        
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

    #----------------------------------------------------------------OLD CODE BELOW        
    def original_call_palm(
        self, input_imgs, design_matrix, contrast_matrix, working_directory=None, output_directory=None, 
        iterations=10000, voxelwise_evs=None,
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
        
