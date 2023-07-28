from __future__ import print_function
import os
import sys
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from nimlab import software as sf
from nimlab import datasets as nimds
from nilearn import image, plotting
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

MNI_brain_mask = nimds.get_img("MNI152_T1_2mm_brain_mask")

def process_nifti_paths(csv_file_path):
    # Load the required packages
    import pandas as pd
    import re
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', None)

    # Function to extract subject ID from the file path
    def extract_subject_id(file_path):
        # Extract 'subject' or 'sub-' and the associated number from the total path
        match = re.search(r'(subject\d+|sub-\d+)', file_path)
        if match:
            # If found, remove non-numeric characters and return the numeric part
            return ''.join(filter(str.isdigit, match.group()))
        
        # If 'subject' or 'sub-' is not found in the total path, extract the basename and find the number in it
        else:
            # Extract the file name from the file path
            file_name = file_path.split('/')[-1]

            # Extract the first number from the file name
            match = re.search(r'\d+', file_name)
            if match:
                return match.group()
            else:
                return pd.np.nan

    # Load the data
    nifti_paths = pd.read_csv(csv_file_path, header=None, names=['nifti_path'])

    # Apply the function to the 'nifti_path' column
    nifti_paths['subject_id'] = nifti_paths['nifti_path'].apply(extract_subject_id)

    # Extract the list of nifti file paths
    nifti_file_paths = nifti_paths['nifti_path'].tolist()

    return nifti_file_paths, nifti_paths
import pandas as pd

def remove_column_spaces(df):
    # Making a copy of the DataFrame
    processed_df = df.copy()
    # Replacing spaces with underscores in column names
    processed_df.columns = [col.replace(" ", "_") for col in processed_df.columns]
    return processed_df

def add_prefix_to_numeric_cols(data_df, prefix='var_'):
    """
    This function renames columns that start with a number by adding a prefix.

    Parameters:
    - data_df: DataFrame, the DataFrame to modify.
    - prefix: str, optional, the prefix to add to column names that start with a number.

    Returns:
    - DataFrame with modified column names.
    """
    new_columns = {col: prefix + col if col[0].isdigit() else col for col in data_df.columns}
    data_df = data_df.rename(columns=new_columns)
    return data_df

def replace_unacceptable_characters(data_df):
    """
    This function replaces unacceptable characters in column names with underscores.

    Parameters:
    - data_df: DataFrame, the DataFrame to modify.

    Returns:
    - DataFrame with modified column names.
    """
    unacceptable_chars = ['-', '#', '%', '(', ')', ' ', ',', ';', '!', '?', '*', '/', ':', '[', ']', '{', '}', '|', '<', '>', '+', '=', '@', '&', '^', '`', '~']
    for char in unacceptable_chars:
        data_df.columns = [col.replace(char, '_') for col in data_df.columns]
    return data_df

def preprocess_colnames_for_regression(data_df):
    data_df = remove_column_spaces(data_df.reset_index(drop=True))
    data_df = add_prefix_to_numeric_cols(data_df)
    data_df = replace_unacceptable_characters(data_df)
    return data_df

def create_design_matrix(formula_vars, data_df, subject_id_column):
    import patsy
    """
    This function creates a design matrix using patsy.

    Parameters:
    - formula_vars: List of strings, each representing a variable in the design formula.
    - data_df: DataFrame, the DataFrame to create the design matrix from.
    - subject_id_column: String, the column in data_df containing the subject id.

    Returns:
    - A DataFrame representing the design matrix.
    """
    # Join the formula_vars with '+' to create the formula string
    formula = ' + '.join(formula_vars)

    # Create the design matrix
    design_matrix = patsy.dmatrix(formula, data_df, return_type='dataframe')

    # Set the index of the design matrix to be the subject id
    design_matrix.index = data_df[subject_id_column]
    design_matrix.index.rename('subject_id', inplace=True)

    return design_matrix

out_dir = '/PHShome/cu135/permutation_tests/fsl_palm/age_on_grey_matter_effect'
path_to_clinical_data = '/PHShome/cu135/datasets/ad_dns/grey_matter_damage_score_and_outcomes.csv'
clinical_df = preprocess_colnames_for_regression(pd.read_csv(path_to_clinical_data))

# Define the explanatory variable formula
# Each variable must be defined as 'Q("var_1"). Interactions are defined as 'Q("var_1"):'Q("var_2")
formula_vars = [
'Age'
]
# Define the column containing th subject id
subject_id_column = 'Patient___CDR__ADAS'


# Create the design matrix#----------------------------------------------------------------
design_matrix = create_design_matrix(formula_vars, clinical_df, subject_id_column)
# Display the design matrix
final_design_matrix = design_matrix

#----------------------------------------------------------------DO NOT TOUCH----------------------------------------------------------------
nifti_path_csv = '/PHShome/cu135/memory/file_paths/paths_to_grey_matter.csv'
ordered_image_list, nifti_df = process_nifti_paths(nifti_path_csv)

#This will generate a basic contrast matrix for you to copy into a cell and edit
contrast_matrix = np.array([[1], [-1]])
print(contrast_matrix)
print('The above contrast matrix is useful to assess if your given explanatory varialbe is significantly different from the slope (ie is a significant coefficient)')
print('Copy it into the cell below and edit it for more control over your analysis.')
contrast_matrix = contrast_matrix
#----------------------------------------------------------------DO NOT EDIT!----------------------------------------------------------------
# https://github.com/nimlab/documentation/wiki/PALM-Analyses
contrast_df = pd.DataFrame(data=contrast_matrix, columns=final_design_matrix.columns)
contrast_df.to_csv(os.path.join(out_dir, 'contrast_matrix.csv'))
print("This is a basic contrast matrix set up to evaluate the significance of each variable. \n Please modify it to assess specific hypotheses.")


DEFAULT_MASK = "MNI152_T1_2mm_brain_mask_dil"
try:
    from pathlib import Path
except:
    from pathlib2 import Path

def calvins_call_palm(
    input_imgs,
    design_matrix,
    contrast_matrix,
    working_directory,
    output_directory,
    iterations,
    voxelwise_evs=None,
    eb=None,
    mask="",
    save_1p=True,
    logp=False,
    tfce=False,
    ise_flag=False,
    two_tailed_flag=True,
    corrcon_flag=False,
    fdr_flag=False,
    accel="",
    cluster_name="",
    username="",
    cluster_email="",
    queue="normal",
    cores=1,
    memory=6000,
    dryrun=False,
    job_name="",
    job_time="",
    num_nodes="",
    num_tasks="",
    x11_forwarding="",
    service_class="",
    debug=False,
    extra="",
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
    config.verify_software(["palm_path"])
    # Concatenate input files
    print("concatenating input...")
    concat_file = os.path.abspath(working_directory) + "/concat.nii"
    image.concat_imgs(input_imgs).to_filename(concat_file)

    # Create and convert design and contrast matrices
    design_matrix_file = working_directory + "/design.mat"
    contrast_matrix_file = working_directory + "/contrast.con"
    design_matrix.to_csv(
        working_directory + "/design.tsv", header=False, index=False, sep="\t"
    )
    contrast_matrix.to_csv(
        working_directory + "/contrast.tsv", header=False, index=False, sep="\t"
    )
    text2vest(working_directory + "/design.tsv", design_matrix_file)
    text2vest(working_directory + "/contrast.tsv", contrast_matrix_file)
    if mask == "":
        mask_file = working_directory + "/" + DEFAULT_MASK + ".nii"
        ds.get_img(DEFAULT_MASK).to_filename(mask_file)
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
        cmd = build_cluster_submit_string(
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
    
def text2vest(input_file, output_file):
    config.verify_software(["fsl_path"])
    process = subprocess.Popen(
        [f"{config.software['fsl_path']}/bin/Text2Vest", input_file, output_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout, stderr = process.communicate()
    if stdout:
        raise Exception("Error in text2vest: " + stdout)
        
def build_cluster_submit_string(
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

# Edit this according to documentation page
cluster_username = 'cu135'
cluster_email = 'choward12@bwh.harvard.edu'
number_of_permutations=2

#----------------------------------------------------------------DO NOT TOUCH----------------------------------------------------------------
working_dir = os.path.join(out_dir, "palm_config")
if not os.path.exists(working_dir):
    os.makedirs(working_dir)
    
output_dir = os.path.join(out_dir, "palm_results")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Current best default settings:
calvins_call_palm(input_imgs=ordered_image_list,
             design_matrix=final_design_matrix,
             contrast_matrix=contrast_df,
             working_directory=working_dir,
             output_directory=output_dir,
             iterations=number_of_permutations,
             accel="tail",
             voxelwise_evs=None,
             eb=None,
             mask="",
             save_1p=True,
             logp=False,
             tfce=False,
             ise_flag=False,
             two_tailed_flag=True,
             corrcon_flag=False,
             fdr_flag=False,
             cluster_name="erisone",
             username=cluster_username,
             cluster_email=cluster_email,
             queue="normal",
             cores="1",
             memory="6000",
             dryrun=False,
             job_name="fsl_palm",
             job_time="",
             num_nodes="",
             num_tasks="",
             x11_forwarding="",
             service_class="",
             debug=False,
             extra=""
    )        
