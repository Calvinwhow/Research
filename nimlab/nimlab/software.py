from __future__ import print_function
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

DEFAULT_MASK = "MNI152_T1_2mm_brain_mask_dil"
try:
    from pathlib import Path
except:
    from pathlib2 import Path


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


def call_connectome_quick(
    read_input,
    output_directory,
    numWorkers=4,
    command="seed",
    connectome="",
    brain_connectome="",
    brain_space="",
    output_mask="",
    cluster_name="",
    username="",
    cluster_email="",
    dryrun=False,
    queue="normal",
    cores="4",
    memory="6000",
    no_warn="False",
    job_name="",
    job_time="",
    num_nodes="",
    num_tasks="",
    x11_forwarding="",
    service_class="",
    debug=False,
    extra="",
):
    """Call the connectome_quick script.

    Args:
        read_input (str): Path to ROI list
        output_directory (str): Output path
        numWorkers (int, optional): Number of worker threads to use. Defaults to 12. 4 is recommended
            for use on ERIS normal queue jobs.
        command (str, optional): Whether to run seed or matrix connectivity. Defaults to "seed".
        connectome (str, optional): Path to connectome files used for ROIs. Defaults to "/data/nimlab/connectome_npy/yeo1000_dil".
        brain_connectome (str, optional): Path to connectome files used for the Brain. If not set, uses same connectome as <connectome>
        brain_space (str, optional): Mask that connectome files were masked with. Defaults to "".
        output_mask (str, optional): Mask to mask the output. Defaults to "".
        cluster_name (str, optional): Cluster to submit to. If "" (blank), runs on local kernel. Defaults to "".
        username (str, optional): Cluster username. Defaults to "".
        cluster_email (str, optional): Cluster email address, for stdout redirection. Defaults to "".
        dryrun (bool): If "True", returns cluster submit string for manual submission. Does not submit to cluster. Defaults to "False"
        queue (str, optional): Queue name. Defaults to "normal".
        cores (str, optional): Number of cluster cores to request. Defaults to 4.
        memory (str, optional): Amount of memory requested. Defaults to 6000
        no_warn (str, optional): If True, suppress warning flags, such as those commonly
            thrown by autocorrelations arising from single voxel ROIs. Defaults to "False".
        job_name (str, optional): Name of job for scheduler
        job_time (str, optional): Job runtime limit for scheduler, may be needed for slurm.
        extra (bool, optional): If True, includes "extra" scripts in job script builder. Defaults to False.

    """
    print("Process starts")

    script_path = str(Path(__file__).parents[0] / "scripts/connectome_quick.py")

    connectome_cmd = [
        "python",
        str(script_path),
        "-cs",
        connectome,
        "-r",
        read_input,
        "-c",
        command,
        "-o",
        output_directory,
        "-w",
        str(numWorkers),
    ]
    if brain_connectome:
        connectome_cmd.append("-csbrain")
        connectome_cmd.append(brain_connectome)
    if brain_space:
        connectome_cmd.append("-bs")
        connectome_cmd.append(brain_space)
    if output_mask:
        connectome_cmd.append("-mout")
        connectome_cmd.append(output_mask)
    if no_warn == "False":
        connectome_cmd.append("--showwarnings")
    if cluster_name == "":
        cmd = connectome_cmd
    else:
        cmd = build_cluster_submit_string(
            directory=output_directory,
            cluster_name=cluster_name,
            username=username,
            cmd=connectome_cmd,
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


def call_disconnectome(
    input_directory,
    output_directory,
    connectome_directory,
    threshold=0.0,
    cluster_name="",
    username="",
    cluster_email="",
    queue="normal",
    cores="4",
    memory="6000",
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
    """Call the disconnectome script.

    Args:
        input_directory (str): Input path or path to input CSV.
        output_directory (str): Output path.
        connectome_directory (str): Path to structural connectome dataset directory.
        threshold (float): Level to threshold output images at. Ranges from 0.0 to 1.0. Defaults to 0.
        cluster_name (str): Cluster to submit to.
        username (str): Cluster username.
        cluster_email (str): Cluster email address, for stdout redirection.
        queue (str): Queue to submit job to.
        cores (str): Number of requested cores.
        memory (str): Amount of requested memory.
        dryrun (bool): If "True", returns cluster submit string for manual submission. Does not submit to cluster.
        job_name (str): Name of job for scheduler
        job_time (str): Job runtime limit for scheduler, may be needed for slurm.
        num_nodes (str): Number of nodes (SLURM)
        num_tasks (str): Number of tasks (SLURM)
        x11_forwarding (str): X11 Forwarding options
        service_class (str): Service class
        extra (bool): If True, includes "extra" scripts in job script builder. Defaults to False.
    """
    config.verify_software(["bcb_path"])
    print("Process starts")
    connectome_cmd = [
        os.path.join(config.software["bcb_path"], "Tools/scripts/xargs_disco.sh"),
        os.path.join(config.software["bcb_path"], "Tools/scripts/disco.sh"),
    ]

    # If input_directory is a path to input CSV, copy all lesions in input CSV to a tmp folder in output directory
    if ".csv" in os.path.basename(input_directory):
        if os.path.exists(input_directory):
            if os.path.exists(os.path.join(output_directory, "tmp_bcb_disco")):
                shutil.rmtree(os.path.join(output_directory, "tmp_bcb_disco"))
            os.makedirs(os.path.join(output_directory, "tmp_bcb_disco"), exist_ok=True)
            df = pd.read_csv(input_directory, header=None)
            for i in range(len(df)):
                shutil.copy(
                    df.iloc[i][0], os.path.join(output_directory, "tmp_bcb_disco")
                )
            input_directory = os.path.join(output_directory, "tmp_bcb_disco")
        else:
            print("Input CSV Doesn't Exist!")

    if not os.path.exists(input_directory):
        print("Input Directory Doesn't Exist!")
    elif not os.path.exists(output_directory):
        print("Output Directory Doesn't Exist!")
    else:
        connectome_cmd.append(str(cores))
        connectome_cmd.append(os.path.abspath(input_directory))
        connectome_cmd.append(os.path.abspath(connectome_directory))
        connectome_cmd.append(os.path.abspath(output_directory))
        connectome_cmd.append(str(threshold))

    if cluster_name == "":
        cmd = connectome_cmd
    else:
        cmd = build_cluster_submit_string(
            directory=output_directory,
            cluster_name=cluster_name,
            username=username,
            cmd=connectome_cmd,
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


def call_lesion_quantification_toolkit(
    input_directory,
    output_directory,
    dataset_name,
    connectivity_type="end",
    sspl_spared_thresh=50,
    smooth=2,
    cluster_name="eristwo",
    username="",
    cluster_email="",
    queue="normal",
    cores="1",
    memory="2000",
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
    """Call the Lesion Quantification Toolkit matlab script.

    Args:
        input_directory (str): Input path or path to input CSV.
        output_directory (str): Output path.
        dataset_name (str): Used as suffix for output files
        connectivity_type (str): Specifies critieria for defining structural connections. Defaults to 'end'
        sspl_spared_thresh (int): Percent threshold for computing SSPLs. Ranges from 1-100. Defaults to 50.
        smooth (int): Corresponds to the FWHM of the smoothing kernel to be applied to percent disconnection voxel maps. Defaults to 2.
        cluster_submit (str): Cluster to submit to.
        username (str): Cluster username.
        cluster_email (str): Cluster email address, for stdout redirection.
        queue (str): Queue to submit job to.
        cores (str): Number of requested cores.
        memory (str): Amount of requested memory.
        dryrun (bool): If "True", returns cluster submit string for manual submission. Does not submit to cluster.
        job_name (str): Name of job for scheduler.
        job_time (str): Job runtime limit for scheduler, may be needed for slurm.
        num_nodes (str): Number of nodes (SLURM).
        num_tasks (str): Number of tasks (SLURM).
        x11_forwarding (str): X11 Forwarding options.
        service_class (str): Service class.
        extra (bool): If True, includes "extra" scripts in job script builder. Defaults to False.
    """
    config.verify_software(["lqt_path"])
    print("Process starts")
    connectome_cmd = [
        f"export MATLABPATH={config.software['lqt_path']}/Functions:$MATLABPATH;"
    ]
    connectome_cmd.append("sh")
    script_path = str(Path(__file__).parents[0] / "scripts/LQT_matlab_wrapper.sh")
    connectome_cmd.append(script_path)

    # If input_directory is a path to input CSV, copy all lesions in input CSV to a tmp folder in output directory
    if ".csv" in os.path.basename(input_directory):
        if os.path.exists(input_directory):
            if os.path.exists(os.path.join(output_directory, "tmp_lqt")):
                shutil.rmtree(os.path.join(output_directory, "tmp_lqt"))
            os.makedirs(os.path.join(output_directory, "tmp_lqt"), exist_ok=True)
            df = pd.read_csv(input_directory, header=None)
            for i in range(len(df)):
                shutil.copy(df.iloc[i][0], os.path.join(output_directory, "tmp_lqt"))
            input_directory = os.path.join(output_directory, "tmp_lqt")
        else:
            print("Input CSV Doesn't Exist!")

    if not os.path.exists(input_directory):
        print("Input Directory Doesn't Exist!")
    elif not os.path.exists(output_directory):
        print("Output Directory Doesn't Exist!")
    else:
        connectome_cmd.append(input_directory)
        connectome_cmd.append(output_directory)
        connectome_cmd.append(dataset_name)
        connectome_cmd.append(connectivity_type)
        connectome_cmd.append(str(sspl_spared_thresh))
        connectome_cmd.append(str(smooth))

    if cluster_name == "":
        cmd = connectome_cmd
    else:
        cmd = build_cluster_submit_string(
            directory=output_directory,
            cluster_name=cluster_name,
            username=username,
            cmd=connectome_cmd,
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


def call_cbig_registrationFusion(
    mode="",
    input_list="",  # -l
    output_dir="",  # -o
    template_type="MNI152_norm",  # -p
    RF_type="RF_ANTs",  # -r
    interp="linear",  # -i
    average_mesh="fsaverage5",  # -f
    cluster_name="",
    cluster_user="",
    cluster_email="",
    queue="vshort",
    cores="1",
    memory="2000",
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
    """Call the CBIG RegistrationFusion script. Surface files in .nii format

    Args:
        mode (str, required): Conversion Direction. Options are "vol2surf" and "surf2vol".
        input_list (str, required): Absolute path to file containing input data file names. Each line in the file should contain the full path to one file. If using 'vol2surf' mode, note that inputs should be in the format of .nii or .nii.gz. If using 'surf2vol' mode, note that the inputs should be in the format of .mat, each with two variables 'lh_label' and 'rh_label' corresponding to the data meshes of those two hemispheres.
        output_dir (str, required): Absolute path to output directory.
        template_type (str, optional): Type of volumetric template used in index files creation. See /data/nimlab/toolkits/CBIG/stable_projects/registration/Wu2017_RegistrationFusion/registration_fusion/scripts_vol2surf/CBIG_RF_step1_make_xyzIndex_volTemplate.sh for more details. Defaults to "MNI152_norm".
        RF_type (str, optional): Type of Registration Fusion approaches used to generate the mappings ("RF_M3Z" or "RF_ANTs"). RF-M3Z is recommended if data was registered from subject's space to the volumetric atlas space using FreeSurfer. RF-ANTs is recommended if such registrations were carried out using other tools, especially ANTs. Defaults to "RF_ANTs".
        interp (str, optional): Interpolation method ("linear" or "nearest"). Defaults to "linear".
        average_mesh (str, optional): fsaverage mesh version to output to if converting vol2surf ("fsaverage", "fsaverage5", or "fsaverage6"). Defaults to "fsaverage5".
        cluster_submit (str, optional): Cluster to submit to. Currently only "erisone" is supported. Defaults to "".
        cluster_user (str, optional): Cluster username. Defaults to "".
        cluster_email (str, optional): Cluster email address, for stdout redirection. Defaults to "".
        cluster_stdout (str, optional): File to direct stdout. Defaults to "".
        queue (str, optional): Queue name. Defaults to "vshort".
        cores (int, optional): Number of cluster cores to request. Defaults to 1.
        memory (int, optional): Amount of memory to request in MB. Defaults to 2000.
        dryrun (bool): If "True", returns cluster submit string for manual submission. Does not submit to cluster.
        job_name (str): Name of job for scheduler
        job_time (str): Job runtime limit for scheduler, may be needed for slurm.
        num_nodes (str): Number of nodes (SLURM)
        num_tasks (str): Number of tasks (SLURM)
        x11_forwarding (str): X11 Forwarding options
        service_class (str): Service class
        extra (bool): If True, includes "extra" scripts in job script builder. Defaults to False.
    """
    config.verify_software(["cbig_path"])
    print("Process starts")

    if mode == "vol2surf":
        script_path = os.path.join(
            config.software["cbig_path"],
            "stable_projects/registration/Wu2017_RegistrationFusion/bin/scripts_final_proj/CBIG_RF_projectVol2fsaverage_batch.sh",
        )
    elif mode == "surf2vol":
        script_path = os.path.join(
            config.software["cbig_path"],
            "stable_projects/registration/Wu2017_RegistrationFusion/bin/scripts_final_proj/CBIG_RF_projectfsaverage2Vol_batch.sh",
        )

    connectome_cmd = [
        "sh",
        str(script_path),
        "-l",
        input_list,
        "-o",
        output_dir,
        "-r",
        RF_type,
        "-i",
        interp,
    ]
    if mode == "vol2surf":
        connectome_cmd.append("-p")
        connectome_cmd.append(template_type)
        connectome_cmd.append("-f")
        connectome_cmd.append(average_mesh)

    if cluster_name == "":
        cmd = connectome_cmd
    else:
        cmd = build_cluster_submit_string(
            directory=output_dir,
            cluster_name=cluster_name,
            username=cluster_user,
            cmd=connectome_cmd,
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


# NEED TO FIX OR DEPRECATE
def call_bsglmm(
    NTypes, NCov, input_dat="bsglmm_input.dat", MaxIter=1000000, BurnIn=500000
):
    """
    Create standard command for BSGLMM with BinCar location, Number of types, number of covariates+ NTypes (same as BSGLMM),
    1 for GPU,
    """
    # Prespecify the input file by default, 1 for mask, 0 for no white mask, specify number of maximum iterations and number of
    # iterations for burn in. We follow the default of 1,000,000 and 500,000
    bsglmm_cmd = [
        "/data1/apps/BSGLMM-0.3/bsglmm/BinCar",
        str(NTypes),
        str(NCov),
        "1",
        input_dat,
        "1",
        "0",
        str(MaxIter),
        str(BurnIn),
    ]
    # Create log file:
    logfile = "BSGLMM.log"
    # Start timer
    start = time()
    # Open connection to log file:
    with io.open(logfile, "wb") as f:
        print("Process starts")
        print("\n")
        # Open process using subprocess.Popen for better control, specify the command output using stdout
        process = subprocess.Popen(
            bsglmm_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        # Use while statement to maintain updat
        while True:
            # Assess line output then run if statement:
            line = process.stdout.readline()
            # If line output is blank, break workflow
            if not line:
                break
            else:
                # Else, print the line at the console and write it in the log file
                print(line)
                f.write(line)
        print("Process finished")
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


def call_palm(
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
            desnum: The design number, i.e., the design in which this EV should be added.
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
            file, evpos, desnum = voxelwise_ev
            palm_cmd += ["-evperdat", file, str(evpos), str(desnum)]
    
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


def call_randomise(read_input, output_directory, iterations=2000):
    config.verify_software(["fsl_path"])
    randomise_cmd = [
        f"{config.software['fsl_path']}/bin/randomise",
        "-i",
        read_input,
        "-o",
        output_directory,
        "-n",
        str(iterations),
        "-1",
        "-T",
    ]
    subprocess.Popen(randomise_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    logfile = os.path.join(output_directory, "randomise.log")
    start = time()
    with open("randomise.log", "w") as f:
        cprint("Process starts", attrs=["bold"])
        print("\n")
        process = subprocess.Popen(
            randomise_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        while True:
            # Assess line output
            line = process.stdout.readline()
            # If line output is blank, break workflow
            if not line:
                break
            else:
                # Else, print the line at the console and write it in the log file
                if line is not None:
                    f.write(line)
                    if "rror" in line:
                        cprint(line, "red")
                    else:
                        print(line)
