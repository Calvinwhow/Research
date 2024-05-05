script_dict = {
    "scripts": [
        {
            "script_name": "launch_f_test_palm.py",
            "method": "F-Test",
            "README": "This script performs voxelwise interaction F-statistic permutation tests on provided outcomes, clinical covariates, and voxelwise neuroimaging data. User input is required to define the number of permutations, the output directory, job name, and paths to the outcome, clinical covariate, and neuroimaging data. The script uses multiprocessing for efficient computation of the permutation tests. It generates permuted versions of the patient labels in the input data and calculates the F-statistic for each permutation. The results are saved to the output directory, with each permutation result stored in a separate csv file.",
            "neuroimaging_df_paths": "The paths to the voxelwise neuroimaging data csv files.",
            "import": "from calvin_utils.statistical_utils.voxelwise_statistical_testing import voxelwise_interaction_f_stat",
            "inputs": {
                "n_cores": "The number of cores per job submission cpu (4 is a good default).",
                "out_dir": "The output directory where the result csv files will be saved.",
                "job_name": "The job name for identification.",
                "memory_per_job": "The memory (in gigabytes) per job submission cpu. General max is 40 up to 498 Gb.",
                "outcome_data_path": "The path to the outcome data csv file.",
                "clinical_covariate_paths": "The paths to the clinical covariate data csv files.",
            }
        },
        {
            "script_name": "launch_voxelwise_fit_test_palm.py",
            "method": "Voxelwise_Fit_Test",
            "README": "This script extracts voxelwise r-squared of provided outcomes on clinical covariates and voxelwise neuroimaging data. User input is required to define the number of permutations, the output directory, job name, and paths to the outcome, clinical covariate, and neuroimaging data. The script uses multiprocessing for efficient computation of the permutation tests. It generates permuted versions of the patient labels in the input data and calculates the t-statistic for each permutation. The results are saved to the output directory, with each permutation result stored in a separate csv file.",
            "import": "from calvin_utils.statistical_utils.voxelwise_statistical_testing import voxelwise_r_squared",
            "get_docstring": "print(voxelwise_r_squared.__doc__)",
            "inputs": {
                "n_cores": "The number of cores per job submission cpu (4 is a good default).",
                "out_dir": "The output directory where the result csv files will be saved.",
                "job_name": "The job name for identification. Can be anything.",
                "memory_per_job": "The memory (in gigabytes) per job submission cpu. General max is 40 up to 498 Gb.",
                "outcome_data_path": "remote_df_paths_dict['outcomes']",
                "clinical_covariate_paths": "remote_df_paths_dict['covariates']",
                "neuroimaging_df_paths": "remote_df_paths_dict['voxelwise']",

            }
        }
    ]
}
