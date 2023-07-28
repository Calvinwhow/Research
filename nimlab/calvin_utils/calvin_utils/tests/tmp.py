
from calvin_utils.permutation_analysis_utils.launch_palm import VoxelwiseInteractionTester

def run_single_permutation_test():
    tester = VoxelwiseInteractionTester(
        n_permutations=2000,
        outcome_path='path_to_outcome.csv',
        neuroimaging_paths=['path1.csv', 'path2.csv'],
        clinical_paths=['clinical1.csv', 'clinical2.csv'],
        out_dir='output_directory_path',
        job_name='job_name',
        subject_column='% Change from baseline (ADAS-Cog11)',
        outcome_column='% Change from baseline (ADAS-Cog11)',
        clinical_information_column='Age'
    )
    tester.run_permutation_test()

run_single_permutation_test()
