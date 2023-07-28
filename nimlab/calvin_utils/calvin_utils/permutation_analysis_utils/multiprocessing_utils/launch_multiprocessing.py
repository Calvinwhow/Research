from calvin_utils.permutation_analysis_utils.multiprocessing_utils import *
# # Usage in Jupyter notebook
# job = ClusterJob()
# job.run()
class ClusterJob:
    def __init__(self):
        self.user_info = None
        self.func = None
        self.args = None
        self.matrix_to_compute = None

    def collect_user_input(self):
        self.user_info = collect_user_and_job_info()
        print(f"User info received:")
        print(self.user_info)

    def collect_function_info(self):
        self.func = collect_function()
        self.args = collect_args(self.func)

    def import_data(self):
        self.matrix_to_compute = pd.read_csv(self.user_info['data_path'], index_col=False)

    def submit_jobs(self):
        submit_jobs_in_batches(
            function_to_submit=self.func,
            function_args=self.args,
            out_dir=self.user_info['out_dir'],
            job_name=self.user_info['job_name'],
            n_permutations=self.user_info['n_permutations'],
            n_cores=self.user_info['cores']
        )

    def run(self):
        self.collect_user_input()
        self.collect_function_info()
        self.import_data()
        self.submit_jobs()

# REMEMBER TO COMPARE THE EMPIRIC DATA TO YOUR OBSERVED DATA SOMEWHERE ELSE 
## -Calvin