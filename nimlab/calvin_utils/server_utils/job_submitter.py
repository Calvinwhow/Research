import subprocess
import os
from nimlab import configuration as config
import time

# Enter Submission Information
## Multiprocessing information
processes = 10000 #How many computations must be done in total
num_multi_processes = 4 #How many computations can be done at a time

## Submission information
job_name = 'multi_sub_multiproc_palm'
user = 'choward12@bwh.harvard.edu'
stdout_dir = '/PHShome/cu135/terminal_outputs'
stderr_dir = '/PHShome/cu135/terminal_outputs'
queue = 'big-multi'
script_language = 'python'
script = '/PHShome/cu135/python_scripts/palm.py'
out_dir = 'memory/server_inputs/age_voxel_interaction_palm/results'
desired_cwd = '/PHShome/cu135/python_scripts' #where the code to execute resides
test_command = False

#-----end user input
#Enter the information into a dict
submission_commands = {
	'-J': f'{job_name}',
	'-u': f'{user}',
	'-o': f'{stdout_dir}',
	'-e': f'{stderr_dir}',
	'-q': f'{queue}',
	f'{script_language}': f'{script}',
	'-cwd': f'{desired_cwd}',
	'-w': f'{num_multi_processes}'
}

batches = [i for i in range(0, processes, num_multi_processes)]

# For each batch of processes, submit an LSF job to run them
job_ids = []
for i, batch in enumerate(batches):
	# Define the command to run the batch
	## Update information particular to this batch
	batch_commands = submission_commands.copy()
	batch_commands['-J'] = batch_commands['-J'] + f'{i}'
	batch_commands['-o'] = os.path.join(batch_commands['-o'], f'{job_name}_{i}.txt')
	batch_commands['-e'] = os.path.join(batch_commands['-e'], f'{job_name}_{i}.txt')
	command = ' '.join([f'{k} {v}' for k, v in batch_commands.items()])
	# Write the command to a file
	with open(f"batch_{i}.sh", "w") as f:
		f.write(f"bsub {command}")
	#Print the command for testing
	with open(f"batch_{i}.sh", "r") as f:
		if test_command:
			with open(f"batch_{i}.sh", "r") as f:
				script = f.read()
				print(script)
		else:
			script = f.read()
			#subprocess.run(script)
			subprocess.Popen(['bash', f'batch_{i}.sh'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
			time.sleep(5)