import os
import re
import sys
import inspect
import subprocess
import getpass
import importlib.util
from pathlib import Path

class SlurmWorker:
    def __init__(self, job_name, env = 'bash', **kwargs):
        self.username = getpass.getuser()
        self.job_name = job_name
        self._set_env(env)
        stdout = kwargs.pop('stdout') if 'stdout' in kwargs.keys() else None
        stderr = kwargs.pop('stderr') if 'stderr' in kwargs.keys() else None
        self._set_config(**kwargs)
        self._set_logging(stdout, stderr)
    
    @staticmethod
    def get_modulescript(command, *args):
        cmd = f"{os.environ['LMOD_CMD']} python {command} "
        if len(args):
            cmd + " ".join(args)
        return os.popen(cmd).read()
    
    @staticmethod
    def get_importscript(python_module):
        Path(inspect.getsourcefile(python_module))
    
    def _set_env(self, env):
        if env == "bash":
            self._env = r"#!/bin/bash"
        else:
            self._env = fr"#!/bin/env {env}"
    
    def _set_config(self, 
                    cpus_per_task=1,
                    mem='64G',
                    time='1-10:00:00'):
        self.cpus_per_task = cpus_per_task
        self.mem = mem
        self.time = time
    
    def _set_logging(self,
                     stdout = None,
                     stderr = None):
        self.stdout = stdout if stdout is not None else 'stdout_%A.out'
        self.stderr = stderr if stderr is not None else 'stderr_%A.out'
    
    def _get_config(self):
        config =  [f"--job-name\t{self.job_name}",
                   f"--cpus-per-task\t{self.cpus_per_task}",
                   f"--mem\t{self.mem}",
                   f"--time\t{self.time}",
                   f"--output\t{self.stdout}",
                   f"--error\t{self.stderr}"]
        return [fr"#SBATCH {c}" for c in config]
    
    def __header__(self):
        header = [self._env]
        header.extend(self._get_config())
        return header
    
    def get_script(self, cmds):
        script = self.__header__()
        if self._env.endswith('python'):
            config = [
                "import sys",
                f"sys.path.append('{Path.cwd().as_posix()}')",
            ]
            script.extend(config)
            script.append('\n')
        script.extend(cmds)
        return '\n'.join(script) + '\n'
    
    def run(self, cmds):
        logging_dir = Path(self.stdout).parent
        logging_dir.mkdir(parents=True, exist_ok=True)
        job_script = self.get_script(cmds)
        return subprocess.run(['sbatch'], input=job_script, text=True, capture_output=True, shell=True)
    
    def cancel(self):
        print(os.popen(f"scancel -u {self.username} -n {self.job_name}").read())
        
    def clear_log(self):
        logging_dir = Path(self.stdout).parent
        for f in logging_dir.iterdir():
            if f.is_file():
                f.unlink()
    
    def check_progress(self, results):
        """
        Check the progress of a submitted Slurm job.

        Args:
        - result (object): Result object from subprocess.run containing stdout with submitted job details.

        Returns:
        - list: A list of dictionaries representing the queued job status.
        """

        # Regular expression patterns
        returned_ptrn = r'Submitted batch job (\d+)'
        line_ptrn = (r'\s+(?P<JOBID>\S+)\s+(?P<PARTITION>\S+)\s+(?P<NAME>\S+)\s+(?P<USER>\S+)' 
                     r'\s+(?P<ST>\S+)\s+(?P<TIME>\S+)\s+(?P<NODES>\S+)\s+(?P<NODELIST>\S+)')

        if not isinstance(results, list):
            results = [results]

        # Extract the job ID from the result
        job_ids = []
        for result in results:
            match = re.match(returned_ptrn, result.stdout)
            if match: 
                job_ids.append(match.groups()[0])
        if len(job_ids) == 0:
            return []

        # Get the Slurm queue for the current user
        cmd = f'squeue -u {self.username} -n {self.job_name}'
        output = subprocess.run(cmd, text=True, shell=True, capture_output=True)

        # Parse the output to extract job details
        parsed_lines = [re.match(line_ptrn, line).groupdict() for line in output.stdout.split('\n') if re.match(line_ptrn, line)]

        # Filter results for the given job_id
        output = []
        for job_id in job_ids:
            output.extend([line for line in parsed_lines if job_id in line['JOBID']])
        return output

    def monitor_jobs(self, results, check_interval=20):
        """
        Monitors the progress of a submitted Slurm job, displaying its status in real-time.

        Args:
        - result (object): Result object from subprocess.run containing stdout with submitted job details.
        - check_interval (float): Time interval (in seconds) between consecutive job status checks. Default is 0.1 seconds.

        Action:
        - Continuously queries and displays the job's status until the job is no longer present in the Slurm queue.
        """
        from IPython.display import clear_output
        import time

        jobs = self.check_progress(results)

        # Continue to monitor while the job exists in the queue
        while jobs:
            clear_output(wait=True)

            # Display the header (column names)
            print("\t".join(jobs[0].keys()))

            # Display job details
            for job in jobs:
                print("\t".join(job.values()))

            # Pause before the next check
            time.sleep(check_interval)

            # Refresh job status
            jobs = self.check_progress(results)

        clear_output(wait=True)
        print('Done...')