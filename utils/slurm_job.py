import time
import os
import logging
import shutil
import datetime
import sys
import pathlib
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

logger = logging.getLogger(__name__)

from utils.utils import setup_logger, str2bool, ArgumentSaver, AddDefaultInformationAction

BATCH_JOB_FILE = os.path.join(str(pathlib.Path(__file__).parent.absolute()), "batch_job.py")

# timelimit_argument_str = "-t 3-23:00:00"
# timelimit_argument_str = "-t 2-23:00:00"
GPU_argument_str = "--gres=gpu:1"
CPU_argument_str = "-c 1"
DEFAULT_MEM = 32000
# RAM_argument_str = "--mem 20000"

GPU_partition_argument_str = "-p gpu.q"

CPU_partition_argument_str = "-p cpu.q"

CPU_exclude_nodes_str = ""
GPU_exclude_nodes_str = ""


def _run_local_job(run_line, log_filename, finish_filename):
    """
    Helper function to run a job locally as a subprocess.
    This is defined at module level to be picklable for multiprocessing.
    """
    try:
        with open(log_filename, 'w') as log_file:
            process = subprocess.run(
                run_line,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        if process.returncode == 0:
            # Create finish file on success
            if finish_filename:
                with open(finish_filename, 'w') as f:
                    f.write("COMPLETED")
            return "COMPLETED", 0
        else:
            return "FAILED", process.returncode
    except Exception as e:
        return "FAILED", str(e)

class SlurmJobState:
    def __init__(self, job, state, extra):
        self.job = job
        self.state = state
        self.extra = extra

    def __repr__(self):
        return "SlurmJobState(job={}({}), state={})"\
        .format(self.job.job_name, self.job.job_id, self.state)

    def is_successfull(self):
        return self.state.startswith("COMPLETED")


class LocalJob:
    """
    A job that runs locally as a subprocess instead of on SLURM.
    Used when --use_local is set.
    """
    def __init__(self, job_name, job_folder, run_line, timelimit=None, mem=DEFAULT_MEM, use_finishfile=False):
        self.job_name = job_name
        self.job_filename = os.path.join(job_folder, job_name + ".job")
        self.log_filename = os.path.join(job_folder, job_name + ".log")
        self.finish_filename = os.path.join(job_folder, job_name + ".finish") if use_finishfile else None
        self.job_id = None
        self.timelimit = timelimit
        self.mem = mem
        self.use_finishfile = use_finishfile
        self.process = None
        self._state = None
        self._future = None

        self.run_line = run_line
        if self.use_finishfile and f"--finish_file" not in self.run_line:
            self.run_line += f" --finish_file {self.finish_filename}"

    def send(self):
        """Start the job as a local subprocess."""
        # Write job file for reference
        with open(self.job_filename, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"# Local job: {self.job_name}\n")
            fh.writelines(self.run_line)
        
        self.job_id = os.getpid()  # Use current PID as placeholder
        logger.info("Local job {} started".format(self.job_name))

    def run_sync(self):
        """Run the job synchronously (blocking)."""
        self.send()
        try:
            with open(self.log_filename, 'w') as log_file:
                process = subprocess.run(
                    self.run_line,
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.timelimit if self.timelimit else None
                )
            
            if process.returncode == 0:
                self._state = "COMPLETED"
                if self.finish_filename:
                    with open(self.finish_filename, 'w') as f:
                        f.write("COMPLETED")
            else:
                self._state = "FAILED"
        except subprocess.TimeoutExpired:
            self._state = "TIMEOUT"
        except Exception as e:
            self._state = "FAILED"
            logger.error(f"Local job {self.job_name} failed with exception: {e}")

    def join(self, timeout=None, sleep_time=10):
        """Wait for the job to complete."""
        total_time_waited = 0
        
        # If job was run synchronously, return immediately
        if self._state is not None:
            state_obj = SlurmJobState(self, self._state, "")
            if self._state == "COMPLETED":
                logger.info("Local job {} completed successfully".format(self.job_name))
            else:
                logger.info("Local job {} finished with state {}".format(self.job_name, self._state))
            return state_obj, total_time_waited
        
        # Check finish file for async jobs
        while True:
            if timeout and total_time_waited > timeout:
                return SlurmJobState(self, "TIMEOUT", ""), total_time_waited
            
            if self.finish_filename and os.path.exists(self.finish_filename):
                logger.info("Local job {} completed successfully".format(self.job_name))
                return SlurmJobState(self, "COMPLETED", ""), total_time_waited
            
            time.sleep(sleep_time)
            total_time_waited += sleep_time

    def cancel(self, extra=""):
        """Cancel the job (terminate the process if running)."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self._state = "CANCELLED"
        return SlurmJobState(self, "CANCELLED", extra)


class SlurmJob:
    def __init__(self, job_name, job_folder, run_line,
     run_on_GPU=False, timelimit=None, mem=DEFAULT_MEM, use_scontrol=False,
       use_finishfile=False):
        self.job_name = job_name
        self.job_filename = os.path.join(job_folder, job_name + ".job")
        self.log_filename = os.path.join(job_folder, job_name + ".log")
        self.finish_filename = os.path.join(job_folder, job_name + ".finish")
        self.run_on_GPU = run_on_GPU
        self.job_id = None
        self.timelimit = timelimit
        self.mem = mem
        self.use_scontrol = use_scontrol
        self.use_finishfile = use_finishfile
        self.job_batcher = None
        self.cluster_command_prefix = "/opt/slurm/bin"

        self.run_line = run_line
        if self.use_finishfile:
            self.run_line += f" --finish_file {self.finish_filename}"

    def set_batcher(self, batcher):
        self.job_batcher = batcher
        self.job_id = -1

    def send(self):
        # write a job file and run it
        with open(self.job_filename, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --job-name %s\n" %(self.job_name))
            fh.writelines("#SBATCH -o %s\n" %(self.log_filename))
            if self.run_on_GPU:
                fh.writelines("#SBATCH %s\n" %(GPU_partition_argument_str))
                fh.writelines("#SBATCH %s\n" %(GPU_argument_str))

                fh.writelines("#SBATCH %s\n" %(GPU_exclude_nodes_str))
                
            else:
                fh.writelines("#SBATCH %s\n" %(CPU_partition_argument_str))

                fh.writelines("#SBATCH %s\n" %(CPU_exclude_nodes_str))

            if self.timelimit:
                # assumed in seconds
                days = self.timelimit // (24 * 60 * 60)
                hours = (self.timelimit - days * 24 * 60 * 60) // (60 * 60)
                minutes = (self.timelimit - days * 24 * 60 * 60 - hours * 60 * 60) // 60
                seconds = self.timelimit - days * 24 * 60 * 60 - hours * 60 * 60 - minutes * 60
                timelimit_argument_str = "-t {}-{:02}:{:02}:{:02}".format(days, hours, minutes, seconds)
                fh.writelines("#SBATCH %s\n" %(timelimit_argument_str))
            fh.writelines("#SBATCH %s\n" %(CPU_argument_str))
            # self.mem = 45000 # TODO: hack remove
            fh.writelines("#SBATCH --mem %s\n" %(self.mem))
            fh.writelines(self.run_line)

        popen_output = os.popen(f"{self.cluster_command_prefix}/sbatch {self.job_filename}").read()
        # print(f"{self.cluster_command_prefix}/sbatch")
        # print(popen_output)
        try:
            self.job_id = int(popen_output.split(" ")[3][:-1])
        except Exception as e:
            logger.error(f"Could not parse job id for job {self.job_name} from output: '{popen_output}'")
            raise e
        logger.info("Job {}({}) submitted".format(self.job_name, self.job_id))
    
    def join(self, timeout=None, sleep_time=10):
        total_time_waited = 0
        while True:
            if timeout and total_time_waited > timeout:
                raise Exception("timeout reached") # TODO: better exception type
            try:
                if self.use_scontrol:
                    # logger.info("Trying scontrol show job {}".format(self.job_id))
                    popen_output = os.popen(f"{self.cluster_command_prefix}/scontrol show job {self.job_id}").read()
                    # logger.info(popen_output[:-1])
                    if '(launch failed requeued held)' in popen_output:
                        state_parts = ["FAULTY"]
                    else:
                        state_parts = popen_output.split("\n")[3][12:].split(" ")
                        # logger.info(state_parts)

                elif self.use_finishfile:
                    if os.path.exists(self.finish_filename):
                        state_parts = ["COMPLETED"]
                    else:
                        state_parts = ["RUNNING"]

                        # TODO: should be a flag to use squeue or not?
                        # if has job_id, try squeue
                        if self.job_id > 0:
                            try:
                                popen_output = os.popen(f"{self.cluster_command_prefix}/squeue | grep {self.job_id}").read()
                                if len(popen_output) > 0:
                                    if '(launch failed requeued held)' in popen_output:
                                        state_parts = ["FAULTY"]
                                    else:
                                        state_parts = ["RUNNING"]
                                else:
                                    if os.path.exists(self.finish_filename):
                                        state_parts = ["COMPLETED"]
                                    else:
                                        state_parts = ["FAILED"]
                            except:
                                pass

                else:
                    # logger.info("Trying seff {}".format(self.job_id))
                    popen_output = os.popen(f"{self.cluster_command_prefix}/seff {self.job_id}").read()
                    # logger.info(popen_output[:-1])
                    if '(launch failed requeued held)' in popen_output:
                        state_parts = ["FAULTY"]
                    else:
                        state_parts = popen_output.split("\n")[3][7:].split(" ")
                        # logger.info(state_parts)
                    
                state = state_parts[0]
                extra = ""
                if len(state_parts) > 1:
                    extra = state_parts[1:]
                state_obj = SlurmJobState(self, state, extra)
                # logger.info(state)
                # logger.info(extra)
                
                if state.startswith("COMPLETED"):
                    logger.info("Job {}({}) completed successfully".format(self.job_name, self.job_id))
                    return state_obj, total_time_waited
                if state.startswith("FAULTY"):
                    logger.info("Job {}({}) is faulty, attempting self cancel".format(self.job_name, self.job_id))
                    state = self.cancel(extra=f"FAULTY with {extra}")
                if state.startswith("FAILED") or state.startswith("CANCELLED") or state.startswith("TIMEOUT"):
                    logger.info("Job {}({}) failed with state {}".format(self.job_name, self.job_id, state))
                    try:
                        # copy logfile in case of failure
                        shutil.copyfile(self.log_filename, self.log_filename + "." + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                    except Exception as e:
                        logger.error(f"Failed to copy logfile with exception: {e}")
                    return state_obj, total_time_waited
            except:
                pass
            time.sleep(sleep_time)
            total_time_waited += sleep_time

    def cancel(self, extra=""):
        if self.job_batcher is not None:
            raise Exception("Cannot cancel job that is part of a batcher")
            
        os.popen(f"{self.cluster_command_prefix}/scancel {self.job_id}")
        return SlurmJobState(self, "CANCELLED", extra)

class SlurmJobFactoryState:
    def __init__(self, states):
        self.states = states
        self.succesfull_states = [(s,e) for (s,e) in self.states if s.is_successfull()]
        self.unsuccessfull_states = [(s,e) for (s,e) in self.states if not s.is_successfull()]

    def __repr__(self):
        return "SlurmJobFactoryState(number of jobs={}, sucessfull jobs={})"\
            .format(len(self.states), len(self.succesfull_states))

    def is_successfull(self):
        return len(self.states) == len(self.succesfull_states)

class SlurmJobFactory:
    def __init__(self, job_folder, count_jobs_to_batch=None, use_local=False, max_local_workers=None):
        """
        Factory for creating and managing SLURM or local jobs.
        
        Parameters
        ----------
        job_folder : str
            Directory to store job files and logs
        count_jobs_to_batch : int, optional
            Number of jobs to batch together (SLURM only)
        use_local : bool
            If True, run jobs locally as subprocesses instead of using SLURM
        max_local_workers : int, optional
            Maximum number of parallel local workers. Defaults to CPU count - 1.
        """
        self.job_folder = job_folder
        self.jobs = []
        self.old_jobs = []
        
        self.jobs_to_batch = []
        self.count_jobs_to_batch = count_jobs_to_batch if not use_local else None
        self.batch_jobs = []
        self.old_batch_jobs = []
        self.count_batch_flushes = 0
        
        # Local execution settings
        self.use_local = use_local
        self.max_local_workers = max_local_workers if max_local_workers else max(1, multiprocessing.cpu_count() - 1)
        self._local_futures = []

    def send_job(self, job_name, run_line, run_on_GPU=False, timelimit=None, mem=DEFAULT_MEM,
     use_scontrol=False, use_finishfile=False, extra=None):
        if self.use_local:
            # Create and run a local job
            job = LocalJob(job_name, self.job_folder, run_line, timelimit, mem, use_finishfile=True)
            job.send()
            self.jobs.append((job, extra))
        else:
            if timelimit is None and not run_on_GPU:
                # be nice on default, make sure jobs are not blocking the cluster for too long
                timelimit = 10 * 60 * 60 # 10 hours in seconds
            job = SlurmJob(job_name, self.job_folder, run_line, run_on_GPU, timelimit, mem, use_scontrol, use_finishfile)
            if self.count_jobs_to_batch is None:
                job.send()
                self.jobs.append((job, extra))
            else:
                job.use_finishfile = True
                self.jobs_to_batch.append((job, extra))
                if len(self.jobs_to_batch) >= self.count_jobs_to_batch:
                    self.run_batch_job()

    def run_batch_job(self):
        if len(self.jobs_to_batch) == 0:
            raise Exception("No jobs to batch")

        # assuming all jobs have the same timelimit, mem, run_on_GPU
        first_job_to_batch = self.jobs_to_batch[0][0]
        timelimit = first_job_to_batch.timelimit
        if timelimit is not None:
            timelimit *= len(self.jobs_to_batch)
        mem = first_job_to_batch.mem
        run_on_GPU = first_job_to_batch.run_on_GPU

        # TODO: better name that includes the job names or at least context?
        batch_job_name = f"{self.count_batch_flushes}_batch_{len(self.batch_jobs)}_of_{len(self.jobs_to_batch)}_jobs"
        job_names = [job.job_name for (job, _) in self.jobs_to_batch]
        job_names_str = " ".join(job_names)
        job_log_filenames = [job.log_filename for (job, _) in self.jobs_to_batch]
        job_log_filenames_str = " ".join(job_log_filenames)
        job_runlines = [job.run_line.replace(" ", "*") for (job, _) in self.jobs_to_batch]
        job_runlines_str = " ".join(job_runlines)
        batch_job_run_line = f"python3 -u {BATCH_JOB_FILE} --job_names {job_names_str} --job_log_filenames {job_log_filenames_str} --job_runlines {job_runlines_str}"
        batch_job = SlurmJob(batch_job_name, self.job_folder, batch_job_run_line, run_on_GPU,
            timelimit, mem, False, False)
        batch_job.send()

        for (job, extra) in self.jobs_to_batch:
            job.set_batcher(batch_job)
            self.jobs.append((job, extra))
        self.batch_jobs.append(batch_job)

        self.jobs_to_batch = []

    def flush(self):
        if self.count_jobs_to_batch is not None:
            if len(self.jobs_to_batch) > 0:
                self.run_batch_job()
            self.count_batch_flushes += 1

    def join_all(self, on_join=None, timeout=None, cancel_on_timeout=True, retry=False, on_retry=None, sleep_time=60, job_timeout=60, job_sleep_time=10):
        if self.use_local:
            return self._join_all_local(on_join, timeout, cancel_on_timeout, retry, on_retry, sleep_time, job_timeout, job_sleep_time)
        else:
            return self._join_all_slurm(on_join, timeout, cancel_on_timeout, retry, on_retry, sleep_time, job_timeout, job_sleep_time)

    def _join_all_local(self, on_join=None, timeout=None, cancel_on_timeout=True, retry=False, on_retry=None, sleep_time=60, job_timeout=60, job_sleep_time=10):
        """
        Run all local jobs with parallel execution using ProcessPoolExecutor.
        """
        states = []
        total_time_waited = 0
        start_time = time.time()
        
        logger.info(f"Running {len(self.jobs)} jobs locally with {self.max_local_workers} workers...")
        
        # Run jobs in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.max_local_workers) as executor:
            # Submit all jobs
            future_to_job = {}
            for job, extra in self.jobs:
                future = executor.submit(
                    _run_local_job,
                    job.run_line,
                    job.log_filename,
                    job.finish_filename
                )
                future_to_job[future] = (job, extra)
            
            # Collect results as they complete
            for future in as_completed(future_to_job):
                job, extra = future_to_job[future]
                try:
                    state_str, return_code = future.result(timeout=job_timeout if job_timeout else None)
                    job._state = state_str
                    job_state = SlurmJobState(job, state_str, str(return_code) if return_code else "")
                    
                    if state_str == "COMPLETED":
                        logger.info(f"Local job {job.job_name} completed successfully")
                    else:
                        logger.info(f"Local job {job.job_name} finished with state {state_str}")
                    
                    if retry and not job_state.is_successfull():
                        logger.info(f"Retrying local job {job.job_name}")
                        # Re-submit the job
                        new_future = executor.submit(
                            _run_local_job,
                            job.run_line,
                            job.log_filename,
                            job.finish_filename
                        )
                        future_to_job[new_future] = (job, extra)
                        if on_retry is not None:
                            on_retry(job_state, extra)
                    else:
                        states.append((job_state, extra))
                        if on_join is not None:
                            on_join(job_state, extra)
                            
                except Exception as e:
                    logger.error(f"Local job {job.job_name} failed with exception: {e}")
                    job._state = "FAILED"
                    job_state = SlurmJobState(job, "FAILED", str(e))
                    states.append((job_state, extra))
                    if on_join is not None:
                        on_join(job_state, extra)
                
                # Check timeout
                total_time_waited = time.time() - start_time
                if timeout and total_time_waited > timeout:
                    if cancel_on_timeout:
                        logger.warning(f"Timeout reached, cancelling remaining jobs")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    else:
                        raise Exception("timeout reached")
        
        self.old_jobs += self.jobs
        self.jobs = []
        
        return SlurmJobFactoryState(states)

    def _join_all_slurm(self, on_join=None, timeout=None, cancel_on_timeout=True, retry=False, on_retry=None, sleep_time=60, job_timeout=60, job_sleep_time=10):
        time.sleep(10)
        total_time_waited = 10

        states = []
        remaining_indexes = list(range(len(self.jobs)))
        current_job_count = len(self.jobs)

        to_remove_from_remaining_indexes = []
        while len(remaining_indexes) > 0:
            if timeout and total_time_waited > timeout and not cancel_on_timeout:
                raise Exception("timeout reached")
            for index, (job, extra) in enumerate(self.jobs):
                if index in remaining_indexes:
                    try:
                        if timeout and total_time_waited > timeout:
                            if cancel_on_timeout:
                                if self.count_jobs_to_batch is None:
                                    job_state = job.cancel()
                            else:
                                break
                        else:
                            job_state, job_waited = job.join(timeout=job_timeout, sleep_time=job_sleep_time)
                            total_time_waited += job_waited
                        if retry and not job_state.is_successfull():
                            logger.info("Retrying Job {}({})".format(job.job_name, job.job_id))
                            job.send()
                            if on_retry is not None:
                                on_retry(job_state, extra)
                        else:
                            states.append((job_state, extra))
                            if on_join is not None:
                                on_join(job_state, extra)
                            to_remove_from_remaining_indexes.append(index)
                    except Exception as e:
                        # if exception is no space left on device, remove the job file
                        if "No space left on device" in str(e):
                            logger.error(f"Job {job.job_name}({job.job_id}) failed with exception {e}")
                            logger.error("Removing job file {}".format(job.job_filename))
                            os.remove(job.job_filename)
                        # timeout reached
                        total_time_waited += job_timeout

                        if self.count_jobs_to_batch is not None:
                            if job.job_batcher is not None:
                                try:
                                    batch_job_state, _ = job.job_batcher.join(timeout=1)
                                    if batch_job_state.is_successfull():
                                        # if the batcher succeeded, but the job timed out,
                                        # most likely the job had an error / exception,
                                        # so we should finish this job as well
                                        job_state = SlurmJobState(self, "ERRORED", "")
                                        states.append((job_state, extra))
                                        if on_join is not None:
                                            on_join(job_state, extra)
                                        to_remove_from_remaining_indexes.append(index)
                                    else:
                                        # if the batcher failed, we should finish this job as well, with the same state
                                        states.append((batch_job_state, extra))
                                        if on_join is not None:
                                            on_join(batch_job_state, extra)
                                        to_remove_from_remaining_indexes.append(index)

                                except:
                                    # timeout reached
                                    total_time_waited += 1

            for index_to_remove in to_remove_from_remaining_indexes:
                remaining_indexes.remove(index_to_remove)
            to_remove_from_remaining_indexes = []

            # accounting for send job while in the loop (e.g. in on_join)
            if len(self.jobs) != current_job_count:
                remaining_indexes += list(range(current_job_count, len(self.jobs)))
                current_job_count = len(self.jobs)

            if len(remaining_indexes) > 0:
                time.sleep(sleep_time)
                total_time_waited += sleep_time

        self.old_jobs += self.jobs
        self.jobs = []

        if self.count_jobs_to_batch is not None:
            self.old_batch_jobs += self.batch_jobs
            self.batch_jobs = []

        return SlurmJobFactoryState(states)

def get_job_submitter_args():
    saver = ArgumentSaver()

    saver.add_argument('--cpu_job_memory', type=int, default=32, action=AddDefaultInformationAction)
    saver.add_argument('--gpu_job_memory', type=int, default=32, action=AddDefaultInformationAction)
    saver.add_argument('--maximum_cpu_parallel_jobs', default=100, type=int, action=AddDefaultInformationAction)
    saver.add_argument('--maximum_gpu_parallel_jobs', default=4, type=int, action=AddDefaultInformationAction)
    saver.add_argument('--retry', type=str2bool, nargs='?', const=True, default=True, action=AddDefaultInformationAction)
    saver.add_argument('--use_scontrol', type=str2bool, nargs='?', const=True, default=False, action=AddDefaultInformationAction)
    saver.add_argument('--use_finishfile', type=str2bool, nargs='?', const=True, default=True, action=AddDefaultInformationAction)
    saver.add_argument('--count_cpu_jobs_to_batch', default=None, type=int, action=AddDefaultInformationAction)
    
    # Local execution (no SLURM)
    saver.add_argument('--use_local', type=str2bool, nargs='?', const=True, default=False, action=AddDefaultInformationAction,
                       help='Run jobs locally as subprocesses instead of using SLURM')
    saver.add_argument('--max_local_workers', type=int, default=None, action=AddDefaultInformationAction,
                       help='Maximum number of parallel local workers (default: CPU count - 1)')

    return saver

def get_job_args():
    saver = ArgumentSaver()

    saver.add_argument('--finish_file', default=None)

    return saver