from __future__ import print_function
import os
import datetime
import subprocess
from future.utils import raise_with_traceback
from zipfile import ZipFile
from datetime import timedelta

from spectral_dagger.utils.misc import (
    make_symlink, str_int_list, ZipObjectLoader, zip_root)


def make_directory_name(experiments_dir, network_name, add_date=True):
    if add_date:
        working_dir = os.path.join(experiments_dir, network_name + "_")
        date_time_string = str(datetime.datetime.now()).split('.')[0]
        date_time_string = reduce(
            lambda y, z: y.replace(z, "_"),
            [date_time_string, ":", " ", "-"])
        working_dir += date_time_string
    else:
        working_dir = os.path.join(experiments_dir, network_name)

    return working_dir


def parse_timedelta(s):
    """ s should be of the form HH:MM:SS """
    args = [int(i) for i in s.split(":")]
    return timedelta(hours=args[0], minutes=args[1], seconds=args[2])


def submit_job(
        task, input_zip, n_jobs=-1, n_nodes=1, ppn=12, walltime="1:00:00",
        cleanup_time="00:15:00", add_date=False, test=0, scratch=None,
        exclude="", verbose=0, show_script=0, dry_run=0, queue=None,
        parallel_exe="parallel", sdbin='$HOME/.virtualenvs/main2.7/bin/'):

    idx_file = 'job_indices.txt'
    name = os.path.splitext(os.path.basename(input_zip))[0]
    exclude = "--exclude \*{}\*".format(exclude) if exclude else ""
    queue = "#PBS -q {}".format(queue) if queue is not None else ""
    # Create directory to run the job from - should be on scratch.
    scratch = os.path.abspath(scratch or os.getenv('SCRATCH'))
    experiments_dir = os.path.join(scratch, "experiments")
    scratch = make_directory_name(
        experiments_dir, '{}_{}'.format(name, task), add_date=add_date)
    dirname = zip_root(input_zip)
    sd_parallel = os.path.join(sdbin, 'sd-parallel')
    sd_experiment = os.path.join(sdbin, 'sd-experiment')

    job_results = os.path.abspath(os.path.join(experiments_dir, 'job_results'))

    input_zip = os.path.abspath(input_zip)
    input_zip_bn = os.path.basename(input_zip)
    local_scratch = 'LSCRATCH' if test else '$LSCRATCH'

    cleanup_time = parse_timedelta(cleanup_time)
    walltime = parse_timedelta(walltime)
    execution_time = int((walltime - cleanup_time).total_seconds())

    kwargs = locals().copy()

    try:
        os.makedirs(job_results)
    except:
        pass

    code = '''
#!/bin/bash

# MOAB/Torque submission script for multiple, dynamically-run serial jobs
#
#PBS -V
#PBS -l nodes={n_nodes}:ppn={ppn},walltime={walltime}
#PBS -N {name}_{task}
#PBS -M eric.crawford@mail.mcgill.ca
#PBS -m abe
#PBS -A jim-594-aa
#PBS -e stderr.txt
#PBS -o stdout.txt
''' + queue + '''

# Turn off implicit threading in Python
export OMP_NUM_THREADS=1

cd {scratch}
mkdir results
echo "Starting job at - "
date

echo "Printing RAMDISK..."
{parallel_exe} --no-notice --sshloginfile $PBS_NODEFILE --nonall \\
    echo Ramdisk on host \\$HOSTNAME is \\$RAMDISK, working directory is \\$PWD.

echo "Staging input archive..."
{parallel_exe} --no-notice --sshloginfile $PBS_NODEFILE --nonall \\
    cp {input_zip} \\$RAMDISK

echo "Listing staging results..."
{parallel_exe} --no-notice --sshloginfile $PBS_NODEFILE --nonall \\
    "echo ls on node \\$HOSTNAME && ls"

echo "Running parallel..."
timeout --signal=TERM {execution_time}s \\
    {parallel_exe} --no-notice -j{ppn} --workdir $PWD \\
        --joblog {scratch}/joblog.txt --env OMP_NUM_THREADS --env PATH\\
        --sshloginfile $PBS_NODEFILE \\
        {sd_parallel} {task} \\$RAMDISK/{input_zip_bn} \\$RAMDISK/{dirname} {{}} --verbose {verbose} < {idx_file}

if [ "$?" -eq 124 ]; then
    echo Timed out after {execution_time} seconds.
fi

echo "Cleaning up at - "
date

{parallel_exe} --no-notice --sshloginfile $PBS_NODEFILE --nonall \\
    "echo Retrieving results from node \\$HOSTNAME && cd \\$RAMDISK && zip -rq \\$HOSTNAME {dirname} && mv \\$HOSTNAME.zip {scratch}/results"

cd {scratch}
echo In scratch dir: $PWD
ls
echo "Unzipping basefile..."
unzip -q {input_zip} -d results

echo "In results: "$PWD
cd results
ls

echo "Unzipping results from different nodes..."
for f in *zip
do
    echo "Storing results from node "$f
    unzip -nuq $f
    rm $f
done

echo "Zipping final results..."
zip -qr {name} {dirname}
mv {name}.zip ..
cd ..
echo "Should be in scratch now..."
ls

echo "Removing final results directory..."
rm -rf results

{sd_experiment} complete {name}.zip
cp {name}.zip {job_results}

'''
    code = code.format(**kwargs)
    if show_script:
        print(code)

    if not os.path.isdir(scratch):
        os.makedirs(scratch)

    # Create convenience `latest` symlinks
    make_symlink(scratch, os.path.join(experiments_dir, 'latest'))
    make_symlink(
        scratch, os.path.join(experiments_dir, 'latest_{}_{}'.format(name, task)))

    loader = ZipObjectLoader(input_zip)

    os.chdir(scratch)

    # Write unfinished indices to file, which is given as input to ``parallel`` command
    if task == 'cv':
        scenario_idx = set(loader.indices_for_kind('train_scenario'))
        finished_idx = set(loader.indices_for_kind('cv_score'))
    else:
        scenario_idx = set(loader.indices_for_kind('test_scenario'))
        finished_idx = set(loader.indices_for_kind('test_scores'))
    unfinished = list(scenario_idx.difference(finished_idx))
    if n_jobs >= 0:
        unfinished = unfinished[:n_jobs]
    else:
        print("Got negative value for ``n_jobs``, so submitting all unfinished jobs.")
    if not unfinished:
        print("All jobs are finished! Exiting.")
        return

    print("Submitting {} unfinished jobs:\n{}.".format(len(unfinished), str_int_list(unfinished)))

    with open(idx_file, 'w') as f:
        [f.write('{}\n'.format(u)) for u in unfinished]

    submit_script = "submit_script.sh"
    with open(submit_script, 'w') as f:
        f.write(code)

    if dry_run:
        print("Dry run, so not submitting.")
    else:
        try:
            if test:
                command = ['sh', submit_script]
                print("Testing.")
                output = subprocess.check_output(command, stderr=subprocess.STDOUT)
                print(output)
            else:
                command = ['qsub', submit_script]
                print("Submitting.")
                # Submit to queue
                output = subprocess.check_output(command, stderr=subprocess.STDOUT)
                print(output)

                # Create a file in the directory with the job_id as its name
                job_id = output.split('.')[0]
                open(job_id, 'w').close()
                print("Job ID: {}".format(job_id))
        except subprocess.CalledProcessError as e:
            print("CalledProcessError has been raised while execiting command: {}.".format(' '.join(command)))
            print("Output of process: ")
            print(e.output)
            raise_with_traceback(e)


def _submit_job():
    from clify import command_line
    command_line(submit_job, collect_kwargs=1, verbose=True)()


if __name__ == "__main__":
    _submit_job()
