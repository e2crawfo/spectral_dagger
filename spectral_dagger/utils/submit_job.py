from __future__ import print_function
import os
import datetime
import subprocess
from future.utils import raise_with_traceback

from spectral_dagger.utils.misc import make_symlink


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


def submit_job(
        task, n_jobs, input_zip, n_nodes=1, ppn=12, walltime="1:00:00",
        add_date=False, test=0, scratch=None, exclude="", verbose=0, show_script=0,
        dry_run=0):

    name = os.path.splitext(os.path.basename(input_zip))[0]
    exclude = "--exclude \*{}\*".format(exclude) if exclude else ""
    kwargs = locals().copy()
    kwargs['n_procs'] = n_nodes * ppn
    kwargs['input_zip'] = os.path.abspath(input_zip)
    kwargs['input_zip_bn'] = os.path.basename(input_zip)
    kwargs['original_dir'] = os.path.abspath(os.path.dirname(input_zip))

    # Create directory to run the job from - should be on local_scratch.
    scratch = os.path.abspath(scratch or os.getenv('SCRATCH'))
    experiments_dir = os.path.join(scratch, "experiments")
    scratch = make_directory_name(experiments_dir, name, add_date=add_date)
    kwargs['scratch'] = scratch
    kwargs['local_scratch'] = 'LSCRATCH' if test else '$LSCRATCH'

    if test:
        preamble = '''
#!/bin/bash
mkdir {scratch}/{local_scratch}'''

    else:
        preamble = '''
#!/bin/bash

# MOAB/Torque submission script for multiple, dynamically-run serial jobs on SciNet GPC
#
#PBS -l nodes={n_nodes}:ppn={ppn},walltime={walltime}
#PBS -N {name}
#PBS -M eric.crawford@mail.mcgill.ca
#PBS -m abe
#PBS -e stderr.txt
#PBS -o stdout.txt

module load gcc/5.4.0
module load GNUParallel/20141022
module load MPI/Gnu/gcc4.9.2/openmpi/1.10.2
module load python/2.7.2'''

    if test:
        command = '''
seq 0 $(({n_jobs}-1)) | parallel --no-notice -j{n_procs} --joblog {scratch}/joblog.txt --workdir $PWD sd-experiment {task} {{}} --d {local_scratch}/{name}_{task} --verbose {verbose} > {scratch}/stdout.txt 2> {scratch}/stderr.txt
'''
    else:
        command = '''
# START PARALLEL JOBS USING NODE LIST IN $PBS_NODEFILE
seq 0 $(({n_jobs}-1)) | parallel --no-notice -j{n_procs} --joblog {scratch}/joblog.txt --sshloginfile $PBS_NODEFILE --workdir $PWD sd-experiment {task} {{}} --d {local_scratch}/{name}_{task} --verbose {verbose}
''' # noqa

    code = (preamble + '''

# Turn off implicit threading in Python, R
export OMP_NUM_THREADS=1

cd {scratch}
mpiexec -np {n_nodes} -pernode sh -c 'cp {input_zip} {local_scratch} && \\
                                      unzip -q {local_scratch}/{input_zip_bn} -d {local_scratch} && \\
                                      mv {local_scratch}/{name} {local_scratch}/{name}_{task}'

echo "Starting job at - "
date
SECONDS=0
''' + command + '''
echo "Job took "$SECONDS" seconds."

cd {local_scratch}
mpiexec -np {n_nodes} -pernode sh -c 'zip -rq $OMPI_COMM_WORLD_RANK {name}_{task} {exclude} && \\
                                      cp $OMPI_COMM_WORLD_RANK.zip {scratch}'
cd {scratch}

for i in `seq 0 $(({n_nodes}-1))`;
do
    unzip -qn $i.zip
    rm $i.zip
done

zip -rq {name}_{task} {name}_{task}
rm -rf {name}_{task}
cp {name}_{task}.zip {original_dir}

''')

    code = code.format(**kwargs)
    if show_script:
        print(code)

    if not os.path.isdir(scratch):
        os.makedirs(scratch)

    # Create convenience `latest` symlink
    make_symlink(scratch, os.path.join(experiments_dir, 'latest'))

    os.chdir(scratch)

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
