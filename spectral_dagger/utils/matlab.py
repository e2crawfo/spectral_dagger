import time
import subprocess
import os
import logging
from contextlib import contextmanager
import scipy.io as sio

logger = logging.getLogger(__name__)


@contextmanager
def cd(path):
    """ cd into dir on __enter__, cd back on exit. """

    old_dir = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(old_dir)


def run_matlab_code(code, working_dir='.', verbose=False, **matlab_kwargs):
    pid = os.getpid()
    infile = 'matlab_input_pid_%d.mat' % pid
    outfile = 'matlab_output_pid_%d.mat' % pid

    code = code.format(infile=infile, outfile=outfile)
    command = [
        "matlab", "-nosplash", "-nodisplay", "-nojvm",
        "-nodesktop",  "-r",
        "try, %s, catch exception, "
        "display(getReport(exception)), exit, end, exit;" % code]

    command_output_name = "stdout_%d.txt" % pid
    command_err_name = "stderr_%d.txt" % pid

    logger.info("Process %d starting Matlab section.", pid)
    t0 = time.time()
    results = {}

    with cd(working_dir):
        sio.savemat(infile, matlab_kwargs)

        try:
            print("Calling matlab with command: ")
            print("    %s" % command)

            with open(command_output_name, 'w') as command_output:
                with open(command_err_name, 'w') as command_err:
                    subprocess.call(
                        command, stdout=command_output, stderr=command_err)

            print("Matlab call complete, took %s seconds. "
                  "Loading results..." % (time.time() - t0))

            sio.loadmat(outfile, results)

            print("Results loaded.")

        except subprocess.CalledProcessError as e:
            logger.info((e.output, e.returncode))
        finally:
            if verbose and os.path.isfile(command_output_name):
                with open(command_output_name, 'r') as f:
                    for line in iter(f.readline, ''):
                        print(line)
            if verbose and os.path.isfile(command_err_name):
                with open(command_err_name, 'r') as f:
                    for line in iter(f.readline, ''):
                        print(line)

    return results
