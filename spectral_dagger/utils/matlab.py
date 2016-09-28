from __future__ import print_function
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


def make_verbose_print(verbosity, threshold=1.0):
    def vprint(*obj):
        if float(verbosity) >= float(threshold):
            print(*obj)
    return vprint


def run_matlab_code(code, working_dir='.', verbose=False, delete_files=False, **matlab_kwargs):

    vprint = make_verbose_print(verbose)

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

    vprint("Process %d starting Matlab section." % pid)
    t0 = time.time()
    results = {}

    with cd(working_dir):
        sio.savemat(infile, matlab_kwargs)

        process = None
        try:
            vprint("Calling matlab with command: ")
            vprint("    %s" % command)

            with open(command_output_name, 'w') as command_output:
                with open(command_err_name, 'w') as command_err:
                    process = subprocess.Popen(
                        command, stdout=command_output, stderr=command_err)
                    process.wait()

            vprint("Matlab call complete, took %s seconds. "
                   "Loading results..." % (time.time() - t0))

            sio.loadmat(outfile, results)

            vprint("Results loaded.")

        except subprocess.CalledProcessError as e:
            logger.info((e.output, e.returncode))
            raise e
        finally:
            if isinstance(process, subprocess.Popen):
                try:
                    process.terminate()
                except OSError:
                    pass
            try:
                if verbose and os.path.isfile(command_output_name):
                    with open(command_output_name, 'r') as f:
                        for line in iter(f.readline, ''):
                            print(line)
            except IOError:
                pass

            try:
                if verbose and os.path.isfile(command_err_name):
                    with open(command_err_name, 'r') as f:
                        for line in iter(f.readline, ''):
                            print(line)
            except IOError:
                pass

            if delete_files:
                try:
                    os.remove(infile)
                except OSError:
                    pass

                try:
                    os.remove(outfile)
                except OSError:
                    pass

                try:
                    os.remove(command_output)
                except OSError:
                    pass

                try:
                    os.remove(command_err)
                except OSError:
                    pass

    return results
