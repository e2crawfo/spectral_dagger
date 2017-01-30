from __future__ import print_function
import time
import subprocess
import os
import logging
import scipy.io as sio

from spectral_dagger.utils.misc import cd

logger = logging.getLogger(__name__)


def make_verbose_print(verbosity, threshold=1.0):
    def vprint(*obj):
        if float(verbosity) >= float(threshold):
            print(*obj)
    return vprint


def safe_print_file(filename):
    try:
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                for line in iter(f.readline, ''):
                    print(line)
    except IOError:
        pass


def safe_remove_file(filename):
    try:
        os.remove(filename)
    except (OSError, TypeError):
        pass


def run_matlab_code(code, working_dir='.', verbose=True, delete_files=False, **matlab_kwargs):

    vprint = make_verbose_print(verbose)

    pid = os.getpid()
    infile = 'matlab_input_pid_%d.mat' % pid
    outfile = 'matlab_output_pid_%d.mat' % pid

    code = code.format(infile=infile, outfile=outfile)
    command = [
        "matlab", "-nosplash", "-nodisplay", "-nojvm",
        "-nodesktop",  "-r",
        "try, %s, catch exception, "
        "display(getReport(exception)), exit(1), end, exit;" % code]

    command_output_name = "stdout_%d.txt" % pid
    command_err_name = "stderr_%d.txt" % pid

    vprint("Process %d starting Matlab section." % pid)
    t0 = time.time()
    results = {}

    with cd(working_dir):
        sio.savemat(infile, matlab_kwargs)

        process = None
        error = False
        try:
            vprint("Calling matlab with command: ")
            vprint("    %s" % command)

            with open(command_output_name, 'w') as command_output:
                with open(command_err_name, 'w') as command_err:
                    process = subprocess.Popen(
                        command, stdout=command_output, stderr=command_err)
                    process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command, None)

            vprint("Matlab call complete, took %s seconds. "
                   "Loading results..." % (time.time() - t0))

            sio.loadmat(outfile, results)

            vprint("Results loaded.")

        except:
            error = True
            raise
        finally:
            if isinstance(process, subprocess.Popen):
                try:
                    process.terminate()
                except OSError:
                    pass

            if error or verbose:
                safe_print_file(command_output_name)
                safe_print_file(command_err_name)

            if delete_files:
                safe_remove_file(infile)
                safe_remove_file(outfile)
                safe_remove_file(command_output_name)
                safe_remove_file(command_err_name)

    return results
