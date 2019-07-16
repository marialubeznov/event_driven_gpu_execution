#!/usr/bin/env python

from optparse import OptionParser
import os
import subprocess
import sys
import re
import shutil
import glob
import datetime

#######################################################################################
# Class the represents each configuration you are going to run
# For example, if your sweep file has 2 entries 32k-L1 and 64k-L1 there will be 2
# ConfigurationSpec classes and the run_subdir name for each will be 32k-L1 and 64k-L1
# respectively
class ConfigurationSpec:
    #########################################################################################
    # Public Interface methods
    #########################################################################################
    # Class is constructed with a single line of text from the sweep_param file
    def __init__(self, init_str):
        [self.run_subdir, self.params] = re.split(":",init_str, 1)

    def my_print(self):
        print "\nRun Subdir = " + self.run_subdir
        print "Parameters = " + self.params

    def run(self, benchmarks, run_directory, cuda_version, libdir, gem5_bin, gem5_config, gem5_debug_flags):
        for benchmark in benchmarks:
            if options.clcuda == "":
                 full_bin_dir = this_directory +\
                      "../../bin/" + "/" + benchmark + "/"
            else:
                full_bin_dir = this_directory +\
                       "../../bin/" + "/" + options.clcuda + "/" + benchmark + "/"

            self.parse_and_setup_commandline_args(benchmark, full_bin_dir)

            for args in self.command_line_args_list:
                this_run_dir = run_directory +\
                            "/" + benchmark + "/" + self.benchmark_args_subdirs[args] +\
                            "/" + self.run_subdir + "/"

                self.setup_run_directory(full_bin_dir, this_run_dir)
                self.text_replace_torque_sim(full_bin_dir,this_run_dir,benchmark,cuda_version, args, libdir, gem5_bin, gem5_config, gem5_debug_flags)
                self.append_gpgpusim_config(full_bin_dir, this_run_dir)

                # Submit the job to torque and dump the output to a file
                if not options.no_launch:
                    torque_out_filename = this_directory + "torque_out.txt"
                    torque_out_file = open(torque_out_filename, 'w+')
                    saved_dir = os.getcwd()
                    os.chdir(this_run_dir)

                    torque_out = "";
                    if subprocess.call(["qsub",\
                                        "-W", "umask=022",\
                                       this_run_dir + "edge_torque.sim"],\
                                       stdout=torque_out_file) < 0:
                        exit("Error Launching Torque Job")
                    else:
                        # Parse the torque output for just the numeric ID
                        torque_out_file.seek(0)
                        torque_out = re.sub(r"(^\d+).*", r"\1",
                            torque_out_file.read().strip())
                        print("Job " + torque_out + " queued (" +\
                            benchmark + "-" + self.benchmark_args_subdirs[args] +\
                            " " + self.run_subdir + ")")

                    torque_out_file.close()
                    os.remove(torque_out_filename)
                    os.chdir(saved_dir)

                    # Dump the benchmark description to the logfile
                    if not os.path.exists(this_directory + "logfiles/"):
                        os.makedirs(this_directory + "logfiles/")
                    now_time = datetime.datetime.now()
                    day_string = now_time.strftime("%y.%m.%d-%A")
                    time_string = now_time.strftime("%H:%M:%S")
                    logfile = open(this_directory +\
                                   "logfiles/simulation_log." +\
                                   day_string + ".txt",'a')
                    print >> logfile, "%s %6s %-22s %-100s %-25s %s" %\
                           ( time_string ,\
                           torque_out ,\
                           benchmark ,\
                           self.benchmark_args_subdirs[args] ,\
                           self.run_subdir,\
                           benchmark)
                    logfile.close()
            self.benchmark_args_subdirs.clear()
            del self.command_line_args_list[:]



    #########################################################################################
    # Internal utilizty methods
    #########################################################################################
    def parse_and_setup_commandline_args(self, benchmark, full_bin_dir):
        # get the command line arguments
        command_line_arg_file = full_bin_dir + "commandline.cuda"
        self.command_line_args_list = []
        if(os.path.isfile(command_line_arg_file)):
            f = open(command_line_arg_file)
            for line in f.readlines():
                if (re.match(r'^(?!#).*$', line) and not re.match("^\s+$", line)):
                    self.command_line_args_list.append(line.strip())
            f.close()
        if len(self.command_line_args_list) == 0:
            self.command_line_args_list.append("")

        self.benchmark_args_subdirs = {}
        for args in self.command_line_args_list:
            if args == "":
                self.benchmark_args_subdirs[args] = "NO_ARGS"
            else:
                self.benchmark_args_subdirs[args] = re.sub(r"[^a-z^A-Z^0-9]", "_", args.strip())



    # copies and links the necessary files to the run directory
    def setup_run_directory(self, full_bin_dir, this_run_dir):
        if not os.path.isdir(this_run_dir):
            os.makedirs(this_run_dir)

         # TODO: Going to want the power XML file
#        files_to_copy_to_run_dir = glob.glob(full_bin_dir + "*.ptx") +\
#                                   glob.glob(full_bin_dir + "*.cl") +\
#                                   glob.glob(full_bin_dir + "*.h") +\
#                                   glob.glob(os.path.dirname(options.gpgpusim_conf_file) + "/*.icnt") +\
#				   glob.glob(os.path.dirname(options.gpgpusim_conf_file) + "/*.xml")
#
#        for file_to_cp in files_to_copy_to_run_dir:
#            new_file = this_run_dir +\
#                       os.path.basename(this_directory + file_to_cp)
#            if os.path.isfile(new_file):
#                os.remove(new_file)
#            shutil.copyfile(file_to_cp,new_file)
#
#        # link the data directory
        if os.path.isdir(full_bin_dir + "data"):
            if os.path.lexists(this_run_dir + "data"):
                os.remove(this_run_dir + "data")
            os.symlink(full_bin_dir + "data", this_run_dir + "data")



    # replaces all the "REAPLCE_*" strings in the edge_torque.sim file
    def text_replace_torque_sim( self,full_bin_dir,this_run_dir,benchmark, cuda_version, command_line_args,
                                 libpath, gem5_bin, gem5_config, gem5_debug_flags ):
        # get the pre-launch sh commands
        prelaunch_filename =  full_bin_dir +\
                             "benchmark_pre_launch_command_line.txt"
        benchmark_command_line = ""
        if(os.path.isfile(prelaunch_filename)):
            f = open(prelaunch_filename)
            benchmark_command_line = f.read().strip()
            f.close()


        # Tayler
        exec_name = options.benchmark_exec_prefix + " " + \
                gem5_bin + " " + gem5_debug_flags + " " + gem5_config + " -c " + this_directory + "../../bin/" + cuda_version + "/" + benchmark

        # do the text replacement for the torque.sim file
        replacement_dict = {"NAME":benchmark + "-" + self.benchmark_args_subdirs[command_line_args],
                            "NODES":"1",
                            "LIBPATH": libpath,
                            "SUBDIR":this_run_dir,
                            "BENCHMARK_SPECIFIC_COMMAND":benchmark_command_line,
                            "PATH":os.getenv("PATH"),
                            "EXEC_NAME":exec_name,
                            "COMMAND_LINE":command_line_args}

        torque_text = open(this_directory + "edge_torque.sim").read().strip()
        for entry in replacement_dict:
            torque_text = re.sub("REPLACE_" + entry,
                                 str(replacement_dict[entry]),
                                 torque_text)
        open(this_run_dir + "edge_torque.sim", 'w').write(torque_text)



    # replaces all the "REPLACE_*" strings in the gpgpusim.config file
    def append_gpgpusim_config(self,full_bin_dir,this_run_dir):
        benchmark_spec_opts_file = full_bin_dir + "benchmark_options.txt"
        benchmark_spec_opts = ""
        if(os.path.isfile(benchmark_spec_opts_file)):
            f = open(benchmark_spec_opts_file)
            benchmark_spec_opts_line_args = f.read().strip()
            f.close()

        # do the text replacement for the config file
        replacement_dict = {"BENCHMARK_SPEC_OPT":benchmark_spec_opts,
                            "SWEEP_PARAM":self.params}
        config_text_file = os.path.join(os.getcwd(), options.gpgpusim_conf_file)
        config_text = open(config_text_file).read()
        config_text += "\n" + benchmark_spec_opts + "\n" + self.params

        edge_config_filename = "edge_torque.config";
        open(this_run_dir + edge_config_filename, 'w').write("\n" + benchmark_spec_opts + "\n" + self.params + "\n");

# This function exists so that this file can accept both absolute and relative paths
# If no name is provided it sets the default
# Either way it does a test if the absolute path exists and if not, tries a relative path
def file_option_test(name, default):
    if name == "":
        name = os.path.join(this_directory, default)
    try:
        with open(name): pass
    except IOError:
        name = os.path.join(os.getcwd(), name)
        try:
            with open(name): pass
        except IOError:
            exit("Error - cannot open file {0}".format(name))
    return name

def setup_sdk_data_links(cuda_version):
    bin_top = os.path.join( this_directory, "..", "..", "bin")
    bin_dir_name = os.path.join( bin_top, "sdk", cuda_version )

    for dir in os.listdir( bin_dir_name ):
        if os.path.isdir( os.path.join( bin_dir_name, dir, "data" ) ):

            # Link the benchmark dir to the bin dir
            staged_bin_dir = os.path.join(bin_top, dir)

            if not os.path.exists( staged_bin_dir ):
                os.makedirs(staged_bin_dir)
            staged_bin_dir = os.path.join(staged_bin_dir, "data")
            if os.path.islink(staged_bin_dir):
                os.unlink(staged_bin_dir)

            bench_data_dir = os.path.join(bin_dir_name, dir)
            if not os.path.exists(bench_data_dir):
                os.makedirs(bench_data_dir)

            bench_data_dir = os.path.join(bench_data_dir, "data")
            os.symlink(bench_data_dir, staged_bin_dir)

#-----------------------------------------------------------
# main script start
#-----------------------------------------------------------
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"

parser = OptionParser()
parser.add_option("-w", "--sweep_file", dest="sweep_file",
                  help="sweep_file used to determine which configurations are run",
                  default="")
parser.add_option("-i", "--include_file", dest="include_file",
                  help="include_file used to define which benchmarks are run",
                  default="")
parser.add_option("-c", "--changelist", dest="changelist",
                  help="Specify a version of the simulator to run",
                  default="")
parser.add_option("-p", "--benchmark_exec_prefix", dest="benchmark_exec_prefix",
                 help="When submitting the job to torque this string" +\
                 " is placed before the command line that runs the benchmark. " +\
                 " Useful when wanting to run valgrind.", default="")
parser.add_option("-g", "--gpgpusim_conf_file", dest="gpgpusim_conf_file",
                  help="The baseline gpgpusim config file that will be used can contain"+\
                    "REPALCE_* strings that the pipeline will fill in",
                  default="")
parser.add_option("-r", "--run_directory", dest="run_directory",
                  help="Name of directory in which to run simulations",
                  default="")
parser.add_option("-d", "--clcuda_dir", dest="commandlinecuda",
                  help="Name of directory in which the benchmarks commandline.cuda files are stored",
                  default="")
parser.add_option("-n", "--no_launch", dest="no_launch", action="store_true",
                  help="When set, no torque jobs are launched.  However, all"+\
                  " the setup for running is performed. ie, the run"+\
                  " directories are created and are ready to run."+\
                  " This can be useful when you want to create a new" +\
                  " configuration, but want to test it locally before "+\
                  " launching a bunch of jobs.")
parser.add_option("-G", "--git_dir", dest="git_dir",
                  help="By default this script assumes that the simulator code is stored in perforce and is built" +\
                       " into the ../../bin/sim/CLXXX directory.  If you want to run this script with the git" +\
                       " version of the simulator - point this option to the top level directory of your git clone.",
                       default="")

# Tayler
parser.add_option("-b", "--gem5-binary", dest="gem5_bin",
                    help="Specifies the full path to the gem5 binary (e.g., gem5.debug) to launch the simulator",
                    default="")

parser.add_option("-e", "--gem5-config", dest="gem5_config",
                    help="Specifies the full path to the gem5 configuration file (e.g., se_fusion.py) to configure the simulator",
                    default="")

parser.add_option("-f", "--gem5-debug-flags", dest="gem5_debug_flags",
                    help="Debug flags to pass to the Gem5-GPU Simulator",
                    default="")

(options, args) = parser.parse_args()

# Parser seems to leave some whitespace on the options, getting rid of it
options.sweep_file = options.sweep_file.strip()
options.changelist = options.changelist.strip()
options.benchmark_exec_prefix = options.benchmark_exec_prefix.strip()
options.gpgpusim_conf_file = options.gpgpusim_conf_file.strip()
options.include_file = options.include_file.strip()
options.run_directory = options.run_directory.strip()
options.clcuda = options.commandlinecuda.strip()
options.git_dir = options.git_dir.strip()

# Tayler
options.gem5_bin = options.gem5_bin.strip()
options.gem5_config = options.gem5_config.strip()
options.gem5_debug_flags = options.gem5_debug_flags.strip()

#if str(os.getenv("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN")) != "1":
#    sys.exit("ERROR - Please run setup_environment before running this script")

# Get CUDA version
nvcc_out_filename = this_directory + "nvcc_out.txt"
nvcc_out_file = open(nvcc_out_filename, 'w+')
subprocess.call(["nvcc", "--version"],\
               stdout=nvcc_out_file)
nvcc_out_file.seek(0)
cuda_version = re.sub(r".*release (\d+\.\d+).*", r"\1", nvcc_out_file.read().strip().replace("\n"," "))
nvcc_out_file.close()
os.remove(nvcc_out_filename)

#setup_sdk_data_links(cuda_version)

## If you want to use git - setup the git variables here
#git_config = None
#if options.git_dir != "":
#    git_config = GitConfig( options.git_dir, os.path.join( "lib", os.getenv("GPGPUSIM_CONFIG") ))
## No changelist specified, just use the newest simulator
#elif options.changelist == "":
#    simulator_dir = this_directory + "../../bin/sim"
#    all_simulator_files = [os.path.join(simulator_dir, f) \
#                       for f in os.listdir(simulator_dir)]
#    options.changelist = os.path.basename(max(all_simulator_files, key=os.path.getmtime))


options.include_file = file_option_test(options.include_file, "example/include.list")
options.gpgpusim_conf_file = file_option_test(options.gpgpusim_conf_file, "example/baseline.config")
options.sweep_file = file_option_test(options.sweep_file, "example/sweep.list")

if options.run_directory == "":
    options.run_directory = os.path.join(this_directory, "../../run_%s"%cuda_version)
else:
    options.run_directory = os.path.join(os.getcwd(), options.run_directory)


# Test for the existance of torque on the system
if not any([os.path.isfile(os.path.join(p, "qsub")) for p in os.getenv("PATH").split(os.pathsep)]):
    exit("ERROR - Cannot find qsub in PATH... Is torque installed on this machine?")

if not any([os.path.isfile(os.path.join(p, "nvcc")) for p in os.getenv("PATH").split(os.pathsep)]):
    exit("ERROR - Cannot find nvcc PATH... Is CUDA_INSTALL_PATH/bin in the system PATH?")

# An array of all the benchmarks in the include.list file
# match functions are to remove commented out lines and lines with only whitespace
benchmarks = [line.strip() for line in open(options.include_file) \
              if (re.match(r'^(?!#).*$', line) and not re.match("^\s+$", line))]

# Construct a new ConfigurationSpec class for each line in the sweep file
# match functions are to remove commented out lines and lines with only whitespace
configurations = [ConfigurationSpec(line.strip()) for line in open(options.sweep_file) \
                 if (re.match(r'^(?!#).*$', line) and not re.match("^\s+$", line))]

version_string = "Gem5-GPU Simulator"
#version_string = ""
#if git_config == None:
#    version_string = "Perforce CL " + options.changelist
#else:
#    version_string = str(git_config)

print("\n\n==================================" +
      "\nRunning Simulations with GPGPU-Sim built for \n{0}\n ".format(version_string) +
      "\nUsing sweep_file " + options.sweep_file +
      "\nBaseline Config File " + options.gpgpusim_conf_file +
      "\nInclude File " + options.include_file +
      "\n====================================\n\n ")

libdir = "/home/common/gcc_opt/gcc-4.8.4/lib64:/home/common/gcc_opt/gcc-4.8.4/lib/"
#if git_config == None:
#    libdir = this_directory + "../../bin/sim/" + options.changelist + "/" + os.getenv("GPGPUSIM_CONFIG") + "/"
#else:
#    libdir = os.path.join( git_config.git_dir, os.getenv("GPGPUSIM_CONFIG") ) + "/"



for config in configurations:
    config.my_print()
    config.run(benchmarks, options.run_directory, cuda_version, libdir, options.gem5_bin, options.gem5_config, options.gem5_debug_flags)

