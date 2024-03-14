import logging
from mpi4py import MPI
import os
import pandas as pd
import sys
from .consts import FORMAT
import timeit


_comm = MPI.COMM_WORLD
_rank = _comm.Get_rank()


def get_global_logger(
        log_file,
        main_lvl=logging.DEBUG,
        second_lvl=logging.ERROR
    ):
    """
    Returns a logger that has nice settings, indicates MPI rank.

    The process with rank=0 will use main_lvl, the rest second_lvl
    """
    format_str = f"[{_rank}] " + FORMAT

    # Interpret level by rank
    if _rank == 0:
        level = main_lvl
    else:
        level = second_lvl

    # Create logger
    logger = logging.getLogger("util")

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)

    # Create formatter, register it
    formatter = logging.Formatter(format_str)
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    # Return
    return logger


def prepare_experiment_dir(result_dir, *args):
    """
    Given a results/ directory, creates a subdirectory with
    names provided in *args, separated by hyphens.
    For that reason, good to not include hyphens in args.

    If it already exists, throws FileExistsError.

    Handles MPI stuff when all processes call.
    
    Returns path to resulting directory.
    """
    # Get full name
    basename = "-".join([str(x) for x in args])
    output_dir = os.path.join(result_dir, basename)

    # Have 1st process check for existence
    if _rank == 0:
        should_abort = os.path.exists(output_dir)
        if not should_abort:
            os.makedirs(output_dir)
    else:
        should_abort = None

    # Sync processes
    should_abort = _comm.bcast(should_abort, root=0)

    # All processes raise if necessary
    if should_abort:
        raise FileExistsError(
            f"{output_dir} already exists - cannot proceed safely"
        )

    return output_dir


def create_experiment_logger(
        output_dir,
        main_lvl=logging.DEBUG,
        second_lvl=logging.INFO
    ):
    """
    Creates a logger. Will send info to a file output{rank}.txt
    in the output_dir

    Rank 0 will write <= main_lvl, rest <= second_lvl

    Returns it, destructor that should be called when done
    """
    # Deal with rank
    if _rank == 0:
        level = main_lvl
    else:
        level = second_lvl

    # Create a logger that is a child of util
    logger = logging.getLogger("util.experiment")

    # Create handler
    handler = logging.FileHandler(os.path.join(output_dir, f"output{_rank}.txt"))
    handler.setLevel(level)

    # Prepare formatting (no need to indicate rank)
    handler.setFormatter(logging.Formatter(FORMAT))

    # Register
    logger.addHandler(handler)

    # Prepare destructor - necessary for changing
    # which experiment this logger will refer to
    destructor = lambda : logger.removeHandler(handler)

    return logger, destructor


class ResultsCSV:

    def __init__(self, csv_path, columns):
        """
        Creates a pandas.DataFrame from a csv file, if it exists, else creates a
        new one with requested columns. Also creates the file.

        For MPI, only rank 0 will do anything. Attempts by other rank
        processes to access variable attributes will result in errors;
        those types of operations must be checked manually. Only method
        calls are guarded.

        Throws an error if columns aren't those expected.
        """
        if self._no_pass():
            return

        if os.path.exists(csv_path):
            # Read in what we already have
            df = pd.read_csv(csv_path, index_col=0)

            # Make sure columns match
            read_columns = df.columns.to_list()
            if read_columns != columns:
                raise ValueError(
                    f"The columns of {csv_path}, {read_columns}, do not match "
                    f"expected {columns}"
                )
        else:
            # Create from scratch
            df = pd.DataFrame(columns=columns)

            # Write
            dirname = os.path.dirname(csv_path)
            if dirname != "":
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
            df.to_csv(csv_path)

        self.df = df
        self.csv_path = csv_path

    def _no_pass(self):
        """Returns true if not rank 0"""
        return _rank != 0

    def add_row(self, row):
        """Appends a row to the DataFrame."""
        if self._no_pass():
            return

        self.df.loc[self.df.shape[0]] = row

    def save(self):
        """Saves the DataFrame to the file."""
        if self._no_pass():
            return

        self.df.to_csv(self.csv_path)


class ExperimentTimer:

    def __init__(self):
        """
        An MPI-safe experiment timer.

        All processes should call, but rank 0 will
        do the starting and stopping of things.
        """
        self._start_time = None
        self._end_time = None

    def start(self):
        # Sync processes
        _ = _comm.bcast(None, root=0)

        # Process 0 starts timer
        if _rank == 0:
            self._start_time = timeit.default_timer()
        else:
            self._start_time = 0

        # Sync processes - should be small latency
        _ = _comm.bcast(None, root=0)

    def end(self):
        """Each process gets total elapsed time."""
        # Sync processes
        _ = _comm.bcast(None, root=0)

        # Process 0 ends timer
        if _rank == 0:
            self._end_time = timeit.default_timer()
        else:
            self._end_time = 0

        # Compute
        total_time = self._end_time - self._start_time

        # Sync processes and get common time
        total_time = _comm.bcast(total_time, root=0)

        return total_time

