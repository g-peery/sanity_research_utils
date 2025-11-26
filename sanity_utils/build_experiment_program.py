import os
from mpi4py import MPI
from .mpi_utils import (get_global_logger, ResultsCSV, ExperimentTimer,
        prepare_experiment_dir)
from glob import glob


DEFAULT_LOG_FILE_NAME = "log.txt"

DEFAULT_TABLE_FILE_NAME = "table.csv"


def sanitize_filename(filename):
    """Returns None if filename None, else sanitized filename."""
    if filename is None:
        return None
    return os.path.basename(filename)


def build_experiment_program(
    parser,
    argument_mappings,
    return_mappings,
    runner,
    log_file_name=DEFAULT_LOG_FILE_NAME,
    table_file_name=DEFAULT_TABLE_FILE_NAME,
):
    """Wrapper for function `runner` to set up logging faculties

    * `parser`: `argparse.ArgumentParser` with arguments loaded up
    * `argument_mappings`: iterable of tuples
    (`parsed_arg`, `column`, `include_in_exp_desc`)
        * `parsed_arg`: str entry into `args` parsed from `parser`
        * `column`: str or None to include with table under results, if
        None then does not make it a column, if str then names the
        column this under the .csv
        * `include_in_exp_desc`: bool or Callable, if True or a
        callable, then adds a field corresponding to this argument in
        the experiment description. If a Callable, then calls it with
        the field as an argument to retrieve a string
    * `return_mappings`: iterable of (str or None); size must correspond
    to length of iterable returned by `runner`, and each entry
    corresponds to either the column name within which to put the output
    of `runner` or None if not to include it
    * `runner`: Callable with signature `runner(exp_dir, args)` where
    `exp_dir` is the directory which `runner` may save things in and
    `args` is parsed from `parser`
    * `log_file_name`: str name of .txt file under the results directory
    at which to log console output
    * `table_file_name`: str name of .csv file under the results
    directory at which to create a table

    Side Effects:
    * Adds arguments to parser including `results` and `overwrite`; see
    --help for their descriptions.
    * Creates directory structure starting with `results` directory
    specified by supplied arguments (if not already present), a
    subdirectory with name corresponding to fields indicated by
    `argument_mappings` and contents according to `runner` as the path
    to it is supplied in `runner`'s first argument `exp_dir`
    * Creates a file under `results`/`field_meanings.txt` specifying 
    what each of the fields means in the experiment directories.
    * Appends to or creates a log file at `log_file_name`
    * Appends to or creates a table with results at `table_file_name`,
    possibly with a modified name if column structure does not match.

    Returns:
    * Callable with signature `wrapped_runner()`. On call, parses
    arguments, calls `runner`, saves to disk as indicated, and forwards
    return from `runner`
    """
    # Add field to parser
    parser.add_argument(
        "-r",
        "--results",
        metavar="RESULTS_DIR",
        default="results",
        help="superdirectory in which all solutions are saved"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow overriding an experiment directory"
    )

    #
    # Construct columns
    #
    _arg_cols = [
        column for _, column, _ in argument_mappings if (column is not None)
    ]
    # Validate no repeats
    _s_arg_cols = set(_arg_cols)
    if len(_arg_cols) != len(_s_arg_cols):
        raise ValueError("argument_mapping columns must all be unique")

    _ret_cols = [column for column in return_mappings if (column is not None)]
    # Validate no repeats
    _s_ret_cols = set(_ret_cols)
    if len(_ret_cols) != len(_s_ret_cols):
        raise ValueError("return_mapping columns must all be unique")

    # Validate no intersection between
    if len(_s_arg_cols.intersection(_s_ret_cols)) > 0:
        raise ValueError(
            "return_mapping columns must not be the same as "
            "argument_mapping columns"
        )

    _provided_cols = _arg_cols + _ret_cols
    # Validate "time" not a repeat
    if "time" in _provided_cols:
        raise ValueError("The column name 'time' is reserved")

    columns = _provided_cols + ["time"]

    #
    # Validate filenames
    #
    if log_file_name[-4:] != ".txt":
        raise ValueError("Log file name must end in .txt")

    if table_file_name[-4:] != ".csv":
        raise ValueError("Table file name must end in .csv")

    #
    # Construct wrapped function
    #
    def wrapped_runner():
        #
        # Parse arguments
        #
        args = parser.parse_args()

        #
        # Prepare and validate disk structure
        #
        # Construct experiment directory fields
        exp_dir_fields = [ ]
        exp_dir_field_meanings = [ ]
        for arg, _, exp_desc in argument_mappings:
            if exp_desc: # True or callable
                if callable(exp_desc):
                    this_value = exp_desc(getattr(args, arg))
                else:
                    this_value = getattr(args, arg)
                exp_dir_fields.append(this_value)
                exp_dir_field_meanings.append(arg)

        # Prepare results and experiment directory (will create both)
        exp_dir = prepare_experiment_dir(
            args.results,
            *exp_dir_fields,
            overwrite=args.overwrite
        )

        # Write meaning of fields to exp_dir to allow inconsistencies to
        # be recovered from
        with open(os.path.join(exp_dir, "field_meanings.txt"), "w") as fd:
            fd.write("\n".join(exp_dir_field_meanings))

        # Logger
        log_file = os.path.join(args.results, log_file_name)
        logger = get_global_logger(log_file)

        # Results CSV
        result_csv_path = os.path.join(args.results, table_file_name)
        try:
            csv = ResultsCSV(result_csv_path, columns)
        except ValueError as e:
            if os.path.exists(result_csv_path):
                # Simply need to re-name
                csv_glob = glob(result_csv_path[:-4]+"_*.csv")

                if len(csv_glob) == 0:
                    i = 2
                else:
                    i = max([
                        int(s[len(result_csv_path)-4:-4])
                        for s in csv_glob
                    ]) + 1

                csv = ResultsCSV(
                    result_csv_path[:-4]+"_"+str(i)+".csv",
                    columns
                )
            else:
                # Something else went wrong; re-raise
                raise(e)

        # Record number of processes
        logger.info(
            f"Detected number of processes: {MPI.COMM_WORLD.Get_size()}"
        )

        # Record column values
        arg_info = [ ]
        for arg, column, _ in argument_mappings:
            logger.info(f"{column}: {getattr(args, arg)}")
            arg_info.append(getattr(args, arg))

        #
        # Call function with timing
        #
        logger.info("Beginning run now.")
        # Timer start
        timer = ExperimentTimer()
        timer.start()
        
        # Peform task
        ret = runner(exp_dir, args)

        # Timer end
        total_time = timer.end()
        logger.info(f"Finished run. Took {total_time} seconds to complete")

        #
        # Save and forward results
        #
        # Determine result columns
        ret_info = [ ]
        for c_i, column in enumerate(return_mappings):
            if column is not None:
                ret_info.append(ret[c_i])

        # Update DataFrame
        csv.add_row(arg_info+ret_info+[float(total_time)])
        # Save into file
        csv.save()

        # Forward return
        return ret

    return wrapped_runner

