"""Primary, command-line entry point for Illustris Black holes module.

"""
from datetime import datetime
import numpy as np
import os
import warnings
from illbh import constants
from . import details
from . import mergers

_DEF_OUTPUT_DIR = './'
_DEF_VERB = True
_ILLUSTRIS_OUTPUT_DIR_BASE = "/n/ghernquist/Illustris/Runs/{:s}/"  # output/"


def main():
    beg_time = datetime.now()
    args, output_dir, input_dir = load_args()
    num_snaps = constants.get_illustris_metadata(args.run, constants.META.NUM_SNAPS)

    # Say Hello
    if args.verbose:
        git_vers = constants._get_git()
        title_str = "Illustris Black Holes Module, version: {}".format(git_vers)
        print("\n\n{}\n{}\n{}\n".format(title_str, '='*len(title_str), beg_time.ctime()))
    print("Input  directory: '{}'".format(input_dir))
    print("Output directory: '{}'".format(output_dir))

    # Make 'raw' 'organized' BH Merger Files
    # --------------------------------------
    if args.verbose:
        print("\nCombining merger files")
    mergers_combined_good = _check_mergers_combined(args, output_dir)
    # Run `combine_raw_merger_files` as needed
    if args.refresh_all or not mergers_combined_good:
        mergers.combine_raw_merger_files(args.run, verbose=args.verbose, output_dir=output_dir)
    if not _check_mergers_combined(args, output_dir):
        raise RuntimeError("Combined merger files are not okay!")

    # Process Details
    # ---------------
    if args.verbose:
        print("\nProcessing Details")
    # Organize details entries by snapshot, in txt files
    details_organized_txt_good = _check_details_organized_txt(args, output_dir, num_snaps)
    if args.refresh_all or not details_organized_txt_good:
        if args.verbose:
            print(" - Organizing Details by snapshot, as 'txt'")
        details.organize_txt_by_snapshot(args.run, verbose=args.verbose, output_dir=output_dir)
    if not _check_details_organized_txt(args, output_dir, num_snaps):
        raise RuntimeError("Details organized by snapshot ('txt') are not okay!")

    # Convert txt files to hdf5 files
    details_organized_hdf5_good = _check_details_organized_hdf5(args, output_dir, num_snaps)
    if args.refresh_all or not details_organized_hdf5_good:
        if args.verbose:
            print("\n - Converting Details by snapshot to 'hdf5'")
        details.convert_txt_to_hdf5(args.run, verbose=args.verbose, output_dir=output_dir)
    if not _check_details_organized_hdf5(args, output_dir, num_snaps):
        raise RuntimeError("Details organized by snapshot ('hdf5') are not okay!")

    # Downsample details and find merger-details
    details_good = _check_final_details(args, output_dir)
    if args.refresh_all or not details_good:
        if args.verbose:
            print("\n - Downsampling details and finding merger-details")
        details.combine_downsample_and_mergers_hdf5(
            args.run, verbose=args.verbose, output_dir=output_dir)
    if not _check_final_details(args, output_dir):
        raise RuntimeError("Details and merger-details are not okay!")

    # Make Final Mergers File
    # -----------------------
    mergers_good = _check_final_mergers(args, output_dir)
    if args.refresh_all or not mergers_good:
        if args.verbose:
            print("\nCombining mergers and merger-details")
        mergers.combine_mergers_with_details(
            args.run, verbose=args.verbose, output_dir=output_dir)
    if not _check_final_mergers(args, output_dir):
        raise RuntimeError("Final mergers are not okay!")

    end_time = datetime.now()
    print("All complete at {}, After {}".format(end_time, end_time-beg_time))
    return


def load_args(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Black hole module for the Illustris simulations.")
    parser.add_argument('--verbose', '-v', dest='verbose', default=_DEF_VERB, action='store_true',
                        help='Print more messages to the screen.')
    parser.add_argument('--update', '-u', dest='refresh_all', default=True, action='store_false',
                        help="Only recreate files that are outdated.  Otherwise, recreate all.")

    parser.add_argument('--output', dest='output_dir', default=None,
                        help="Destination directory for output "
                             "(Default: a directory called '[RUN]' will be created in './').")
    parser.add_argument('--run_dir', dest='input_dir', default=None,
                        help="Directory in which to find the target `run` "
                             "(Default: '/n/ghernquist/Illustris/Runs/').")
    parser.add_argument('--run', '-r', dest='run', default='L75n1820FP',
                        help="Simulation run name (Default: 'L75n1820FP').")

    args = parser.parse_args(args=args)

    # Setup input directory
    input_dir = args.input_dir
    if input_dir is None:
        input_dir = _ILLUSTRIS_OUTPUT_DIR_BASE.format(args.run)
        if not os.path.isdir(input_dir):
            raise ValueError("'{}' is invalid.  Could not construct path using run '{}'.".format(
                input_dir, args.run))

        if not _check_has_output_dir(input_dir):
            raise ValueError("Could not find an 'output' directory in '{}'".format(input_dir))

    if not _check_has_output_dir(input_dir):
        expand_input_dir = os.path.append(input_dir, args.run, '')
        warnings.warn("`input_dir`: '{}' doesn't have an output directory, trying '{}'.".format(
            input_dir, expand_input_dir))
        if _check_has_output_dir(expand_input_dir):
            input_dir = expand_input_dir
        else:
            raise ValueError("Could not find an 'output' directory")

    # Setup Output Directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.abspath(_DEF_OUTPUT_DIR), args.run, '')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir):
        raise ValueError("`output_dir` '{}' is invalid.".format(output_dir))

    return args, output_dir, input_dir


def _check_mergers_combined(args, output_dir):
    # Make sure files exist and are up to date
    mergers_raw_fname = constants.GET_MERGERS_COMBINED_FILENAME(
        args.run, filtered=False, type='txt', output_dir=output_dir)
    mergers_filt_fname = constants.GET_MERGERS_COMBINED_FILENAME(
        args.run, filtered=True, type='hdf5', output_dir=output_dir)
    raw_exists = os.path.isfile(mergers_raw_fname)
    filt_exists = os.path.isfile(mergers_filt_fname)
    if filt_exists:
        filt_current = mergers._check_version(mergers_filt_fname)
    else:
        filt_current = False
    is_good = (raw_exists and filt_exists and filt_current)
    if args.verbose:
        print(" - File '{}' exists: {}".format(mergers_raw_fname, raw_exists))
        print(" - File '{}' exists: {}, Current: {}".format(
            mergers_filt_fname, filt_exists, filt_current))
    return is_good


def _check_details_organized_txt(args, output_dir, num_snaps):
    output_fnames = \
        [constants.GET_DETAILS_ORGANIZED_FILENAME(args.run, snap, type='txt', output_dir=output_dir)
         for snap in range(num_snaps)]
    is_good = np.all([os.path.exists(ofn) for ofn in output_fnames])
    if args.verbose:
        print(" - Files '{}'... all exist: {}".format(output_fnames[0], is_good))
    return is_good


def _check_details_organized_hdf5(args, out_dir, num_snaps):
    output_fnames = \
        [constants.GET_DETAILS_ORGANIZED_FILENAME(args.run, snap, type='hdf5', output_dir=out_dir)
         for snap in range(num_snaps)]
    is_good = np.all([os.path.exists(ofn) for ofn in output_fnames])
    if is_good:
        is_current = np.all([details._check_version(ofn) for ofn in output_fnames])
    else:
        is_current = False
    if args.verbose:
        print(" - Files '{}'... all exist: {}, all current: {}".format(
            output_fnames[0], is_good, is_current))
    is_good = (is_good and is_current)
    return is_good


def _check_final_details(args, output_dir):
    mdet_fname = constants.GET_MERGERS_DETAILS_FILENAME(args.run, output_dir=output_dir)
    dets_fname = constants.GET_PUBLIC_DETAILS_FILENAME(args.run, output_dir=output_dir)
    mdet_exists = os.path.isfile(mdet_fname)
    dets_exists = os.path.isfile(dets_fname)
    mdet_current = dets_current = False
    if mdet_exists:
        mdet_current = details._check_version(mdet_fname)
    if dets_exists:
        dets_current = details._check_version(dets_fname)

    if args.verbose:
        print(" - File '{}' exists: {}, Current: {}".format(
            mdet_fname, mdet_exists, mdet_current))
        print(" - File '{}' exists: {}, Current: {}".format(
            dets_fname, dets_exists, dets_current))

    is_good = (mdet_exists and mdet_current and dets_exists and dets_current)
    return is_good


def _check_final_mergers(args, output_dir):
    mergers_fname = constants.GET_PUBLIC_MERGERS_FILENAME(args.run, output_dir=output_dir)
    exists = os.path.isfile(mergers_fname)
    if exists:
        current = mergers._check_version(mergers_fname)
    else:
        current = False
    if args.verbose:
        print(" - File '{}' exists: {}, Current: {}".format(
            mergers_fname, exists, current))
    is_good = exists and current
    return is_good


def _check_has_output_dir(path):
    # Get list of directories in `input_dir`
    list_dir = [ld for ld in os.listdir(path) if os.path.isdir(os.path.join(path, ld))]
    return ('output' in list_dir)


if __name__ == "__main__":
    main()
