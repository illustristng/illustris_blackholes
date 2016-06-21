# Illustris Black Holes Post-Processing Module #

This module compiles and processes the raw black hole (BH) output files from the Illustris simulations, and converts them into publicly usable HDF5 files.  The two primary types of BH output from the illustris simulation are 'details' and 'mergers' files:
-   'details' : handled by `illbh.details.py`
    -    These files contain the output from each BH particle active in the simulation at each integration time-step.
    -    The parameters included are 'ID, time, mass, mdot, rho, cs' which are the BH's unique ID number, the time the entry was recorded (in cosmological scale-factor), the mass of the BH particles, the mass-accretion rate of the particle, the local gas density ('rho'), and the local gas sound-speed ('cs').
-   'mergers' : handled by `illbh.mergers.py`
    -    These files contain a record of each BH-BH merger in the Illustris simulations.
    -    The parameters included are 'time, id_out, mass_out*, id_in, mass_in'.  We distinguish the two BH which participate in the merger as 'in' and 'out' BH.  The difference, determined randomly in the code, is which BH ID number persists along with the remnant after the merger.  The 'out' BH survives after the merger, increased in mass by that of the 'in' BH---which no longer exists.

### Public Output Files ###

Two primary output files are produced for each the details and mergers respectively.
-   Details: `ill-[N]_blackhole_details.hdf5` (for illustris simulation number `N`)
    ```
    'Header'
        'script' : Filename of the generating script (`illbh.details.py`)
        'script_version' : Internal version str of the script.
        'git_version' : Git commit SHA for the repository at time of file creation.
        'created' : Time of file creation.
        'simulation' : Illustris simulation number.
        'target_times' : The times (scale-factors) which were the targets for details entries
        'num_entries' : The total number of details entries stored `N`
        'num_blackholes' : The total number of unique BH with details entries `M`
        'description' : A description of the data.

    'unique'
        'id' : (M,) uint64 - ID numbers of each unique BH
        'first_index' : (M,) int - The index number (i.e. row) of the first entry for each unique BH
        'num_entries' : (M,) int - The total number of entries for each unique BH

    'id' : (N,) uint64 - ID number of the BH for each details entry
    'time' : (N,) float64 - Time (cosmological scale-factor) for each entry.
    'mass' : (N,) float64 - BH mass
    'mdot' : (N,) float64 - BH mass accretion rate
    'rho' : (N,) float64 - local gas density
    'cs' : (N,) float64 - local gas sound-speed
    ```

-   Mergers: `ill-[N]_blackhole_mergers.hdf5` (for illustris simulation number `N`)
    ```
    'Header'
        'script' : Filename of the generating script (`illbh.mergers.py`)
        'script_version' : Internal version str of the script
        'git_version' : Git commit SHA for the repository at time of file creation
        'created' : Time of file creation
        'simulation' : Illustris simulation number
        'target_times' : The times (scale-factors) which were the targets for details entries
        'num_mergers' : The total number of mergers stored
        'num_blackholes' : The total number of unique BH participating in mergers
        'description' : A description of the data
        'unique_ids' : The ID numbers of all unique BH participating in mergers
        
    'tree' : Information describing the BH merger tree.  If one of the below events does not exist, 
             the value in the array is '-1', NOTE: not zero.
        'next' (N,) int - The index number of the next merger this remnant takes part in.
        'prev_in' (N,) int - The index number of the previous merger this 'in' BH was part of.
        'prev_out' (N,) int - The index number of the previous merger this 'out' BH was part of. 
    
    'details' : Information from the 'details' files for BH in each merger.  Details entries were
                searched for trying to match the 'in' BH just before merger, and the 'out' BH both
                just before, and just after merger.  This corresponds to the three 'columns' for
                each entry 'row': [0: 'in-bef', 1: 'out-bef', 2: 'out-aft'].  Many of these details
                were not found, in which case the array values are zero.
        'time' (N,3) float64 - Time (scale-factor) for each entry
        'mass' (N,3) float64 - BH mass
        'mdot' (N,3) float64 - BH mass accretion rate
        'rho' (N,3) float64 - local gas density
        'cs' (N,3) float64 - local gas sound-speed
    
    'time' (N,) float64 - Time (scale-factor) for each merger event
    'id_in' (N,) uint64 - ID number of the 'in' BH
    'id_out' (N,) uint64 - ID number of the 'out' BH
    'mass_in' (N,) float64 - mass of the 'in' BH (immediately preceding merger)
    'mass_out' (N,) float64 - mass of the 'out' BH (immediately preceding merger)
    'snapshot' (N,) int - Illustris output snapshot during which, or following, this merger event
    ```
