# -*- coding: utf-8 -*-

"""
gen_signal.py

Generates WH signal events WH(->l v b b~), divided by W decay channel and charge (250k events each)

Can use different morphing setups (default: CP-odd operator only).

- sample contains weights for different benchmarks (from MG reweighting)

Can also generate events at the BSM benchmarks to populate regions of phase space not well populated by the SM sample
- smaller number than for SM point, 50k for each charge+flavour combination
- reweighted to other benchmarks (inc. SM point)

Marta Silva (LIP/IST/CERN-ATLAS), 08/02/2024
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
import argparse 
import yaml
from madminer.core import MadMiner
from madminer.lhe import LHEReader

# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG #INFO 
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

def gen_signal(main_dir, setup_file, do_pythia, pythia_card, auto_widths, prepare_scripts, generate_BSM, mg_dir, launch_SLURM_jobs):
    """
    Generates WH signal events WH(->l v b b~), divided by W decay channel and charge.
    """

    # Load morphing setup file
    miner = MadMiner()
    miner.load(f'{main_dir}/{setup_file}.h5')
    lhe = LHEReader(f'{main_dir}/{setup_file}.h5')

    # List of BSM benchmarks - SM + 1 BSM benchmarks (from Madminer)
    list_BSM_benchmarks = [x for x in lhe.benchmark_names_phys if x != 'sm']

    # auto width calculation
    # NB: Madgraph+SMEFTsim include terms up to quadratic order in the automatic width calculation, even when the ME^2 is truncated at the SM+interference term
    if auto_widths:
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme_autoWidths.dat'
    else:
        param_card_template_file='cards/param_card_template_SMEFTsim3_MwScheme.dat'

    # LIP specifics
    init_command='module load gcc63/madgraph/3.3.1',
    
   
    channels = ['wph_mu', 'wph_e', 'wmh_mu', 'wmh_e']

    # SM samples with MG (re)weights of BSM benchmarks
    for channel in channels:
        miner.run(
            mg_directory=mg_dir,
            log_directory=f'{main_dir}/logs/{channel}_smeftsim_SM',
            mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_SM',
            proc_card_file=f'cards/signal_processes/proc_card_{channel}_smeftsim.dat',
            param_card_template_file=param_card_template_file,
            pythia8_card_file=pythia_card if do_pythia else None,
            sample_benchmark='sm',
            run_card_file='cards/run_card_250k_WHMadminerCuts.dat',
            initial_command=init_command if init_command != '' else None,
            only_prepare_script=prepare_scripts
        )

    # BSM samples with MG (re)weights of other benchmarks (inc. SM)
    if generate_BSM:
        for channel in channels:
            miner.run_multiple(
                mg_directory=mg_dir,
                log_directory=f'{main_dir}/logs/{channel}_smeftsim_BSM',
                mg_process_directory=f'{main_dir}/signal_samples/{channel}_smeftsim_BSM',
                proc_card_file=f'cards/signal_processes/proc_card_{channel}_smeftsim.dat',
                param_card_template_file=param_card_template_file,
                pythia8_card_file=pythia_card if do_pythia else None,
                sample_benchmarks=list_BSM_benchmarks,
                run_card_files=['cards/run_card_50k_WHMadminerCuts.dat'],
                initial_command=init_command if init_command != '' else None,
                only_prepare_script=prepare_scripts
            )

    # launch gen jobs to SLURM # LIP specifics
    if args.launch_SLURM_jobs and args.prepare_scripts:
        logging.info("Launching SLURM generation jobs")
        cmd=f'find {main_dir}/signal_samples/*/madminer -name "run.sh" -exec sbatch -p lipq --mem=4G {{}} \;'
        os.popen(cmd)
        print(cmd)


    os.remove('/tmp/generate.mg5')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generates WH signal events WH(->l v b b~), divided by W decay channel and charge.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config.yaml')

    parser.add_argument('--do_pythia',help='whether or not to run Pythia after Madgraph',action='store_true',default=False)

    parser.add_argument('--auto_widths',help='Use parameter card with automatic width calculation',action='store_true',default=False)

    parser.add_argument('--prepare_scripts',help='Prepares only run scripts to e.g. submit to a batch system separately',action='store_true',default=False)

    parser.add_argument('--generate_BSM',help='Generate additional events at the BSM benchmarks',action='store_true',default=False)

    parser.add_argument('--launch_SLURM_jobs',help='If SLURM jobs are to be launched immediately after preparation of scripts',action="store_true",default=False)

    args=parser.parse_args()

    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

        main_dir = config['main_dir']
        setup_file = config['setup_file']
        pythia_card = config['pythia_card']
        mg_dir = config['mg_dir']

    # Generate signal
    gen_signal(main_dir, setup_file, args.do_pythia, pythia_card, args.auto_widths, args.prepare_scripts, args.generate_BSM, mg_dir, args.launch_SLURM_jobs)

