# -*- coding: utf-8 -*-

"""
analysis_sample_combiner.py

Combines analyzed samples from the output of the analysis scripts.

Marta Silva (LIP/IST/CERN-ATLAS), 09/02/2024
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os, shutil, sys
import argparse 
import yaml
from madminer.core import MadMiner
from madminer.lhe import LHEReader
from madminer.sampling import combine_and_shuffle

# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

def combine_individual(full_proc_dir,charge,flavor,args):

    if args.do_signal:

        # generated SM samples
        os.symlink(f'{full_proc_dir}/signal/w{charge}h_{flavor}_smeftsim_SM_lhe.h5'.format(),f'{full_proc_dir}/w{charge}h_{flavor}_signalOnly_SMonly_noSysts_lhe.h5'.format())

        # generated SM samples + generated BSM samples
        if args.combine_BSM:
            combine_and_shuffle([
            f'{full_proc_dir}/signal/w{charge}h_{flavor}_smeftsim_SM_lhe.h5',
            f'{full_proc_dir}/signal/w{charge}h_{flavor}_smeftsim_BSM_lhe.h5'],
            f'{full_proc_dir}/w{charge}h_{flavor}_signalOnly_noSysts_lhe.h5'
            )


    if args.do_backgrounds:

      combine_and_shuffle([
        f'{full_proc_dir}/background/t{charge}b_{flavor}_background_lhe.h5',
        f'{full_proc_dir}/background/tt_{flavor}{charge}jj_background_lhe.h5',
        f'{full_proc_dir}/background/w{charge}bb_{flavor}_background_lhe.h5'],
        f'{full_proc_dir}/w{charge}h_{flavor}_backgroundOnly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        
        combine_and_shuffle([
            f'{full_proc_dir}/w{charge}h_{flavor}_signalOnly_noSysts_lhe.h5',
            f'{full_proc_dir}/w{charge}h_{flavor}_backgroundOnly_noSysts_lhe.h5'],
            f'{full_proc_dir}/w{charge}h_{flavor}_withBackgrounds_noSysts_lhe.h5'
        )

    if args.do_signal and args.do_backgrounds:

      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/signal/w{charge}h_{flavor}_smeftsim_SM_lhe.h5',
        f'{full_proc_dir}/w{charge}h_{flavor}_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/w{charge}h_{flavor}_withBackgrounds_SMonly_noSysts_lhe.h5'
      )   

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        
        combine_and_shuffle([
          f'{full_proc_dir}/w{charge}h_{flavor}_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/w{charge}h_{flavor}_backgroundOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/w{charge}h_{flavor}_withBackgrounds_noSysts_lhe.h5'
        )

    logging.info('finished standard sample combination for training of ML methods, will now start the combinations used for plotting')
  
def combine_charges(full_proc_dir, flavor, args):

    if args.do_signal:

    # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wph_{flavor}_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_{flavor}_signalOnly_SMonly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_{flavor}_signalOnly_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wph_{flavor}_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wmh_{flavor}_signalOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wh_{flavor}_signalOnly_noSysts_lhe.h5'
        )
    
    if args.do_backgrounds:
        
        combine_and_shuffle([
        f'{full_proc_dir}/wph_{flavor}_backgroundOnly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_{flavor}_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_{flavor}_backgroundOnly_noSysts_lhe.h5'
      )

    if args.do_signal and args.do_backgrounds:
      
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wh_{flavor}_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wh_{flavor}_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_{flavor}_withBackgrounds_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wh_{flavor}_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wh_{flavor}_backgroundOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wh_{flavor}_withBackgrounds_noSysts_lhe.h5'
        )

def combine_flavors(full_proc_dir, charge, args):
    if args.do_signal:
        # generated SM samples
        combine_and_shuffle([
            f'{full_proc_dir}/w{charge}h_e_signalOnly_SMonly_noSysts_lhe.h5',
            f'{full_proc_dir}/w{charge}h_mu_signalOnly_SMonly_noSysts_lhe.h5'],
            f'{full_proc_dir}/w{charge}h_signalOnly_SMonly_noSysts_lhe.h5'
        )

        # generated SM samples + generated BSM samples
        if args.combine_BSM:
            combine_and_shuffle([
            f'{full_proc_dir}/w{charge}h_e_signalOnly_noSysts_lhe.h5',
            f'{full_proc_dir}/w{charge}h_mu_signalOnly_noSysts_lhe.h5'],
            f'{full_proc_dir}/w{charge}h_signalOnly_noSysts_lhe.h5'
            )

    if args.do_backgrounds:

        combine_and_shuffle([
            f'{full_proc_dir}/w{charge}h_e_backgroundOnly_noSysts_lhe.h5',
            f'{full_proc_dir}/w{charge}h_mu_backgroundOnly_noSysts_lhe.h5'],
            f'{full_proc_dir}/w{charge}h_backgroundOnly_noSysts_lhe.h5'
        )

    if args.do_signal and args.do_backgrounds:
        
        # generated SM samples
        combine_and_shuffle([
            f'{full_proc_dir}/w{charge}h_signalOnly_SMonly_noSysts_lhe.h5',
            f'{full_proc_dir}/w{charge}h_backgroundOnly_noSysts_lhe.h5'],
            f'{full_proc_dir}/w{charge}h_withBackgrounds_SMonly_noSysts_lhe.h5'
        )

        # generated SM samples + generated BSM samples
        if args.combine_BSM:
            combine_and_shuffle([
            f'{full_proc_dir}/w{charge}h_signalOnly_noSysts_lhe.h5',
            f'{full_proc_dir}/w{charge}h_backgroundOnly_noSysts_lhe.h5'],
            f'{full_proc_dir}/w{charge}h_withBackgrounds_noSysts_lhe.h5'
            )

def combine_all(full_proc_dir,args):
    if args.do_signal:
      
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wph_e_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_e_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wph_mu_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_mu_signalOnly_SMonly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_signalOnly_SMonly_noSysts_lhe.h5'
      )

      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wph_e_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wmh_e_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wph_mu_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wmh_mu_signalOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wh_signalOnly_noSysts_lhe.h5'
        )
    
    if args.do_backgrounds:
      combine_and_shuffle([
        f'{full_proc_dir}/wph_e_backgroundOnly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_e_backgroundOnly_noSysts_lhe.h5',
        f'{full_proc_dir}/wph_mu_backgroundOnly_noSysts_lhe.h5',
        f'{full_proc_dir}/wmh_mu_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_backgroundOnly_noSysts_lhe.h5'
      )

    if args.do_signal and args.do_backgrounds:
      # generated SM samples
      combine_and_shuffle([
        f'{full_proc_dir}/wh_signalOnly_SMonly_noSysts_lhe.h5',
        f'{full_proc_dir}/wh_backgroundOnly_noSysts_lhe.h5'],
        f'{full_proc_dir}/wh_withBackgrounds_SMonly_noSysts_lhe.h5'
      )
      # generated SM samples + generated BSM samples
      if args.combine_BSM:
        combine_and_shuffle([
          f'{full_proc_dir}/wh_signalOnly_noSysts_lhe.h5',
          f'{full_proc_dir}/wh_backgroundOnly_noSysts_lhe.h5'],
          f'{full_proc_dir}/wh_withBackgrounds_noSysts_lhe.h5'
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combines and shuffles different samples, depending on the purposes.',
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config.yaml')

    parser.add_argument('--do_signal', help='analyze/combine signal samples', action='store_true', default=True)

    parser.add_argument('--do_backgrounds', help='combine background samples', action='store_true', default=True)

    parser.add_argument('--combine_individual', help='combine samples for each of the charge+flavor combination separately (should be done once before all other combination possibilities)',
                        action='store_true', default=True)

    parser.add_argument('--combine_flavors', help='combine muon and electron events for each charges separately', action='store_true', default=True)

    parser.add_argument('--combine_charges', help='combine w+ and w- events for each flavor separately', action='store_true', default=True)

    parser.add_argument('--combine_all', help='combine all charges + flavors', action='store_true', default=True)

    parser.add_argument('--combine_BSM', help='combine samples generated at the BSM benchmarks with those generated at the SM point', action='store_true', default=False)

    args = parser.parse_args()

    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

        main_dir = config['main_dir']
        observable_set = config['observable_set']


    if not (args.do_signal or args.do_backgrounds):
        sys.exit('asking to process events but not asking for either signal or background ! Exiting...')

    if not args.combine_individual and (args.combine_flavors or args.combine_charges or args.combine_all):
        logging.warning('not asking to combine samples for each of the charge+flavor combination separately, but asking to make a combination of samples from different charges or flavors. watch out for possible crash from missing samples')

    if args.combine_flavors and args.combine_charges:
        if not args.combine_all:
            logging.warning('asked to combine samples in flavor and in charge separately but combine_all=False, won\'t create single WH sample')

    full_proc_dir = f'{main_dir}/{observable_set}/'

    flavors = ['mu', 'e']
    charges = ['p', 'm']

    if args.combine_individual:
        for flavor in flavors:
            for charge in charges:
                combine_individual(full_proc_dir,charge,flavor,args)

    # combining positive and negative charge channels for each of the flavors separately
    if args.combine_charges:
        for flavor in flavors:
            combine_charges(full_proc_dir,flavor,args)

    # combining electron and muon channels for each of the charges separately
    if args.combine_flavors:
        for charge in charges:
            combine_flavors(full_proc_dir,charge,args)

    # combining positive and negative charge channels and all the flavors
    if args.combine_all:
        combine_all(full_proc_dir,args)

