from __future__ import absolute_import, division, print_function, unicode_literals
import logging

import sys
import os
from time import strftime
import argparse
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from madminer.utils.interfaces.hdf5 import load_madminer_settings
from madminer.sampling import SampleAugmenter
from madminer import sampling
from madminer.ml import ParameterizedRatioEstimator,ScoreEstimator, Ensemble

# MadMiner output
logging.basicConfig(
  format='%(asctime)-5.5s %(funcName)-20.20s %(levelname)-7.7s %(message)s',
  datefmt='%H:%M',
  level=logging.DEBUG)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
  if "madminer" not in key:
    logging.getLogger(key).setLevel(logging.WARNING)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
#os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")

def augment_test(config):

  """  
  Extracts the joint likelihood ratio and the joint score for the test partition with the ALICES method
  """

  # access to the .h5 file with MadMiner settings
  madminer_settings=load_madminer_settings(f"{config['main_dir']}/{config['observable_set']}/{config['sample_name']}.h5", include_nuisance_benchmarks=False)

  if config['alices']['testing']['n_samples'] == -1:
    nsamples=madminer_settings[6]

  else: 
    nsamples = config['alices']['testing']['n_samples']

    logging.info(
        f'sample_name: {config["sample_name"]}; '
        f'observable set: {config["observable_set"]}; '
        f'nsamples: {nsamples}'
    )
  ########## Sample Augmentation ###########
  # # object to create the augmented training samples
  sampler=SampleAugmenter( f'{config["main_dir"]}/{config["observable_set"]}/{config["sample_name"]}.h5')

  # Creates a set of testing data - centered around the SM
  
  x, theta0, theta1, y, r_xz, t_xz, n_effective = sampler.sample_train_ratio(
  theta0=sampling.random_morphing_points(config["alices"]["testing"]["n_thetas"], [config["alices"]["testing"]["priors"]]),
  theta1=sampling.benchmark("sm"),
  n_samples=int(nsamples),
  folder=f'{config["main_dir"]}/{config["observable_set"]}/testing_samples/alices_{config["alices"]["testing"]["prior_name"]}',
  filename=f'test_ratio_{config["sample_name"]}',
  sample_only_from_closest_benchmark=True,
  return_individual_n_effective=True,
  partition = "test",
  n_processes = config["alices"]["testing"]["n_processes"]
  )

  logging.info(f'effective number of samples: {n_effective}')

def validation_plot(config,args):
    
    if args.model == 'alices':
        
        model_name = f"hidden_{config['alices']['training']['n_hidden']}_{config['alices']['training']['activation']}_alpha_{config['alices']['training']['alpha']}_epochs_{config['alices']['training']['n_epochs']}_bs_{config['alices']['training']['batch_size']}"

        model_path = f"{config['main_dir']}/{config['observable_set']}/models/{config['alices']['training']['training_samples_name']}/{config['alices']['training']['observables']}/{model_name}/alices_ensemble_{config['sample_name']}"

        alices = Ensemble()
        alices.load(model_path)
        
        joint_likelihood_ratio = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alices']['testing']['prior_name']}/r_xz_test_ratio_{config['sample_name']}.npy")
        joint_score = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alices']['testing']['prior_name']}/t_xz_test_ratio_{config['sample_name']}.npy")
        thetas = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alices']['testing']['prior_name']}/theta0_test_ratio_{config['sample_name']}.npy")
        x = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['alices']['testing']['prior_name']}/x_test_ratio_{config['sample_name']}.npy")

        joint_likelihood_ratio_log = np.log(joint_likelihood_ratio)
        log_r_hat, _ = alices.evaluate_log_likelihood_ratio(x=x, theta = thetas, test_all_combinations=False)
        

        fig=plt.figure()
        
        plt.scatter(-2*log_r_hat,-2*joint_likelihood_ratio_log)
        min_val = min(-2 * log_r_hat.min(), -2 * joint_likelihood_ratio_log.min())
        max_val = max(-2 * log_r_hat.max(), -2 * joint_likelihood_ratio_log.max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="k")
        

        plt.xlabel(r'True log likelihood ratio log r(x)')
        plt.ylabel(r'Estimated log likelihood ratio log $\hat{r}(x)$ (ALICES)')
        plt.tight_layout()
      
        return fig 
        # fig_score = plt.figure()
        # plt.scatter(-2*t_hat,-2*joint_score)
        # min_val = min(-2 * t_hat.min(), -2 * joint_score.min())
        # max_val = max(-2 * t_hat.max(), -2 * joint_score.max())
        # plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="k")
        # plt.xlabel(r'True score t(x)')
        # plt.ylabel(r'Estimated score $\hat{t}(x)$ (ALICES)')
        # plt.tight_layout()
        
        return fig #, fig_score
    
    # STILL TO  BE IMPLEMENTED
    # if args.model== 'alice':
    #     alice = ParameterizedRatioEstimator()
    #     alice.load(model_path)
        
    #     joint_likelihood_ratio = np.load(f"{args.main_dir}/testing_samples/alices/r_xz_test_ratio_{channel}_{args.sample_type}.npy")
    #     thetas = np.load(f"{args.main_dir}/testing_samples/alices/theta0_test_ratio_{channel}_{args.sample_type}.npy")
    #     x = np.load(f"{args.main_dir}/testing_samples/alices/x_test_ratio_{channel}_{args.sample_type}.npy")

    #     joint_likelihood_ratio_log = np.log(joint_likelihood_ratio)
    #     log_r_hat, _ = alice.evaluate_log_likelihood_ratio(x=x, theta = thetas, test_all_combinations=False)
        
    #     fig=plt.figure()
        
    #     plt.scatter(-2*log_r_hat,-2*joint_likelihood_ratio_log)
    #     min_val = min(-2 * log_r_hat.min(), -2 * joint_likelihood_ratio_log.min())
    #     max_val = max(-2 * log_r_hat.max(), -2 * joint_likelihood_ratio_log.max())
    #     plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="k")
        

    #     plt.xlabel(r'True log likelihood ratio log r(x)')
    #     plt.ylabel(r'Estimated log likelihood ratio log $\hat{r}(x)$ (ALICE)')
    #     plt.tight_layout()
    
        
    #     return fig
    
    # if args.model== 'sally':
        
    #     sally = ScoreEstimator()
    #     sally.load(model_path)
        
    #     joint_score = np.load(f"{args.main_dir}/testing_samples/alices/t_xz_test_ratio_{channel}_{args.sample_type}.npy")
    #     thetas = np.load(f"{args.main_dir}/testing_samples/alices/theta0_test_ratio_{channel}_{args.sample_type}.npy")
    #     x = np.load(f"{args.main_dir}/testing_samples/alices/x_test_ratio_{channel}_{args.sample_type}.npy")

    #     t_hat = sally.evaluate_score(x=x, theta = thetas)
        
        
    #     fig_score = plt.figure()
    #     plt.scatter(-2*t_hat,-2*joint_score)
    #     min_val = min(-2 * t_hat.min(), -2 * joint_score.min())
    #     max_val = max(-2 * t_hat.max(), -2 * joint_score.max())
    #     plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="k")
    #     plt.xlabel(r'True score t(x)')
    #     plt.ylabel(r'Estimated score $\hat{t}(x)$ (SALLY)')
    #     plt.tight_layout()
        
    #     return fig_score
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Computes distributions of different variables for signal and backgrounds.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--config_file', help='Path to the YAML configuration file', default='../config.yaml')

    parser.add_argument('--augment_test',help="creates testing samples with the ALICES method;",action='store_true',  default =False)
    
    parser.add_argument('-m','--model',help='which model to plot validation plot: SALLY, ALICE, ALICES' , choices=['sally','alice','alices'], default='alices')
    
    parser.add_argument('-v','--validation',help='wheter to plot the validation plot',action='store_true',default=False )

    args=parser.parse_args()
    
    # Read configuration parameters from the YAML file
    with open(args.config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)

                
    

    if args.augment_test:
        augment_test(config)

    if args.validation:
        validation_llr= validation_plot(config,args)#validation_llr, validation_score = validation_plot(config,args)

        model_name = f"hidden_{config['alices']['training']['n_hidden']}_{config['alices']['training']['activation']}_alpha_{config['alices']['training']['alpha']}_epochs_{config['alices']['training']['n_epochs']}_bs_{config['alices']['training']['batch_size']}"

        os.makedirs(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['alices']['testing']['prior_name']}/{config['alices']['testing']['observables']}/{model_name}/", exist_ok=True)

        validation_llr.savefig(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['alices']['testing']['prior_name']}/{config['alices']['testing']['observables']}/{model_name}/{args.model}_validation_llr_{config['sample_name']}.pdf")
        
        # validation_score.savefig(f"{config['plot_dir']}/{config['observable_set']}/validation/{config['alices']['testing']['prior_name']}/{config['alices']['testing']['observables']}/{model_name}/{args.model}_validation_score_{config['sample_name']}.pdf")

    # for channel in args.channel:

    #     if args.model == 'alices':
    #         {config["main_dir"]}/{config["observable_set"]}/{config["sample_name"]}.h5'
           
    #         validation_llr, validation_score = validation_plot(f"{args.main_dir}/models/{args.observables}/{args.model_architecture}/{args.model}_{channel}_{args.sample_type}", args, channel)
        
                    
    #         validation_llr.savefig(f'{args.plot_dir}/{args.model}_validation_llr_alices_gaussian_prior_0_0.2_{channel}_{args.sample_type}_{args.observables}_{args.model_architecture}.pdf')
                    
    #         validation_score.savefig(f'{args.plot_dir}/{args.model}_validation_score_alices_gaussian_prior_0_0.2_{channel}_{args.sample_type}_{args.observables}_{args.model_architecture}.pdf')

        # if args.model == 'alice':
           
        #     validation_llr = validation_plot(f"{args.main_dir}/models/{args.observables}/{args.model_architecture}/{args.model}_{channel}_{args.sample_type}", args, channel)
                    
        #     validation_llr.savefig(f'{args.plot_dir}/{args.model}_validation_llr_{channel}_{args.sample_type}_{args.observables}_{args.model_architecture}.pdf')
                    
        # if args.model == 'sally':
           
        #     validation_score = validation_plot(f"{args.main_dir}/models/{args.observables}/{args.model_architecture}/{args.model}_{channel}_{args.sample_type}", args, channel)
        
                     
        #     validation_score.savefig(f'{args.plot_dir}/{args.model}_validation_score_{channel}_{args.sample_type}_{args.observables}_{args.model_architecture}.pdf')
