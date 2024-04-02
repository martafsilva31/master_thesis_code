from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os,sys
import argparse as ap
from madminer.limits import AsymptoticLimits
from madminer import sampling
from madminer.plotting import plot_histograms
from madminer.utils.histo import Histo
import numpy as np
import matplotlib
import yaml
import matplotlib.pyplot as plt
import argparse 
matplotlib.use('Agg') 
from madminer.sampling import combine_and_shuffle

#from operator import 
# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")

def plot_llr_individual(base_model_path, filename,fig_name,args):
      
    limits_file=AsymptoticLimits(filename)

    plt.figure(figsize=(10, 6))

    all_parameter_points_to_plot = [] 

    for i in range(5):
    
      estimator_number = i + 1
      model_path = base_model_path + f"estimator_{i}"

      parameter_grid,p_values,index_best_point,llr_kin,llr_rate,(histos, observed, observed_weights)=limits_file.expected_limits(
            mode="ml",theta_true=[0.0], include_xsec = True,
            model_file=model_path,
            luminosity=300*1000.0,
            return_asimov=True,test_split=0.2,n_histo_toys=None,
            grid_ranges=[[-1.2,1.2]],grid_resolutions=[300])
            #thetas_eval=theta_grid,grid_resolutions=None) # can be set given min, max and spacing using the get_thetas_eval funcion

      rescaled_log_r = llr_kin+llr_rate
      rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])    

      plt.plot(parameter_grid,rescaled_log_r,lw=0.5, label=f"Estimator {estimator_number}")    

      if args.CL: 
            # Store parameter points for this estimator
            parameter_points_to_plot = [parameter_grid[i] for i, llr_val in enumerate(rescaled_log_r) if llr_val < 3.28]
            all_parameter_points_to_plot.extend(parameter_points_to_plot)

    if args.CL:
      # Set x-axis and y-axis limits using all parameter points
      abs_parameter_points = [abs(point) for point in all_parameter_points_to_plot]
      plt.axhline(y=1.64,linestyle='--',color='blue',label='95%CL')
      plt.axhline(y=1.0,linestyle='--',color='red',label='68%CL')
      plt.xlim(-max(all_parameter_points_to_plot), max(all_parameter_points_to_plot))
      plt.ylim(-1, 3.28)

    plt.xlabel("cHWtil")
    plt.ylabel('Rescaled -2*log_r')
    plt.legend()
    plt.savefig(fig_name)
    plt.show()

def plot_llr_ensemble(base_model_path, filename,fig_name,args):


    limits_file=AsymptoticLimits(filename)

    plt.figure(figsize=(10, 6))

    parameter_grid,p_values,index_best_point,llr_kin,llr_rate,(histos, observed, observed_weights)=limits_file.expected_limits(
        mode="ml",theta_true=[0.0], include_xsec = True,
        model_file = base_model_path,
        luminosity=300*1000.0,
        return_asimov=True,test_split=0.2,n_histo_toys=None,
        grid_ranges=[[-1.2,1.2]],grid_resolutions=[300])
        #thetas_eval=theta_grid,grid_resolutions=None) # can be set given min, max and spacing using the get_thetas_eval funcion

    rescaled_log_r = llr_kin+llr_rate
    rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])    

    plt.plot(parameter_grid,rescaled_log_r,lw=0.5, color = 'k')    

 

    if args.CL:
      # Set x-axis and y-axis limits using all parameter points
      parameter_points_to_plot=[parameter_grid[i] for i,llr_val in enumerate(rescaled_log_r) if llr_val<3.28]
      plt.axhline(y=1.64,linestyle='--',color='blue',label='95%CL')
      plt.axhline(y=1.0,linestyle='--',color='red',label='68%CL')
      plt.xlim(-max(abs(parameter_points_to_plot[0]),abs(parameter_points_to_plot[-1])),+max(abs(parameter_points_to_plot[0]),abs(parameter_points_to_plot[-1])))
      plt.ylim(-1, 3.28)

    plt.xlabel("cHWtil")
    plt.ylabel('Rescaled -2*log_r')
    plt.legend()
    plt.savefig(fig_name)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plots log likelihood ratio evaluate for all estimators or with and ensemble', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', help='Plot llr for all estimators or the ensemble', choices=['individual', 'ensemble'])

    parser.add_argument('--CL', help='Wether to plot the CLs lines', action='store_true')

    args = parser.parse_args()


    base_model_path = "/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_smearing_output/met/models/gaussian_prior_0_0.4/kinematic_only/alices_hidden_[50]_relu_alpha_5_epochs_50_bs_128_thetas_with_scaling/alices_ensemble_wh_signalOnly_SMonly_noSysts_lhe/"
    filename = "/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_smearing_output/met/wh_signalOnly_SMonly_noSysts_lhe.h5"
    fig_name_individual = "llr_all_estimator_gaussian_prior_0_0.4_kinematic_only_alices_hidden_[50]_relu_alpha_5_epochs_50_bs_128_thetas_with_scaling_CL.pdf"
    fig_name_ensemble = "llr_ensemble_gaussian_prior_0_0.4_kinematic_only_alices_hidden_[50]_relu_alpha_5_epochs_50_bs_128_ud_wph_mu_smeftsim_SM_lhe.pdf"

    if args.mode == 'individual':
        plot_llr_individual(base_model_path, filename,fig_name_individual,args)

    if args.mode == 'ensemble':
        plot_llr_ensemble(base_model_path, filename,fig_name_ensemble,args)