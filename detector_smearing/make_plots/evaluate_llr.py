from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import matplotlib
import os, sys
import argparse as ap
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from madminer.plotting.distributions import *
from madminer.ml import ScoreEstimator, Ensemble, ParameterizedRatioEstimator
from madminer import sampling

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
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")

def plot_llr_individual(base_model_path, filename,fig_name):
    theta_each = np.linspace(-1.2, 1.2, 25)
    theta_grid = np.array([theta_each]).T

    sa = SampleAugmenter(filename, include_nuisance_parameters=True)
    start_event_test, end_event_test, correction_factor_test = sa._train_validation_test_split('test', validation_split=0.2, test_split=0.2)
    x, weights = sa.weighted_events(theta='sm', start_event=start_event_test, end_event=end_event_test)
    weights *= correction_factor_test  # Scale the events by the correction factor

    plt.figure(figsize=(10, 6))

    for i in range(5):
        estimator_number = i + 1 
        model_path = base_model_path + f"estimator_{i}"
        
        alices = ParameterizedRatioEstimator()
        alices.load(model_path)

        log_r_hat, _ = alices.evaluate_log_likelihood_ratio(x=x, theta=theta_grid)
        plt.plot(theta_each, -2*np.average(log_r_hat,axis=1, weights = weights), ls='-', label=f"Estimator {estimator_number}")

    plt.xlabel(r'$\theta$')
    plt.ylabel('Log Likelihood Ratio')
    plt.legend()
    plt.savefig(fig_name)
    plt.show()

def plot_llr_ensemble(base_model_path, filename,fig_name):


    alices = Ensemble()
    alices.load(base_model_path)
    theta_each = np.linspace(-1.2,1.2,25)
    theta_grid = np.array([theta_each]).T

    sa = SampleAugmenter(filename, include_nuisance_parameters=True)

    start_event_test, end_event_test, correction_factor_test = sa._train_validation_test_split('test',validation_split=0.2,test_split=0.2)

    x,weights=sa.weighted_events(theta='sm',start_event=start_event_test,end_event=end_event_test)

    weights*=correction_factor_test # scale the events by the correction factor


    log_r_hat, _=alices.evaluate_log_likelihood_ratio(x=x, theta = theta_grid)


    plt.plot(theta_each,-2*np.average(log_r_hat,axis=1, weights = weights), ls='-', label = r"$\hat{r}(x|\theta)$ (ALICE)")
    plt.xlabel(r'$\theta$')
    plt.ylabel('Log Likelihood Ratio')
    plt.legend()

    plt.savefig(fig_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plots log likelihood ratio evaluate for all estimators or with and ensemble', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', help='Plot llr for all estimators or the ensemble', choices=['individual', 'ensemble'])

    args = parser.parse_args()


    base_model_path = "/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_smearing_output/met/models/gaussian_prior_0_0.4/kinematic_only/alices_hidden_[50]_relu_alpha_5_epochs_50_bs_128/alices_ensemble_wh_signalOnly_noSysts_lhe/"
    filename = "/lstore/titan/martafsilva/master_thesis/master_thesis_output/detector_smearing_output/met/wh_signalOnly_noSysts_lhe.h5"
    fig_name_individual = "llr_all_estimator_gaussian_prior_0_0.4_kinematic_only_alices_hidden_[50]_relu_alpha_5_epochs_50_bs_128_ud_wph_mu_smeftsim_SM_lhe.pdf"
    fig_name_ensemble = "llr_ensemble_gaussian_prior_0_0.4_kinematic_only_alices_hidden_[50]_relu_alpha_5_epochs_50_bs_128_ud_wph_mu_smeftsim_SM_lhe.pdf"

    if args.mode == 'individual':
        plot_llr_individual(base_model_path, filename,fig_name_individual)

    if args.mode == 'ensemble':
        plot_llr_ensemble(base_model_path, filename,fig_name_ensemble)