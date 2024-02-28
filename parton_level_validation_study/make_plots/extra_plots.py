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

# Choose the GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")

path = '/lstore/titan/martafsilva/master_thesis/master_thesis_code/parton_level_validation_study/config.yaml'

with open(path, 'r') as config_file:
    config = yaml.safe_load(config_file)

                

model_name = f"sally_hidden_{config['sally']['training']['n_hidden']}_{config['sally']['training']['activation']}_epochs_{config['sally']['training']['n_epochs']}_bs_{config['sally']['training']['batch_size']}"

model_path = f"{config['main_dir']}/{config['observable_set']}/models/{config['sally']['training']['training_samples_name']}/{config['sally']['training']['observables']}/{model_name}/sally_ensemble_{config['sample_name']}"


sally = ScoreEstimator()
sally.load('/lstore/titan/martafsilva/master_thesis/master_thesis_output/parton_level_validation_study_output/all/models/gaussian_prior_0_0.4/kinematic_only/sally_hidden_[50]_relu_epochs_50_bs_128/sally_ensemble_wh_signalOnly_SMonly_noSysts_lhe/estimator_1')

joint_score = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['sally']['testing']['prior_name']}/t_xz_test_ratio_{config['sample_name']}.npy")
thetas = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['sally']['testing']['prior_name']}/theta0_test_ratio_{config['sample_name']}.npy")
x = np.load(f"{config['main_dir']}/{config['observable_set']}/testing_samples/alices_{config['sally']['testing']['prior_name']}/x_test_ratio_{config['sample_name']}.npy")

t_hat = sally.evaluate_score(x=x, theta = thetas)

plt.scatter(-2*t_hat,-2*joint_score) # Had to put a [0] because for the ensemble the output is a tuple (?)
min_val = min(-2 * t_hat.min(), -2 * joint_score.min())
max_val = max(-2 * t_hat.max(), -2 * joint_score.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="k")
plt.ylim(-10,10)
plt.xlabel(r'True score t(x)')
plt.ylabel(r'Estimated score $\hat{t}(x)$ (SALLY)')
plt.tight_layout()

plt.savefig("best_estimator_adjusted_axis.pdf")