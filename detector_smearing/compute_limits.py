# -*- coding: utf-8 -*-

"""
compute_asymptotic_limits.py

Extracts limits based on the asymptotic properties of the likelihood ratio as test statistics.

This allows taking into account the effect of square terms in a consistent way (not possible with the Fisher information formalism)

Marta Silva (LIP/IST/CERN-ATLAS), 04/03/2024

"""
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
matplotlib.use('Agg') 
from madminer.sampling import combine_and_shuffle

#MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

# Choose the GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")


# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
  if "madminer" not in key:
    logging.getLogger(key).setLevel(logging.WARNING)

def plot_likelihood_ratio(parameter_grid,llr,xlabel,ylabel,do_log):

  fig = plt.figure()

  plt.plot(parameter_grid,llr,color='black',lw=0.5)

  if do_log:
    plt.yscale ('log')
  plt.axhline(y=1.64,linestyle='--',color='blue',label='95%CL')
  plt.axhline(y=1.0,linestyle='--',color='red',label='68%CL')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  parameter_points_to_plot=[parameter_grid[i] for i,llr_val in enumerate(llr) if llr_val<3.28]
  plt.xlim(-max(abs(parameter_points_to_plot[0]),abs(parameter_points_to_plot[-1])),+max(abs(parameter_points_to_plot[0]),abs(parameter_points_to_plot[-1])))
  plt.ylim(-1,3.28)
  plt.legend()
  plt.tight_layout()

  return fig

def extract_limits_single_parameter(parameter_grid,p_values,index_central_point):
  
  n_parameters=len(parameter_grid[0])
  list_points_inside_68_cl=[parameter_grid[i][0] if n_parameters==1 else parameter_grid[i] for i in range(len(parameter_grid)) if p_values[i] > 0.32 ]
  list_points_inside_95_cl=[parameter_grid[i][0] if n_parameters==1 else parameter_grid[i] for i in range(len(parameter_grid)) if p_values[i] > 0.05 ]  
  return parameter_grid[index_central_point],[round(list_points_inside_68_cl[0],5),round(list_points_inside_68_cl[-1],5)],[round(list_points_inside_95_cl[0],5),round(list_points_inside_95_cl[-1],5)]

# alternative to setting the points where to calculate the likelihoods for a single coefficient
def get_thetas_eval(theta_min,theta_max,spacing):
  theta_max=1.2
  theta_min=-1.2
  spacing=0.1
  thetas_eval=np.array([[round(i*spacing - (theta_max-theta_min)/2,4)] for i in range(int((theta_max-theta_min)/spacing)+1)])
  if np.array(0.0) not in thetas_eval:
    thetas_eval=np.append(thetas_eval,np.array(0.0))
  
  return thetas_eval

# get indices of likelihood histograms to plot for a single coefficient
def get_indices_llr_histograms(parameter_grid,npoints_plotting,plot_parameter_spacing=None,index_best_point=None):
  
  if plot_parameter_spacing is not None:
    npoints_plotting=int((parameter_grid[0,0]-parameter_grid[-1,0])/plot_parameter_spacing)
  
  if npoints_plotting>len(parameter_grid):
    logging.warning('asking for more points than what the parameter grid has, will plot all points')
    npoints_plotting=len(parameter_grid)
  
  # span the entire parameter grid
  if index_best_point is None:
    indices = list(range(0,len(parameter_grid),len(parameter_grid)/npoints_plotting))
  # span a small region around the best fit point
  else:
    indices = list(range(max(0,index_best_point-int(npoints_plotting/2)),min(len(parameter_grid),index_best_point+int(npoints_plotting/2))))
  
  # make sure that the SM point is always included
  sm_point_index=np.where(parameter_grid==[0.0])[0][0] # 1D case
  if sm_point_index not in indices:
    indices.append(sm_point_index)

  return list(indices)

def get_minmax_y_histograms(histos,epsilon=0.02):

  min_value=np.min([np.min(histo.histo) for histo in histos])*(1-epsilon)
  max_value=np.max([np.max(histo.histo) for histo in histos])*(1+epsilon)

  return (min_value,max_value)

if __name__ == "__main__":

  parser = ap.ArgumentParser(description='Computes limits using asymptotic (large sample size) limit of the likelihood ratio as a chi-square distribution.',formatter_class=ap.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('--config_file', help='Path to the YAML configuration file', default='config.yaml')

  args=parser.parse_args()
  
  # Read configuration parameters from the YAML file
  with open(args.config_file, 'r') as config_file:
      config = yaml.safe_load(config_file)

  os.makedirs(f"{config['plot_dir']}/{config['observable_set']}/limits/",exist_ok=True)
  
  
  if config['limits']['observable_y'] is None:
    hist_vars=[config['limits']['observable_x']]
    hist_bins=[config['limits']['binning_x']] if config['limits']['binning_x'] is not None else None
  else:
    hist_vars=[config['limits']['observable_x'],config['limits']['observable_y']]
    hist_bins=[config['limits']['binning_x'],config['limits']['binning_y']] if (config['limits']['binning_x'] is not None and config['limits']['binning_y'] is not None) else None
  
  logging.debug(f'hist variables: {str(hist_vars)}; hist bins: {str(hist_bins)}')

  #for sample_type in config['sample_name']:
  logging.info(f"sample type: {config['sample_name']}")
  list_central_values=[]

  if config['limits']['mode'] == 'ml':
    log_file_path=f"full_{config['limits']['method']}_{config['sample_name']}_range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}_lumi{config['limits']['lumi']}"  

  elif config['limits']['mode'] == 'sally':
    log_file_path=f"full_sally_{config['sample_name']}_range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}_lumi{config['limits']['lumi']}"  
    
  else:
    log_file_path=f"full_histograms_{config['sample_name']}_range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}_lumi{config['limits']['lumi']}"

  if config['limits']['shape_only']:
    log_file_path += '_shape_only'

  folder =   f"{config['plot_dir']}/{config['observable_set']}/limits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}"

  os.makedirs(folder,exist_ok=True)

  log_file=open(f"{folder}/{log_file_path}"+".csv",'a')

  if os.path.getsize(f"{folder}/{log_file_path}"+".csv") == 0:
    
    if config['limits']['mode'] == 'ml':
      if config['limits']['method']== 'alices':
        log_file.write(f'observables, ALICES model, binning_x, central value, 68% CL, 95% CL \n')

    elif config['limits']['mode'] == 'ml':
      if config['limits']['method'] == 'alice':
        log_file.write(f'observables, ALICE model, binning_x, central value, 68% CL, 95% CL \n')

    elif config['limits']['mode'] == 'sally':
      log_file.write('observables, SALLY model, binning_x, central value, 68% CL, 95% CL \n')

    else:
      log_file.write('observable_x, binning_x, observable_y, binning_y, central value, 68% CL, 95% CL \n')

  for i in range(config['limits']['n_fits']):
    
    # object/class to perform likelihood/limit extraction, shuffling to have different datasets
    if i==0:
      limits_file=AsymptoticLimits(f"{config['main_dir']}/{config['observable_set']}/{config['sample_name']}.h5")
    else:
      combine_and_shuffle([f"{config['main_dir']}/{config['sample_name']}.h5"],
        f"{config['main_dir']}/{config['observable_set']}/{config['sample_name']}_shuffled.h5",recalculate_header=False)
      limits_file=AsymptoticLimits(f"{config['main_dir']}/{config['observable_set']}/{config['sample_name']}_shuffled.h5")

    if config['limits']['method']=='sally':
      parameter_grid,p_values,index_best_point,log_r_kin,log_r_rate,(histos, observed, observed_weights)=limits_file.expected_limits(
      mode=config['limits']['mode'],theta_true=[0.0], include_xsec=not config['limits']['shape_only'],
      model_file=f"{config['main_dir']}/{config['observable_set']}/models/sally/{config['limits']['observables']}/{config['limits']['model']}/sally_ensemble_{config['sample_name']}" if 'sally' == config['limits']['mode'] else None,
      hist_vars=hist_vars if 'histo' == config['limits']['mode'] else None,
      hist_bins=hist_bins,
      luminosity=config['limits']['lumi']*1000.0,
      return_asimov=True,test_split=0.5,n_histo_toys=None,
      grid_ranges=[config['limits']['grid_ranges']],grid_resolutions=[config['limits']['grid_resolutions']])
      #thetas_eval=thetas_eval,grid_resolutions=None) # can be set given min, max and spacing using the get_thetas_eval funcion

    else:
      parameter_grid,p_values,index_best_point,llr_kin,llr_rate,(histos, observed, observed_weights)=limits_file.expected_limits(
      mode=config['limits']['mode'],theta_true=[0.0], include_xsec=not config['limits']['shape_only'],
      model_file=f"{config['main_dir']}/{config['observable_set']}/models/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/{config['limits']['method']}_ensemble_{config['sample_name']}" if 'ml' == config['limits']['mode'] else None,
      hist_vars=hist_vars if 'histo' == config['limits']['mode'] else None,
      hist_bins=hist_bins,
      luminosity=config['limits']['lumi']*1000.0,
      return_asimov=True,test_split=0.5,n_histo_toys=None,
      grid_ranges=[config['limits']['grid_ranges']],grid_resolutions=[config['limits']['grid_resolutions']])
      #thetas_eval=thetas_eval,grid_resolutions=None) # can be set given min, max and spacing using the get_thetas_eval funcion

    logging.debug(f'parameter grid: {parameter_grid}')
    # if i%5==0:

      # Confusa com esta parte do Ricardo
      # # plotting a subset of the likelihood histograms for debugging
      # if args.debug:
        
      #   indices=get_indices_llr_histograms(parameter_grid,npoints_plotting=4,index_best_point=index_best_point)

      #   likelihood_histograms = plot_histograms(
      #       histos=[histos[i] for i in indices],
      #       observed=[observed[i] for i in indices],
      #       observed_weights=observed_weights,
      #       histo_labels=[f"$cHWtil = {parameter_grid[i,0]:.2f}$" for i in indices],
      #       xlabel=config['observable_x'],
      #       xrange=(hist_bins[0][0],hist_bins[0][-1]) if hist_bins is not None else None,
      #       yrange=get_minmax_y_histograms([histos[i] for i in indices],epsilon=0.05) ,
      #       log=args.do_log,
      #   )
      #   if 'ml' in config['limits']['mode'] and 'alices' in config['limits']['method']:
      #     likelihood_histograms.savefig(f'{config['plot_dir']}/{config['observable_set']}/limits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/likelihoods_alices_{config['sample_name']}_{i}.pdf')
      #   if 'ml' in config['limits']['mode'] and 'alice' in config['limits']['method']:
      #     likelihood_histograms.savefig(f'{config['plot_dir']}/{config['observable_set']}/limits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/likelihoods_alice_{config['sample_name']}_{i}.pdf')
      #   else:
      #     if config['observable_y'] is None:
      #       likelihood_histograms.savefig(f'{config['plot_dir']}/{config['observable_set']}/limits/likelihoods_{config['sample_name']}_{config['observable_x']}_{len(config['binning_x'])-1}bins_lumi{config['limits']['lumi']}_{i}.pdf')
      #     else:
      #       likelihood_histograms.savefig(f'{config['plot_dir']}/{config['observable_set']}/limits/likelihoods_{config['sample_name']}_{config['observable_x']}_{len(config['binning_x'])-1}bins_{config['observable_y']}_{len(config['binning_y'])-1}bins_lumi{config['limits']['lumi']}_{i}.pdf')

    rescaled_log_r = llr_kin+llr_rate
    rescaled_log_r = -2.0*(rescaled_log_r[:] - rescaled_log_r[index_best_point])
    llr_histo= plot_likelihood_ratio(parameter_grid[:,0],rescaled_log_r,xlabel='cHWtil',ylabel='Rescaled -2*log_r',do_log=False)

    if 'ml' == config['limits']['mode']:
      llr_histo.savefig(f"{config['plot_dir']}/{config['observable_set']}/limits/{config['limits']['prior']}/{config['limits']['observables']}/{config['limits']['model']}/llr_curve_{config['limits']['method']}_{config['sample_name']}_lumi{config['limits']['lumi']}_range_{config['limits']['grid_ranges']}_resolutions_{config['limits']['grid_resolutions']}_{i}.pdf")

    #isto não mudei do Ricardo
    else:
      if config['observable_y'] is None:
        llr_histo.savefig(f"{config['plot_dir']}/{config['observable_set']}/limits/llr_curve_{config['sample_name']}_{config['observable_x']}_{len(config['binning_x'])-1}bins_lumi{config['limits']['lumi']}_{i}.pdf")
      else:
        llr_histo.savefig(f"{config['plot_dir']}/{config['observable_set']}/limits/llr_curve_{config['sample_name']}_{config['observable_x']}_{len(config['binning_x'])-1}bins_{config['observable_y']}_{len(config['binning_y'])-1}bins_lumi{config['limits']['lumi']}_{i}.pdf")
  
    central_value,cl_68,cl_95=extract_limits_single_parameter(parameter_grid,p_values,index_best_point)
    logging.debug(f'n_fit: {str(i)}, central value: {str(central_value)}; 68% CL: {str(cl_68)}; 95% CL: {str(cl_95)}')
    list_central_values.append(central_value[0]) 

    if 'ml' == config['limits']['mode']:
      log_file.write(f"{config['limits']['observables']}, {config['limits']['model']}, {str(config['limits']['binning_x']).replace(',',' ')}, {str(central_value[0])}, {str(cl_68).replace(',',' ')}, {str(cl_95).replace(',',' ')} \n") 
    elif 'sally' == config['limits']['mode']:
      log_file.write(f"{config['limits']['observables']}, {config['limits']['model']}, {str(config['limits']['binning_x']).replace(',',' ')}, {str(central_value[0])}, {str(cl_68).replace(',',' ')}, {str(cl_95).replace(',',' ')} \n") 
    else:
      log_file.write(f"{config['limits']['observable_x']}, {str(config['limits']['binning_x']).replace(',',' ')}, {config['limits']['observable_y']}, {str(config['limits']['binning_y']).replace(',',' ')}, {str(central_value[0]).replace(',',' ')}, {str(cl_68).replace(',',' ')}, {str(cl_95).replace(',',' ')} \n") 

  logging.debug("list of central values : "+str(list_central_values))

  log_file.close()

