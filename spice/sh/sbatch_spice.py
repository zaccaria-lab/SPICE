import pandas as pd
import numpy as np
import os

standard_params = {'min_segsize' : 0, 'max_n_segs': None, 'allele_specific' : True, 'allele_phasing' : False, 'leaf_dependent_phasing' : False,
                    'max_cn' : 8, 'verbose' : True, 'weighted_wgd' : True, 'multiple_solutions' : True, 'solution_pool_size' : 1000, 
                    'n_bootstraps' : 1000, 'rm_small_segs' : False, 'rank_multiple_trees' : True}

def create_spice_script(patient, 
                        input_file, 
                        output_dir, 
                        suffix='',
                        params=standard_params, 
                        j=1, 
                        mem=16, 
                        nhours=8):
    
    save_dir = os.path.join(output_dir, 'spice' + suffix)
                    
    out = os.path.join(save_dir, 'eval.out')
    err = os.path.join(save_dir, 'eval.err')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    parameter_input = (lambda flag, value: '--{} {}'.format(flag, value if (type(value) != bool) | (value != True) else '') if (type(value) != bool) | (value != False) else '')

    param_string = ' '.join([parameter_input(k, v) for k, v in params.items()])

    run = '''#!/usr/bin/env bash
#SBATCH --job-name=CN_{}
#SBATCH --ntasks=1
#SBATCH --time={}:00:00
#SBATCH --mem={}G
#SBATCH --partition=ncpu
#SBATCH --cpus-per-task={}
#SBATCH -o {}
#SBATCH -e {}

source ~/.bashrc
conda activate spice_env

python /camp/home/bunkuma/working/bunkuma/repos/abi-spice/src/spice/sh/run_spice.py \\
--input_file {} --output_dir {} {} --time_limit {} --threads {}
    '''.format(patient, nhours+12, mem, j, out, err, input_file, save_dir, param_string, nhours, j)
    path=os.path.join(output_dir, 'run_spice{}.sh'.format(suffix))
    with open(path, 'w') as out:
        out.write(run)

post_process_args = ['verbose'] #'min_segsize', 'allele_specific', 'max_cn', 'rank_multiple_trees', 
post_process_params = {k : v for k, v in standard_params.items() if k in post_process_args}
        

def create_postprocess_script(patient, 
                            output_dir, 
                            suffix='',
                            top_prop=None,
                            weight_func='linear_max',
                            j=10, 
                            mem=32, 
                            nhours=8):
    
    filename_suffix = '' if weight_func == 'linear_max' else '_' + weight_func.replace('_', '') if str(weight_func) != 'False' else '_unweighted'
    top_prop_suffix = '' if top_prop is None else '_{}'.format(top_prop)
    out = os.path.join(output_dir, 'spice' + suffix, f'postprocess{filename_suffix}{top_prop_suffix}.out')
    err = os.path.join(output_dir, 'spice' + suffix, f'postprocess{filename_suffix}{top_prop_suffix}.err')
    run_dir = os.path.join(output_dir, 'spice' + suffix)
    
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    parameter_input = (lambda flag, value: '--{} {}'.format(flag, value if value != True else '') if value != False else '')
    suffix_string = '' if suffix == '' else '--suffix {}'.format(suffix)
    # param_string = ' '.join([parameter_input(k, v) for k, v in params.items()])

    run = '''#!/usr/bin/env bash
#SBATCH --job-name=CN{}
#SBATCH --ntasks=1
#SBATCH --time={}:00:00
#SBATCH --mem={}G
#SBATCH --cpus-per-task={}
#SBATCH -o {}
#SBATCH -e {}

source ~/.bashrc
conda activate spice_env

python /camp/home/bunkuma/working/bunkuma/repos/abi-spice/src/spice/sh/run_postprocess.py \\
--output_dir {} --verbose --threads {} {} --weight_func {} --top_prop {}
    '''.format(patient[-3:], nhours, mem, j, out, err, output_dir, j, suffix_string, weight_func, top_prop)
    path=os.path.join(output_dir, f'run_spice_postprocess{filename_suffix}{top_prop_suffix}.sh')
    print(path)
    with open(path, 'w') as out:
        out.write(run)