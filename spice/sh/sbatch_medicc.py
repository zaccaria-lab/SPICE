import pandas as pd
import numpy as np
import os

def create_medicc_script(patient, 
                        input_file, 
                        output_dir, 
                        max_cn=6, 
                        j=1, 
                        mem=16, 
                        nhours=8):
                    
    out = os.path.join(output_dir, 'medicc', 'eval.out')
    err = os.path.join(output_dir, 'medicc', 'eval.err')
    
    if not os.path.exists(os.path.join(output_dir, 'medicc')):
        os.makedirs(os.path.join(output_dir, 'medicc'))

    run = '''#!/usr/bin/env bash
#SBATCH --job-name=MED_{}
#SBATCH --ntasks=1
#SBATCH --time={}:00:00
#SBATCH --mem={}G
#SBATCH --cpus-per-task={}
#SBATCH -o {}
#SBATCH -e {}

source ~/.bashrc
conda activate medicc2_1.0.4

medicc2 {} {} -j {} --maxcn {} -v --prefix {} --events
    '''.format(patient, nhours, mem, j, out, err, input_file, os.path.join(output_dir, 'medicc'), j, max_cn, f'{patient}_medicc')
    path=os.path.join(output_dir, 'run_medicc.sh')
    with open(path, 'w') as out:
        out.write(run)