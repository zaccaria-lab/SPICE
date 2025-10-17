import pandas as pd
import numpy as np

def find_sol_differences(debug, solutions):
    a = debug[debug.Solution.isin(solutions)].pivot_table(index='variables', values='values', columns='Solution')
    return a[a.nunique(axis=1) > 1] 