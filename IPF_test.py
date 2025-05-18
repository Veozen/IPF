from generate_table import *
from IPF import *
import numpy as np


# test IPF
table = generate_random_table(4,8,scale=2)

table, margins, constraints = aggregate_table(table, by=[0,1,2,3], var="value")   

#rename margin column
margins = margins.rename(columns={"value":"target"})
margins["target"] +=  np.random.uniform(-2, 2, margins.shape[0])
 

adjusted_table = IPF(input=table, constraints=constraints, targets=margins, unit_id="unit_id", var="value", cons_id="cons_id", db_file=None, tol=1, maxIter=100)
