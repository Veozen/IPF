from generate_table import *
from IPF import *
import numpy as np


# test IPF
#step1 - create a table and generate the margins as well as the file that maps the cells of the inner table to the margins
raw_table = generate_random_table(4,8,scale=2)
input_table, margins, constraints = aggregate_table(raw_table, by=[0,1,2,3], var="value")   
margins = margins.rename(columns={"value":"target"}) #rename margin column

#step2 - modify the margins by adding noise to the inner cells
new_table = input_table.copy().drop("unit_id",axis=1)
new_table["value"] =  input_table["value"] + np.random.uniform(-1, 1, input_table.shape[0])
modified_table, modified_margins, constraints = aggregate_table(new_table, by=[0,1,2,3], var="value")   
modified_margins = modified_margins.rename(columns={"value":"target"})

# adjust the table in step1 to the margin obtained in step2
adjusted_table = IPF(input=input_table, constraints=constraints, targets=modified_margins, unit_id="unit_id", var="value", cons_id="cons_id", db_file=None, tol=1, maxIter=100)
