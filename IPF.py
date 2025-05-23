#Iterative proportional fitting
import duckdb
import pandas as pd
import itertools
from itertools import combinations
import functools
from time import perf_counter
import numpy as np

def generate_random_table(n_dim,n_cat,scale=1):
  #generate n_dim columns each with n_cat values
  sets = [set(range(n_cat)) for _ in range(n_dim)]
  cartesian_product = list(itertools.product(*sets))
  df = pd.DataFrame(cartesian_product, columns=[*range(n_dim)])
  #generate random values between 0 and scale
  df["value"] = np.random.rand(len(df)) * scale
  return df


def get_unique_col_name(df, base_name):
  # Generate a unique column name
  i = 1
  new_name = base_name
  while new_name in df.columns:
      new_name = f"{base_name}_{i}"
      i += 1   
  return new_name


def agg_by_sql(df: pd.DataFrame, by, var, id):
    if by is None or not by:
        # Aggregate over the entire dataset
        query = f"""
        SELECT 
            SUM({var}) AS {var},
            LIST({id}) AS {id}
        FROM 'df'
        """
    else:
        # Aggregate with grouping
        group_by_columns = ", ".join(map(lambda x: '"'+str(x)+'"' if isinstance(x, int) else str(x) , by))
        query = f"""
        SELECT 
            {group_by_columns},
            SUM({var}) AS {var},
            LIST({id}) AS {id}
        FROM 'df'
        GROUP BY {group_by_columns}
        """
    # Execute the query
    with duckdb.connect() as con:
      df_agg = con.execute(query).fetchdf()
    return df_agg


def aggregate_and_list(df:pd.DataFrame, by, var=None, margins=None, id=None):
    if by is not None and not isinstance(by,list):
        by = [by]
        
    subsets=[]
    if by is not None:
        for i in range(0,len(by)):
            comb = combinations(by,i)
            subsets = subsets + [list(c) for c in comb]
    else:
        subsets=[[]]
        
    if margins is not None:
        subsets = [sub for sub in subsets if sub in margins]
        
    df_out = pd.DataFrame()
    for sub in subsets:
        sub_agg = agg_by_sql(df, by=sub, var=var, id=id)
        df_out = pd.concat([df_out,sub_agg],ignore_index=True)
    return df_out  


def aggregate_table(df_in, by, var, margins=None):
  """
  aggreagate the input table into a table of the form
  
  output:
      Table
      unit_id     : identifier for the decision variables
      weight      : decision variables. >=0
      lb			    : weight >= lb
      ub			    : weight <= up
      
      Table
      unit_id     : identifier for the decision variables
      cons_id     : identifiant des contraintes
      
  """
  # aggregate "var" by "by" columns in case there are duplicates in the input to make sure we have a table with signle entries per cell
  by_values               = df_in.groupby(by).sum(var).reset_index()
  
  # get a unique name not already present in the dataframe to store cell identifier
  cell_id_name            = get_unique_col_name(by_values,"unit_id")
  
  # create a unique identifer for each cell of the table
  by_values[cell_id_name] = range(len(by_values))
  cell_id_lst             = list(by_values[cell_id_name])
  n_cells                 = len(cell_id_lst)
  
  # get margins of the input table
  df_margins                = aggregate_and_list(by_values, by, var, margins, cell_id_name)
  cons_id_name              = get_unique_col_name(df_margins,"cons_id")
  df_margins[cons_id_name]  = range(len(df_margins))
  n_margins                 = len(df_margins)
  
  # create a mapping of each margin identifer to a list of each cell identifer adding up to it
  constraints = df_margins.explode(cell_id_name).reset_index(drop=True)
  
  return by_values, df_margins.drop([cell_id_name],axis=1), constraints[[cell_id_name,cons_id_name]]


def get_discrepancy(con):
  """
    returns the discrepancies between then aggregated margins and their target values
    
    input: from the database connection con
      table wrk_weights
      table wrk_input_constraints
      table wrk_input_targets
      
    output: in the database connection con
      table wrk_discrepancies
    output:
      value maxDiscrepancy
      
  """
  con.execute(f"""
    CREATE OR REPLACE TABLE wrk_constraints AS
    SELECT a.cons_id,  sum(b.weight)  as aggregated_weight_per_constraint
    FROM wrk_input_constraints AS a 
    LEFT JOIN wrk_weights AS b
    ON a.unit_id=b.unit_id
    GROUP by a.cons_id
    ;
  """)
  con.execute(f"""
    CREATE OR REPLACE TABLE wrk_discrepancies AS
    SELECT a.cons_id, a.cons_type, a.target, b.aggregated_weight_per_constraint as target_approximation
    FROM wrk_input_targets AS a 
    LEFT JOIN wrk_constraints AS b
    ON a.cons_id = b.cons_id
    ;
  """)
  con.execute("""
    CREATE OR REPLACE TABLE wrk_discrepancies AS
    SELECT *,
           -- Step 1: Compute diff and adjustement
           target - target_approximation AS diff,
           target / target_approximation AS adjustement
    FROM wrk_discrepancies;

    -- Step 2: Apply constraints on adjustement and diff
    UPDATE wrk_discrepancies
    SET adjustement = CASE 
                          WHEN cons_type = 'le' AND adjustement > 1 THEN 1
                          WHEN cons_type = 'ge' AND adjustement < 1 THEN 1
                          ELSE adjustement
                     END,
        diff        = CASE 
                          WHEN cons_type = 'le' AND adjustement > 1 THEN 0
                          WHEN cons_type = 'ge' AND adjustement < 1 THEN 0
                          ELSE diff
                    END;
    ;
    """)
  maxDiscrepancy = con.execute("SELECT max(abs(diff)) FROM wrk_discrepancies ;").fetchone()[0]
  return maxDiscrepancy


def timer(func):
  @functools.wraps(func)
  def wrapper_timer(*args, **kwargs):
    tic = perf_counter()
    value = func(*args, **kwargs)
    toc = perf_counter()
    elapsed_time = toc - tic
    print(f"Elapsed time: {elapsed_time:0.4f} seconds")
    return value
  return wrapper_timer

@timer
def IPF(input=None, 
        constraints=None, 
        targets=None, 
        unit_id="unit_id", 
        var="weight", 
        cons_id="cons_id", 
        lb=None, 
        ub=None, 
        cons_type=None,
        db_file=None, 
        tol=1, 
        maxIter=100):
  """
  input: table
      Thif table lists all the cells or units in a table whose value will be adjusted by Iterative proportional fitting along with boundaries whose adjusted value is meant to stay within.
      unit_id     : identifier for the decision variables
      weight      : decision variables. >=0
      lb			    : weight >= lb
      ub			    : weight <= up

  constraints : table
      This table maps for each constaint identifier, which unit_id to aggregate
      unit_id     : identifier for the decision variables
      cons_id     : identifiant des contraintes

  targets : table
      This table lists all the target values that the margins should add up to once adjusted
      cons_id    	: identifiant des contraintes
      cons_type	  : constraint must be greater or equal (ge) the target, lesser or equal (le), or equal (eq)
      target      : value for the constaint
  
  db_file (optional ): name fo the database file that will hold the temporary tables
  
  output : table
      Output table lists all the initials cells/units along with their adjusted values.
      untiId      : identifier for the decision variables
      weight		  : adjusted weight. Will fit in the interval lb <=	weight <= ub

  """
  print()
  print("-----------")
  print("Calibration")
  print("-----------")
  print()
  
  with duckdb.connect() as con:
    # Collect the values from dataset &targets
    n_units       = con.execute(f"SELECT COUNT(DISTINCT unit_id ) FROM input;").fetchone()[0]
    n_var         = con.execute(f"SELECT COUNT(DISTINCT cons_id ) FROM constraints;").fetchone()[0]
    
    print(f"Number of equations: {n_var}")
    print(f"Number of units    : {n_units}")
    print()
    
    # set up the working table of weights to be adjusted
    sql_select = f"SELECT {unit_id} as unit_id, {var} as weight"
    if lb:
      sql_select += ", lb"
    if ub:
      sql_select += ", ub"
    
    con.execute(f"""
    CREATE TABLE wrk_weights AS
    {sql_select}
    FROM input
    """)
    
    # read in the constraints
    con.execute(f"""
    CREATE TABLE wrk_input_constraints AS
    SELECT unit_id, cons_id
    FROM constraints
    """)
    
    # read in the target values for the constraints
    sql_select = f"SELECT cons_id, 'eq' as cons_type, target"
    if cons_type:
      sql_select = f"SELECT cons_id, {cons_type}, target"
    con.execute(f"""
    CREATE TABLE wrk_input_targets AS
    {sql_select}
    FROM targets
    """)
    
    # get the initial state of adjustment between the margins and the target margins
    maxDiscrepancy  = tol
    maxDiscrepancy  = get_discrepancy(con)
    print(f"Initial max discrepancy : {maxDiscrepancy} ")
    
    n_iter = 0
    while ( ( (maxDiscrepancy >= tol) and (n_iter <= maxIter) ) ):
      # for each unit_id, fetch the adjustment required by the constraint
      con.execute(f"""
        CREATE OR REPLACE TABLE wrk_constraints as
        SELECT a.*, b.adjustement
        FROM wrk_input_constraints as a 
        LEFT JOIN wrk_discrepancies as b
        ON a.cons_id = b.cons_id
        ;
      """)
      # compute the geometric mean of the adjustements to be made
      con.execute(f"""
        CREATE OR REPLACE TABLE wrk_unit_adjustement AS
        SELECT unit_id, exp(mean(log(adjustement))) as adjust
        FROM wrk_constraints 
        GROUP BY unit_id
      """)
      # adjust the weights
      con.execute(f"""
        CREATE OR REPLACE TABLE wrk_weights AS
        SELECT a.* EXCLUDE weight, a.weight*b.adjust  as weight
        FROM wrk_weights as a 
        LEFT JOIN wrk_unit_adjustement as b
        ON a.unit_id = b.unit_id
      """)
      # make sure the values are within bounds*/
      if lb :
           con.execute("""
        CREATE OR REPLACE TABLE wrk_weights AS
        SELECT *, GREATEST(weight, lb) AS weight
        FROM wrk_weights
        EXCLUDE weight_;
        """)
      if ub:
        con.execute("""
        CREATE OR REPLACE TABLE wrk_weights AS
        SELECT *, LEAST(weight, ub) AS weight
        FROM wrk_weights
        EXCLUDE weight_;
        """)
      
      maxDiscrepancy = get_discrepancy(con)
      
      print(f"iteration {n_iter} : {maxDiscrepancy}") 
      n_iter += 1
      
    return con.execute("SELECT * FROM wrk_weights").fetchdf()
  
