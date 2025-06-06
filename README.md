# Replication-Project

This codebase replicates João F. Cocco's paper [Portfolio Choice in the Presence of Housing](https://www.jstor.org/stable/3598045?seq=1) (2005,Review of Financial Studies). I am now completing the write-up of the results. 

I completed this as part of the Macroeconomics field at UW Madison.  

# Files 
Data 
1. `psid_hp_analysis.do` - uses PSID - SHELF data to replicate housing estimates. 
2. `psid_income_analysis.do` - uses PSID - Equivalent data to replicate income estimates. 

Model
1. `portfolio_choice_with_housing.jl` - file solves and simulates the model in Cocco (2005) using their parameter estimates. 
2. `Initialize_Model.jl` - Function which sets up the key structures used in the replication
3. `auxilary_functions.jl` - A set of helper functions (e.g. utility & constraint functions) for help with solving / simulating 
4. `rouwenhorst.jl` - discretizes a potentially persistent normal process onto a grid using the Rouwenhorst method. 
5. `Solve_Retiree_Problem.jl` - solves the retiree's problem and stores the values into the solutions structure. 
6. `Solve_Worker_Problem.jl` -  solves the worker's problem, conditional on having already solved the retiree's problem.
7. `Simulate_Model.jl` -  simulates the full lifecycle model and outputs a simulation structure using the 1989 PSID education - age fractions as weights. 

Figures 
1. `inc_and_hp_comparison_plots.jl` - plots deterministic income and house prices from against Cocco's estimates. 
2. `Table4_5.jl` - replicates Cocco's summary stats. on portfolio shares by financial assets (Table 4) and age (Table 5) using model-simulated data. 
3. `Table6.do` - replicates Cocco's Tobit regressions of stock market participation on various characteristics using model-simulated data. 