###########################################
# Solves the Lifecycle Portfolio Choice Model of 
# Cocco (2005, RFS) "Portfolio Choice in the Presence of Housing"
###########################################
# Packages 
###########################################
using Parameters, DelimitedFiles, CSV, Plots, Distributions,LaTeXStrings, Statistics, FastGaussQuadrature 
using DataFrames, LinearAlgebra, Optim, Interpolations, Base.Threads, Roots, StatsBase, Random

# indir_parameters = "C:/Users/zacha/Documents/Research Ideas/Housing and Portfolio Choice/Replication/parameters"
indir_parameters = "parameters"
###########################################
# Parameters
###########################################
cd(indir_parameters)

# Idiosyncratic labor market parameters for each education group - taken from Cocco
# I solve the model separately for each group. 
σ_ω_nhs =  0.136^2  # No High-school
σ_ω_hs =   0.131^2  # High-school
σ_ω_clg =  0.133^2  # College

# Deterministic earnings path for each group - taken by eyeballing Cocco's Figure 1. 
κ_nhs   = vcat(hcat(CSV.File("life_cycle_income_1_eyeballed_from_paper.csv").age_group, CSV.File("life_cycle_income_1_eyeballed_from_paper.csv").age_dummies),["Death" 0.0])
κ_hs   = vcat(hcat(CSV.File("life_cycle_income_2_eyeballed_from_paper.csv").age_group, CSV.File("life_cycle_income_2_eyeballed_from_paper.csv").age_dummies),["Death" 0.0])
κ_clg  = vcat(hcat(CSV.File("life_cycle_income_3_eyeballed_from_paper.csv").age_group, CSV.File("life_cycle_income_3_eyeballed_from_paper.csv").age_dummies),["Death" 0.0])
###########################################
# Functions 
###########################################
# Function which sets up the key structures used in the replication
# This is where the model parameters which do not vary across education groups are set. 
include("Initialize_Model.jl") 

# A set of helper functions (e.g. utility & constraint functions) for help with solving / simulating 
include("auxilary_functions.jl") 

# Function which when given a vector of processes and an N for each process, returns a transition matrix and grid 
include("rouwenhorst.jl")

# Function which computes the stationary distribution of a Markov Chain 
include("stationary_distribution.jl")

# Function which solves the retiree's problem and stores the values into the solutions structure 
include("Solve_Retiree_Problem.jl")

# Function which solves the woker's problem, conditional on having already solved the retiree's problem, 
# and stores the values into a Solutions structure defined in Initialize
include("Solve_Worker_Problem.jl")

# Function which simulates S agent's lifecycles given having already solved the lifecycle problem 
# and stores the values into a Sim_Results structure defined in Initialize
include("Simulate_model.jl")
#########################################
# Solve the Model!
#########################################
# Solve the model for each group of agents 

# No highschool
para, sols_nhs = Initialize_Model(σ_ω_nhs, κ_nhs)
@time Solve_Retiree_Problem(para, sols_nhs)
@time Solve_Worker_Problem(para, sols_nhs)

# High school
para, sols_hs = Initialize_Model(σ_ω_hs, κ_hs)
@time Solve_Retiree_Problem(para, sols_hs)
@time Solve_Worker_Problem(para, sols_hs)

# College 
para, sols_clg = Initialize_Model(σ_ω_clg, κ_clg)
@time Solve_Retiree_Problem(para, sols_clg)
@time Solve_Worker_Problem(para, sols_clg)

#########################################
# Simulate the Model! 
#########################################
S = 10000

# Returns a simulation struct containing the model outputs

# Allowing both the aggregate state and idiosyncratic states to vary
sim_nhs = simulate_model(para, sols_nhs, S, 1)
sim_hs = simulate_model(para, sols_hs, S, 2)
sim_clg = simulate_model(para, sols_clg, S, 3)

# Imposing that the aggregate state matches the one closest to the data 
sim_nhs_cons = simulate_model_constant_shocks(para, sols_nhs, S, 1)
sim_hs_cons = simulate_model_constant_shocks(para, sols_hs, S, 2)
sim_clg_cons = simulate_model_constant_shocks(para, sols_clg, S, 3)
#########################################
# Write 
#########################################
cols = [:bonds,:stocks,:stock_share,:sm_entry,:IFC_paid,:LTV,
        :housing, :moved,:cash,:expected_earnings,:debt,:cons,:wealth,:bequest,
        :persistent,:transitory,:sm_shock,:age,:edu]

# Allowing η to vary between simulations 
mat_nhs  = sim_to_matrix(sim_nhs)
mat_hs   = sim_to_matrix(sim_hs)
mat_clg  = sim_to_matrix(sim_clg)

combined = vcat(mat_nhs, mat_hs, mat_clg)
df = DataFrame(combined, cols)

CSV.write("simulations_panel_eyeballed_from_paper.csv", df)

# Fixing η's path across simulations 
mat_nhs_cons  = sim_to_matrix(sim_nhs_cons)
mat_hs_cons   = sim_to_matrix(sim_hs_cons)
mat_clg_cons  = sim_to_matrix(sim_clg_cons)

combined_cons = vcat(mat_nhs_cons, mat_hs_cons, mat_clg_cons)
df_cons = DataFrame(combined_cons, cols)

CSV.write("simulations_panel_eyeballed_from_paper_fixed_shocks.csv", df_cons)