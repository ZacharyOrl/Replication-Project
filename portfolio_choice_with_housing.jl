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
#########################################
# Checks
#########################################
@unpack_Model_Parameters para 
@unpack val_func, c_pol_func, H_pol_func, LTV_pol_func, FC_pol_func, α_pol_func, Move_pol_func, κ, σ_ω = sols_nhs

start_age = 25 
end_age = 70

age_grid = collect(range(start_age, length = 10, stop = end_age))

plot(1:T+1, val_func[1,1,3,1,1,1:T+1])

# Value function across X
plot(X_grid[25:nX],sols_nhs.val_func[1,1,1,1,:,10][25:nX])
plot!(sols_nhs.val_func[2,1,1,10,:,10])

# Consumption
plot(X_grid,c_pol_func[1,1,1,1,:,10] ./X_grid)
plot!(sols.c_pol_func[2,1,1,1,:,1])

# Housing 
plot(X_grid,H_pol_func[1,1,1:3,1,:,8]')
plot!(H_pol_func[2,1,1,5,:,10])

# Moving 
plot(X_grid,Move_pol_func[1,1,2,2,:,2])

# Debt 
plot(X_grid,LTV_pol_func[1,1,1:3,1,:,2]')
plot!(sols.D_pol_func[2,1,1,:,:,10])

# Stock share 
plot(α_pol_func[1,1,1,1,:,:]) 

# Stock market entry payment 
plot(X_grid[10:25], FC_pol_func[1,1,1,1,10:25,1]) 

# CHeck constraints
v = sols.val_func[1,2,1,1,3,10]
D = sols.D_pol_func[1,2,1,1,3,10]
c = sols.c_pol_func[1,2,1,1,3,10]
FC = Int64(sols.FC_pol_func[1,2,1,1,3,10])
α = sols.α_pol_func[1,2,1,1,3,10]
H_prime = sols.H_pol_func[1,2,1,1,3,10]
H_prime_index = 1
# Compute X_prime
j = 10
X = X_grid[3]
H = H_grid[1]
Inv_Move = 0
η_index = 1
IFC_index = 2
IFC = IFC_grid[IFC_index]

P = 1 * exp(b * (j-1) + p_grid[η_index])

# S_and_B  = budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)
budget_constraint(X, H, P, Inv_Move,c, H_prime, D, FC, para)
debt_constraint(D , H_prime, P, para)
find_zero(c -> budget_constraint(X, H, P, Inv_Move,c, H_prime, D, FC, para), 100.0)
sols.val_func[9,2,2,1,1,1]
R_prime = exp(ι_prime + μ)
Y_Prime = κ[j, 2]

S = α * S_and_B
B = (1-α) * S_and_B
# Compute next period's liquid wealth
X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime

P =  P_bar * exp(b * (10-1) + p_grid[η_index])
X = X_grid[1]
S_and_B  = budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)

Y = zeros(10,1000)

for s = 1:1000
    for j = 1:10
    ω = rand(ω_grid)
    η = rand(η_grid)
    Y[j,s] = κ[j+1, 2] * exp(ω + η)
    end 
end 

############################
# Check simulation 
############################
@unpack_Model_Parameters para 
@unpack val_func, c_pol_func, H_pol_func, LTV_pol_func, FC_pol_func, α_pol_func, κ, σ_ω = sols_nhs
start_age = 25 
end_age = 70

age_grid = collect(range(start_age, length = 10, stop = end_age))
uage = sort(unique(sim_nhs.age))

consumption_path = Dict(a => mean(sim_nhs.consumption[sim_nhs.age .== a]) for a in uage)
cash_on_hand_path =  Dict(a => mean(sim_nhs.cash_on_hand[sim_nhs.age .== a]) for a in uage)
wealth_path = Dict(a => mean(sim_nhs.wealth[sim_nhs.age .== a]) for a in uage)
stock_path = Dict(a => mean(sim_nhs.stocks[sim_nhs.age .== a]) for a in uage)
stock_share_path = Dict(a => mean(sim_nhs.stock_share[sim_nhs.age .== a]) for a in uage)
bond_path = Dict(a => mean(sim_nhs.bonds[sim_nhs.age .== a]) for a in uage)
LTV_path = Dict(a => mean(sim_nhs.LTV[sim_nhs.age .== a]) for a in uage)
debt_path = Dict(a => mean(sim_nhs.debt[sim_nhs.age .== a]) for a in uage)
housing_path = Dict(a => mean(sim_nhs.housing[sim_nhs.age .== a]) for a in uage)
stock_market_entry_path = Dict(a => mean(sim_nhs.IFC_paid[sim_nhs.age .== a]) for a in uage)
moved_path = Dict(a => mean(sim_nhs.moved[sim_nhs.age .== a]) for a in uage)
Dict(a => mean(m2[age .== a]) for a in uage)

plot(consumption_path)
plot(cash_on_hand_path)
plot(wealth_path)
plot(stock_path)
plot(bond_path)
plot(debt_path)
plot(LTV_path)
plot(housing_path)
plot(stock_market_entry_path)
plot(moved_path)

histogram(sim_hs.cash_on_hand)
histogram(sim_hs.housing)
histogram(sim_nhs.bonds[:,1])
histogram(sim_nhs.cash_on_hand[:,1])
histogram(sim_nhs.debt[:,3])
histogram(sim_nhs.stocks[:,1])

histogram(sim_nhs.cash_on_hand[:,5])
histogram(sim_hs.stocks[:,1])
histogram(sim_hs.stock_market_entry[:,1])
histogram(sim_nhs.bequest)
histogram(stocks .+ bonds)

# Stocks check 
# Does anyone enter the stock market while holding 0 stocks? 
stocks_chk = sim_nhs.stocks[:,1]
stocks_entry_chk = sim_nhs.stock_market_entry[:,1]
no_stocks = ifelse.(stocks_chk .== 0.0, stocks_entry_chk, missing)

plot(sim_nhs.housing[1:5,:])
histogram(filtered)

Threads.nthreads()