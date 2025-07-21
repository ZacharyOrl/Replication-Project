###########################################
# Solves the Lifecycle Portfolio Choice Model of 
# Cocco (2005, RFS) "Portfolio Choice in the Presence of Housing"
###########################################
# Packages 
###########################################
using Pkg

Pkg.add([
    "Parameters",
    "Interpolations",
    "Optim",
    "Plots",
    "FastGaussQuadrature",
    "LaTeXStrings",
    "Distributions",
    "CSV",
    "Dierckx",
    "StatsBase",
    "DataFrames",
])

using Parameters, LinearAlgebra, Random, Interpolations, Optim, Plots, Statistics, FastGaussQuadrature
using LaTeXStrings, Distributions, Serialization, DelimitedFiles,CSV
using Dierckx, StatsBase, DataFrames

# indir_parameters = "C:/Users/zacha/Documents/Research Ideas/Housing and Portfolio Choice/Replication/parameters"
indir_parameters = "/home/z/zorlando/Replication/parameters"
###########################################
# Parameters
###########################################
cd(indir_parameters)

# Idiosyncratic labor market parameters for each education group - taken from Cocco
# I solve the model separately for each group. 
σ_ω_nhs =   0.136^2  # No High-school
σ_ω_hs =    0.131^2  # High-school
σ_ω_clg =   0.133^2  # College

# Deterministic earnings path for each group - taken by eyeballing Cocco's Figure 1. 
κ_nhs   = vcat(hcat(CSV.File("life_cycle_income_1_eyeballed_from_paper.csv").age_group, CSV.File("life_cycle_income_1_eyeballed_from_paper.csv").age_dummies ),["Death" 0.0]) 
κ_hs   = vcat(hcat(CSV.File("life_cycle_income_2_eyeballed_from_paper.csv").age_group, CSV.File("life_cycle_income_2_eyeballed_from_paper.csv").age_dummies ),["Death" 0.0])  
κ_clg  = vcat(hcat(CSV.File("life_cycle_income_3_eyeballed_from_paper.csv").age_group, CSV.File("life_cycle_income_3_eyeballed_from_paper.csv").age_dummies ),["Death" 0.0])
###########################################
# Functions 
###########################################
# Function which sets up the key structures used in the replication
# This is where the model parameters which do not vary across education groups are set. 
include("Initialize_Model.jl") 

# A set of helper functions (e.g. utility & constraint functions) for help with solving / simulating 
include("auxilary_functions.jl") 

# Function which when given a vector of processes and an N for each process, returns a transition matrix and grid 
include("tauchen_functions.jl")

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
S = 10000 # Size of the final panel 

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
cols = [:bonds,:stocks,:stock_share,:sm_entry,:IFC_paid,
        :housing, :moved,:Inv_Move_shock, :cash,:expected_earnings,:debt, :LTV, :cons,:wealth,:bequest, 
        :income,:persistent,:transitory,:sm_shock,:age,:edu]

# Allowing η to vary between simulations 
mat_nhs  = sim_to_matrix(sim_nhs)
mat_hs   = sim_to_matrix(sim_hs)
mat_clg  = sim_to_matrix(sim_clg)

combined = vcat(mat_nhs, mat_hs, mat_clg)
df = DataFrame(combined, cols)
writedlm("SMP_F5K_update_6.csv", [mean(df.IFC_paid)])

CSV.write("simulations_panel_F5K_update_6.csv", df)

# Fixing η's path across simulations 
mat_nhs_cons  = sim_to_matrix(sim_nhs_cons)
mat_hs_cons   = sim_to_matrix(sim_hs_cons)
mat_clg_cons  = sim_to_matrix(sim_clg_cons)

combined_cons = vcat(mat_nhs_cons, mat_hs_cons, mat_clg_cons)
df_cons = DataFrame(combined_cons, cols)
mean(df_cons.IFC_paid)
CSV.write("simulations_panel_F5K_update_6_cons.csv", df_cons)
#########################################
# Checks
#########################################
#=
@unpack_Model_Parameters para 
@unpack val_func, c_pol_func, H_pol_func, LTV_pol_func, FC_pol_func, α_pol_func, Move_pol_func, S_and_B_pol_func, κ, σ_ω = sols_nhs

start_age = 25 
end_age = 70

age_grid = collect(range(start_age, length = 10, stop = end_age))


# Value function across X
plot(sols_nhs.val_func[1,1,1,1,4:nX,1])
plot!(sols_nhs.val_func[2,1,1,10,:,10])

# Consumption
plot(X_grid[10:nX],sols_nhs.c_pol_func[1,1,1,2,10:nX,9], title = "Effect of SMP on Consumption/Saving is Het. in COH", label = "Unpaid, T = 10 H = 5KK")
plot!(X_grid[1:20],sols_nhs.c_pol_func[1,2,1,2,1:20,10], label = "Paid", xlabel = "COH")
plot!(sols.c_pol_func[2,1,1,1,:,1])

# Housing 
plot(X_grid,H_pol_func[:,1,1,8,:,10]')

plot!(H_pol_func[2,1,1,5,:,10])

# Moving 
plot(X_grid,Move_pol_func[:,1,1,2,:,2]')

# Debt 
plot(X_grid, LTV_pol_func[1,1,1:3,8,:,2]')

# Stock share
plot(X_grid[1:30], sols_nhs.α_pol_func[1,1,2,1,1:30,1],title = "Impact of Housing on Portfolio Choice: T = 2", label = L"H = $5KK", xlabel = L"COH ($)", ylabel = "Stock Share")

savefig("Stock share policy function_F5K_update_6.png")
# Stock market entry payment 
plot(X_grid[1:30], X_grid[1:30] .- sols_nhs.c_pol_func[1,1,1,1,1:30,1] .- F * sols_nhs.FC_pol_func[1,1,1,1,1:30,1] .- sols_nhs.S_and_B_pol_func[1,1,1,1,1:30,1]  .- 0.94 .* sols_nhs.H_pol_func[1,1,1,1,1:30,1], xlabel = L"COH ($)") 
plot!(X_grid[1:30], FC_pol_func[1,1,1,3,1:30,10], xlabel = L"COH ($)") 

plot(X_grid[1:50], H_pol_func[1,1,1,2,1:50,2], xlabel = L"COH ($)") 
plot!(X_grid[1:50], H_pol_func[1,1,1,2,1:50,8], xlabel = L"COH ($)") 
=#
############################
# Check simulation 
############################
cd("/home/z/zorlando/Replication/Plots")
start_age = 25 
end_age = 70

age_grid = collect(range(start_age, length = 10, stop = end_age))
uage = sort(unique(df.age))

consumption_path_mean = Dict(a => mean(df.cons[df.age .== a]) for a in uage)
consumption_path_med = Dict(a => median(df.cons[df.age .== a]) for a in uage)

cash_on_hand_path =  Dict(a => mean(df.cash[df.age .== a]) for a in uage)
wealth_path = Dict(a => mean(df.wealth[df.age .== a]) for a in uage)

stock_path_med = Dict(a => median(df.stocks[df.age .== a]) for a in uage)
stock_path_mean = Dict(a => mean(df.stocks[df.age .== a]) for a in uage)
stock_share_path = Dict(a => median(df.stock_share[df.age .== a]) for a in uage)

bond_path = Dict(a => mean(df.bonds[df.age .== a]) for a in uage)

LTV_path = Dict(a => mean(df.LTV[df.age .== a]) for a in uage)
debt_path_mean = Dict(a => mean(df.debt[df.age .== a]) for a in uage)
debt_path_med = Dict(a => median(df.debt[df.age .== a]) for a in uage)

housing_path_mean = Dict(a => mean(df.housing[df.age .== a]) for a in uage)
housing_path_med = Dict(a => median(df.housing[df.age .== a]) for a in uage)

stock_market_entry_path = Dict(a => mean(df.IFC_paid[df.age .== a]) for a in uage)
moved_path = Dict(a => mean(df.moved[df.age .== a]) for a in uage[2:10])
income_path =  Dict(a => mean(df.income[df.age .== a]) for a in uage)

# Plot Paths
# Consumption
plot(consumption_path_mean, xlabel = "Period", ylabel = "Dollars", title = "Consumption by Age", label = "Mean")
plot!(consumption_path_med, xlabel = "Period", ylabel = "Dollars", title = "Consumption by Age", label = "Median")
savefig("Consumption_Path_F5K_update_6.png")

# Cash on Hand
plot(cash_on_hand_path, xlabel = "Period", ylabel = "Dollars", title = "Mean Cash by Age", label = "")
savefig("Cash_Path_F5K_update_6.png")

# Wealth
plot(wealth_path, xlabel = "Period", ylabel = "Dollars", title = "Mean Wealth by Age", label = "")
savefig("Wealth_Path_F5K_update_6.png")

# Stock Path
plot(stock_path_mean, xlabel = "Period", ylabel = "Dollars", title = "Stocks by Age", label = "Mean")
plot!(stock_path_med, xlabel = "Period", ylabel = "Dollars", label = "Median")
savefig("Stock_Path_F5K_update_6.png")

# Bonds
plot(bond_path, xlabel = "Period", ylabel = "Dollars", title = "Mean Bonds by Age", label = "")
savefig("Bond_Path_F5K_update_6.png")

# Debt Path
plot(debt_path_mean, xlabel = "Period", ylabel = "Dollars", title = "Debt by Age", label = "Mean")
plot!(debt_path_med, xlabel = "Period", ylabel = "Dollars", title = "Debt by Age", label = "Median")
savefig("Debt_Path_F5K_update_6.png")

# LTV Path
plot(LTV_path, xlabel = "Period", ylabel = "Dollars", title = "Mean LTV Ratio by Age", label = "")
savefig("LTV_Path_F5K_update_6.png")

# Housing Path
plot(housing_path_mean, xlabel = "Period", ylabel = "Dollars", title = "Housing by Age", label = "Mean")
plot!(housing_path_med, xlabel = "Period", ylabel = "Dollars", title = "Housing by Age", label = "Median")

savefig("Housing_Path_F5K_update_6.png")

# Stock Market Entry Path
plot(stock_market_entry_path, xlabel = "Period", ylabel = "Fraction", title = "Stock Market Participation by Age", label = "")
savefig("SMP_F5K_update_6.png")

# Proportion Moving
plot(moved_path, xlabel = "Period", ylabel = "Fraction", title = "Proportion Moving by Age", label = "")
savefig("Moving_Path_F5K_update_6.png")

# Income vs Consumption
plot(income_path, xlabel = "Period", ylabel = "Dollars", title = "", label = "Mean Income")
plot!(consumption_path_mean, xlabel = "Period", ylabel = "Dollars", label = "Mean Consumption")
savefig("Income_Consumption_Path_F5K_update_6.png")

# Cash on hand histogram
histogram(df.cash, xlabel = "Period", ylabel = "Dollars", title = "Cash Distribution", label = "")
savefig("Cash_Histogram_F5K_update_6.png")

# Housing Histogram 
histogram(df.housing, xlabel = "Period", ylabel = "Dollars", title = "Housing Distribution", label = "")
savefig("Housing_Histogram_F5K_update_6.png")

# Stocks Histogram 
histogram(df.stocks, xlabel = "Period", ylabel = "Dollars", title = "Stocks Distribution", label = "")
savefig("Stocks_Histogram_F5K_update_6.png")

# Consumption histogram 
histogram(df.cons, xlabel = "Period", ylabel = "Dollars", title = "Consumption Distribution", label = "")
savefig("Consumption_Histogram_F5K_update_6.png")