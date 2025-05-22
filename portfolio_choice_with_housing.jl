###########################################
# Solves the Lifecycle Portfolio Choice Model of 
# Cocco (2005, RFS) "Portfolio Choice in the Presence of Housing"
###########################################
# Packages 
###########################################
using Parameters, DelimitedFiles, CSV, Plots, Distributions,LaTeXStrings, Statistics, FastGaussQuadrature 
using DataFrames, LinearAlgebra, Optim, Interpolations, Base.Threads, Roots, StatsBase
###########################################
# indir = "C:/Users/zacha/Documents/Research Ideas/Housing and Portfolio Choice/Replication"
# indir = "/Client/C$/Users/zacha/Documents/Research Ideas/Housing and Portfolio Choice/Replication"
# indir = "/Z:/Replication" - Need to put the folder on linux to use. 

# indir_parameters = "C:/Users/zacha/Documents/Research Ideas/Housing and Portfolio Choice/Replication/parameters"
indir_parameters = "parameters"
###########################################
# Parameters
###########################################
cd(indir_parameters)

@with_kw struct Model_Parameters
    # Number of gridpoints for random variables 
    g = 7

    # Parameters - Converting all annual variables to their five-year equivalents
    # Variance Parameters 

    # Variance of aggregate component of earnings, 
    # 16.64 is the five year variance of an annual AR(1) with autocorrelation of 0.748 and var 0.019^2 
    σ_η::Float64 =     16.64 *  0.019^2  
    σ_p::Float64 =     16.64 *  0.062^2  # Variance of house prices - perfectly correlated with aggregate labor market. 
    σ_ι::Float64 =     5  * 0.1674^2  # Variance of stock market innovation

    # Correlations between processes 
    κ_ω::Float64 = 0.00     # Correlation between house prices and transitory component
    κ_η::Float64 = sqrt(σ_η/σ_p)  # Regression coefficient of cyclical fluctuations in house prices on aggregae component (correlation is 1)
    ρ_ϵ_ι::Float64 = 0.0    # Correlation between aggregae component of income and the stock market
    φ::Float64 = 0.415      # Autocorrelation in the aggregae component 

    # One time stock market entry cost 
    F::Float64 = 8000.0 

    # Returns / interest rates 
    R_D::Float64 =  compound(1 + 0.04, 5)                  # Mortgage interest rate 
    R_F::Float64 =  compound(1 + 0.02, 5)                  # Risk-free rate
    R_S::Float64 =  compound(1 + 0.08, 5)                   # Expected return on stocks 
    μ::Float64 = log(R_S) - σ_ι/2                          # Expected five-year log-return on stocks

    # Housing parameters
    d::Float64 = 0.15                               # Down-Payment proportion 
    π_m::Float64 = 1 - (1 - 0.032)^5                  # Moving shock probability 
    δ::Float64 = 1 - 0.99^5                         # Housing Depreciation
    λ::Float64 = 0.08                               # House-sale cost 
    b::Float64 = 5 * 0.01                           # Real log house price growth over 5 years  - matching the way it is presented in the paper 

    # Utility function parameters
    θ::Float64 = 0.1              # Utility from housing services relative to consumption
    γ::Float64 = 5.0              # Risk-aversion parameter 
    β::Float64 = compound(0.96,5) # Discount Rate 

    # Lifecycle Parameters 
    T::Int64 = 10 # Each T represents five years of life from 25 to 75
    TR::Int64 = 9 # The final two time periods represent retirement 

    # Grids & Transition Matrices
    # aggregae Earnings
    η_grid::Vector{Float64} =  tauchen_persistent(3,φ,sqrt((1-φ^2)*σ_η))[1] 
    T_η::Matrix{Float64} = tauchen_persistent(3,φ,sqrt((1-φ^2)*σ_η))[2]
    nη::Int64 = length(η_grid)

    # Compute the stationary distribution of aggregate earnings
    # This will be the distribution over the initial aggregate state 
    π_η::Vector{Float64} = stationary_distribution(η_grid, T_η)

    # Stock market grids
    ι_grid::Vector{Float64} = tauchen_hussey_iid(g,sqrt(σ_ι), 0.0)[1]
    T_ι::Matrix{Float64} = tauchen_hussey_iid(g,sqrt(σ_ι), 0.0)[2][1:g,1:1]'
    nι::Int64 = length(ι_grid)

    # Housing grids
    p_grid::Vector{Float64} = η_grid ./κ_η
    P_bar::Float64 = 1 # Don't do anything - just let prices grow  [For now, deflate using the ratio of CPI in 1972 to CPI in 1992 
                                          # CPI taken from https://fred.stlouisfed.org/series/CPIAUCSL   * 41.8/140.3]
    np::Int64 = nη

    # Punishment value 
    pun::Float64 = -Inf # The value agents face if they default. 
    # State / Choice Grids 
    X_min::Float64 = -1400000.0
    X_max::Float64 =  4000000.0
    nX::Int64 = 201
    X_grid::Vector{Float64} = collect(range(X_min, length = nX, stop = X_max))

    H_min::Float64 = 20000.0
    H_max::Float64 = 700000.0
    nH::Int64 = 29

    # Agents start life with no housing and are forced to purchase a home in the first period. 
    H_choice_grid::Vector{Float64} = collect(range(H_min, length = nH, stop = H_max))
    H_state_grid::Vector{Float64} = vcat(0.0, H_choice_grid)

    α_min::Float64 = 0.0
    α_max::Float64 = 1.0
    nα::Int64 = 50
    α_grid::Vector{Float64} = collect(range(α_min, length = nα, stop = α_max))

    Inv_Move_grid::Vector{Int64} = [0, 1]

    FC_grid::Vector{Int64}       = [0, 1]

    IFC_grid::Vector{Int64}      = [0, 1]

    # Index grid 
    lin::LinearIndices{4,Tuple{Base.OneTo{Int64},Base.OneTo{Int64},Base.OneTo{Int64},Base.OneTo{Int64}}} = LinearIndices((2, 2, nη, nH + 1))
    
    tol::Float64 = 500.0          # stop optimizing once the candidate bracket is ≤ $1000 wide

    # Weighting grid for simulations - taken from Cocco 
                            #    nhs  hs    clg
    weights::Matrix{Float64} = [
                                0.01  0.06  0.02;   # 25–29
                                0.02  0.08  0.04;   # 30–34
                                0.01  0.10  0.06;   # 35–39
                                0.0175  0.075  0.0675;   # 40–44 - Values slightly adjusted to account for Cocco rounding
                                0.01  0.05  0.03;   # 45–49
                                0.0175  0.035  0.0175;   # 50–54 - Values slightly adjusted so the row sum matches Cocco
                                0.02  0.04  0.02;   # 55–59
                                0.02  0.03  0.02;   # 60–64
                                0.02  0.04  0.01;   # 65–69
                                0.02  0.03  0.01    # 70–74
                                ]
                            
end

#initialize value function and policy functions
mutable struct Solutions

    # 6 states, it turns out that the retired's value function still depends on η even after retirement,as it pins down housing.
    val_func::Array{Float64,6} 
    c_pol_func::Array{Float64,6}
    H_pol_func::Array{Float64,6}
    D_pol_func::Array{Float64,6}
    FC_pol_func::Array{Float64,6}
    α_pol_func::Array{Float64,6}

    σ_ω::Float64
    κ::Matrix{Any}

end

function build_solutions(para, σ_ω::Float64, κ::Matrix{Any}) 

    # Value function has an extra level for housing due to the initial housing state of 0.0 
    # It has an extra level for age due to bequest. 
    val_func    = zeros(Float64,2,2,para.nη, para.nH + 1, para.nX, para.T + 1 ) 
    c_pol_func  = zeros(Float64,2,2,para.nη, para.nH + 1, para.nX, para.T ) 
    H_pol_func  = zeros(Float64,2,2,para.nη, para.nH + 1, para.nX, para.T ) 
    D_pol_func  = zeros(Float64,2,2,para.nη, para.nH + 1, para.nX, para.T ) 
    FC_pol_func = zeros(Float64,2,2,para.nη, para.nH + 1, para.nX, para.T ) 
    α_pol_func  = zeros(Float64,2,2,para.nη, para.nH + 1, para.nX, para.T ) 

    sols = Solutions(val_func, c_pol_func, H_pol_func, D_pol_func, FC_pol_func, α_pol_func, σ_ω , κ)

    return sols
end 

function Initialize_Model(σ_ω::Float64, κ::Matrix{Any}) 

    para = Model_Parameters()
    sols = build_solutions(para, σ_ω, κ)

    return para, sols 
end

# Idiosyncratic labor market parameters 
σ_ω_nhs =  5 * 0.136^2  # No High-school
σ_ω_hs =   5 * 0.131^2  # High-school
σ_ω_clg =  5 * 0.133^2  # College

# Deterministic earnings path for each group. 
κ_nhs   = vcat(hcat(CSV.File("life_cycle_income_1_eyeballed_from_paper.csv").age_group, CSV.File("life_cycle_income_1_eyeballed_from_paper.csv").age_dummies),["Death" 0.0])
κ_hs   = vcat(hcat(CSV.File("life_cycle_income_2_eyeballed_from_paper.csv").age_group, CSV.File("life_cycle_income_2_eyeballed_from_paper.csv").age_dummies),["Death" 0.0])
κ_clg  = vcat(hcat(CSV.File("life_cycle_income_3_eyeballed_from_paper.csv").age_group, CSV.File("life_cycle_income_3_eyeballed_from_paper.csv").age_dummies),["Death" 0.0])

###########################################
# Functions 
###########################################
# Function which when given an annual rate returns the compounded rate over T periods. 
include("compound.jl") 

# Function which when given a vector of processes and an N for each process, returns a transition matrix and grid 
include("rouwenhorst.jl")

# Tauchen functions from Cutberto 
include("tauchen_functions.jl")

# Function which computes the stationary distribution of a Markov Chain 
include("stationary_distribution.jl")

# Function which computes an initial distribution over a given grid 
include("compute_initial_dist.jl")

# Function which solves the retiree's problem and stores the values into the solutions structure 
include("Solve_Retiree_Problem.jl")

# Function which solves the woker's problem, conditional on having already solved the retiree's problem, and stores the values into the solutions structure 
include("Solve_Worker_Problem.jl")

# Function which simulates S agent's lifecycles given having already solved the lifecycle problem 
include("Simulate_model.jl")

#########################################################
# Functions 
#########################################################
function flow_utility_func(c::Float64, H_prime::Float64, para::Model_Parameters)
    @unpack γ, θ = para

    return  (    ( c^(1-θ) * H_prime^θ )^( 1 - γ )   ) / (1 - γ)
end 

# Takes as input all states and choices necessary to pin down the budget constraint
# and outputs the sum of stocks and bonds (LHS of the budget constraint in Cocco)
function budget_constraint(X::Float64, H::Float64, P::Float64, Inv_Move::Int64, c::Float64, 
                           H_prime::Float64, D::Float64, FC::Int64, para::Model_Parameters)
    @unpack δ, F, λ = para

    # If there is no house trade
    if (Inv_Move == 0) && (H_prime == H)
        S_and_B = X - c - FC*F - δ * P * H + D
    # If there is a house trade 
    else 
        S_and_B = X - c - FC*F - δ * P * H + D + (1-λ)* P * H - P * H_prime
    end 

    return S_and_B
end 

# Reports the difference between debt taken out today and the collateral limit: 
function debt_constraint(D::Float64, H_prime::Float64, P::Float64,para::Model_Parameters)
    @unpack_Model_Parameters para

    return (1-d) * H_prime * P - D
end 

# Given realized wealth of a state in T+1
function compute_bequest_value(V::Array{Float64,5}, para::Model_Parameters)
    @unpack_Model_Parameters para 

    # Loop over Cash-on-hand states
    for X_index in 1:nX
        X = X_grid[X_index]

        # Loop over Housing States 
        for H_index in 1:nH + 1
            H = H_state_grid[H_index]
            
            # Loop over aggregate income states
            for η_index in 1:nη
                η =  η_grid[η_index]

                P = P_bar * exp(b * (T+1) + p_grid[η_index])
    
                # Agents are forced to sell their house when they die
                W = X - δ * H * P +  (1-λ) *  P * H     
                
                # Account for agents essentially taking on impractical levels of debt that could leave them endowing nothing. 
                if W < 0 
                    V[:, :, η_index, H_index, X_index] .+= pun 
                else

                V[:, :, η_index, H_index, X_index] .+= ( W^(1-γ) )/(1-γ)  
                end 
            end 
        end 
    end 

    return V
end 

# Creates a linear interpolation function using a mapping from grid x1 to outcome F. 
# Cubic-spline interpolation within-grid - changed to linear because cubic spline wasn't working well for a pretty sparse grid. 
# Allows for extrapolation outside the grid (Flat extrapolation)
function linear_interp(F::Vector{Float64}, x1::Vector{Float64})
    x1_grid = range(minimum(x1), maximum(x1), length=length(x1))

    interp = interpolate(F, BSpline(Linear()))
    scaled_interp = Interpolations.scale(interp,x1_grid)
    return  scaled_interp
end

function sim_to_matrix(sim::SimResults)
    #=
    sim_to_matrix(sim)

    Flatten every (S × T+1) field in `sim` into a column vector so the result is
    `S*(T+1) × K`, where `K` is the number of variables.
    =#

    S, TT = size(sim.bonds)        # TT = T+1
    N     = S*TT

    # helper: vec(mat) flattens column-major into length-N vector
    rep(x) = repeat(x, inner = TT)           # repeat vector S times

    return hcat(
        vec(sim.bonds),           vec(sim.stocks),            vec(sim.stock_share),
        vec(sim.stock_market_entry), vec(sim.IFC_paid),
        vec(sim.housing),         vec(sim.moved),             vec(sim.cash_on_hand),
        vec(sim.debt),            vec(sim.consumption),       vec(sim.wealth),
        rep(sim.bequest),                               # 1×S → repeat across ages
        vec(sim.persistent),      vec(sim.transitory),        vec(sim.stock_market_shock),
        vec(sim.weights_var),     vec(sim.education)   
    )
end
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
sim_nhs = @time simulate_model(para, sols_nhs, S, 1)
sim_hs = @time simulate_model(para, sols_hs, S, 2)
sim_clg = @time simulate_model(para, sols_clg, S, 3)

#########################################
# Write 
#########################################
mat_nhs  = sim_to_matrix(sim_nhs)
mat_hs   = sim_to_matrix(sim_hs)
mat_clg  = sim_to_matrix(sim_clg)

combined = vcat(mat_nhs, mat_hs, mat_clg)

cols = [:bonds,:stocks,:stock_share,:sm_entry,:IFC_paid,
        :housing,:moved,:cash,:debt,:cons,:wealth,:bequest,
        :persistent,:transitory,:sm_shock,:wt,:edu]

df = DataFrame(combined, cols)

CSV.write("simulations_panel_eyeballed_from_paper.csv", df)
#########################################
# Checks
#########################################
@unpack_Model_Parameters para 
@unpack val_func, c_pol_func, H_pol_func, D_pol_func, FC_pol_func, α_pol_func, κ, σ_ω = sols_nhs

start_age = 25 
end_age = 70

age_grid = collect(range(start_age, length = 10, stop = end_age))

plot(1:T+1, val_func[1,1,3,1,1,1:T+1])

# Value function across X
plot(sols_nhs.val_func[1,1,1,10,18:51,10])
plot!(sols_nhs.val_func[2,1,1,10,18:51,10])

# Consumption
plot(c_pol_func[1,1,1,:,25,:]')
plot!(sols.c_pol_func[2,1,1,1,:,1])

# Housing 
plot(H_pol_func[1,1,1,10,25,:])
plot!(H_pol_func[2,1,1,5,:,10])

# Debt 
plot(X_grid,D_pol_func[1,1,1,1,25,:])
plot!(sols.D_pol_func[2,1,1,:,:,10])

# Stock share 
plot(X_grid, α_pol_func[1,1,1,:,:,9]') 

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
@unpack val_func, c_pol_func, H_pol_func, D_pol_func, FC_pol_func, α_pol_func, κ, σ_ω = sols_nhs
start_age = 25 
end_age = 70

age_grid = collect(range(start_age, length = 10, stop = end_age))

consumption_path = mean(sim_nhs.consumption, dims = 1)[1:T]
cash_on_hand_path = mean(sim_nhs.cash_on_hand, dims = 1)[1:T]
wealth_path = mean(sim_nhs.wealth, dims = 1)[1:T]
stock_path = mean(sim_nhs.stocks, dims = 1)[1:T]
debt_path = mean(sim_nhs.debt, dims = 1)[1:T]
housing_path = mean(sim_nhs.housing, dims = 1)[1:T]
stock_market_entry_path = mean(sim_nhs.IFC_paid, dims = 1)[1:T]
moved_path = mean(sim_nhs.moved, dims = 1)[1:T]

plot(age_grid, consumption_path)
plot(age_grid, cash_on_hand_path)
plot(age_grid, wealth_path)
plot(age_grid,stock_path)
plot(age_grid,debt_path)
plot(age_grid,housing_path)
plot(age_grid,stock_market_entry_path)
plot(age_grid,moved_path)

histogram(sim_hs.debt[:,4])
histogram(sim_hs.bonds[:,1])
histogram(sim_hs.cash_on_hand[:,2])
histogram(sim_hs.housing[:,6])
histogram(sim_hs.stocks[:,2])
histogram(sim_hs.stock_market_entry[:,2])
histogram(sim_hs.bequest)
histogram(stocks .+ bonds)

# Stocks check 
# Does anyone enter the stock market while holding 0 stocks? 
stocks_chk = sim_hs.stocks[:,1]
stocks_entry_chk = sim_hs.stock_market_entry[:,1]
no_stocks = ifelse.(stocks_chk .== 0.0, stocks_entry_chk, missing)


histogram(filtered)

Threads.nthreads()