###########################################
# Solves the Lifecycle Portfolio Choice Model of 
# Cocco (2005, RFS) "Portfolio Choice in the Presence of Housing"
###########################################
# Packages 
###########################################
using Parameters,DelimitedFiles, CSV, Plots, Distributions,LaTeXStrings, Statistics, DataFrames, LinearAlgebra, Optim, Interpolations, Base.Threads, Roots, StaticArrays
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
    g = 3

    # Parameters - Converting all annual variables to their five-year equivalents
    # Variance Parameters 

    # Variance of aggregate component of earnings, 
    # 2.142 is the five year variance of an annual AR(1) with autocorrelation of 0.748 and var 0.019^2  
    σ_η::Float64 = 2.142 * 0.019^2  
    σ_p::Float64 = 2.142 * 0.062^2  # Variance of house prices - perfectly correlated with aggregate labor market. 
    σ_ι::Float64 = 5 * 0.1674^2 # Variance of stock market innovation
    σ_ω::Float64 = 0.136^2  # Start with the no-college case 

    # Correlations between processes 
    κ_ω::Float64 = 0.00     # Correlation between house prices and transitory component
    κ_η::Float64 = σ_η/σ_p  # Regression coefficient of cyclical fluctuations in house prices on persistent component (correlation is 1)
    ρ_ϵ_ι::Float64 = 0.0    # Correlation between persistent component of income and the stock market
    φ::Float64 = compound(0.748,5)      # Autocorrelation in the persistent component 

    # One time stock market entry cost 
    F::Float64 = 1000.0 

    # Returns / interest rates 
    R_D::Float64 =  compound(1 + 0.04, 5)                  # Mortgage interest rate 
    R_F::Float64 =  compound(1 + 0.02, 5)                  # Risk-free rate
    R_S::Float64 =  compound(1 + 0.1, 5)                   # Expected return on stocks 
    μ::Float64 = log(R_S) - σ_ι/2                          # Expected five-year log-return on stocks

    # Housing parameters
    d::Float64 = 0.15                               # Down-Payment proportion 
    π::Float64 = compound(1 + 0.032, 5) - 1          # Moving shock probability 
    δ::Float64 = compound(1 + 0.01, 5) - 1          # Housing Depreciation
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
    # Persistent Earnings
    η_grid::Vector{Float64} = rouwenhorst(σ_η, φ, 3)[1] 
    T_η::Matrix{Float64} = rouwenhorst(σ_η, φ, 3)[2]
    nη::Int64 = length(η_grid)

    # Transitory Earnings
    ω_grid::Vector{Float64} = rouwenhorst(σ_ω, 0.0, 3)[1] 
    T_ω::Matrix{Float64} = rouwenhorst(σ_ω, 0.0, 3)[2]
    nω::Int64 = length(ω_grid)

    # In the paper, there is no initial level of housing and LW_1 = 0 
    # But what is η_0? He doesn't say... 
    η_0::Float64 = 0.0 # For now, assume that persistent earnings starts at the unconditional mean

    # Deterministic earnings
    # For now, use what my own estimates from the PSID. 
    κ::Matrix{Any} = vcat(hcat(CSV.File("life_cycle_income_1.csv").age_group, CSV.File("life_cycle_income_1.csv").age_dummies),["Death" 0.0])

    # Stock market grids
    ι_grid::Vector{Float64} = rouwenhorst(σ_ι, 0.0, 3)[1]
    T_ι::Matrix{Float64} = rouwenhorst(σ_ι, 0.0, 3)[2]
    nι::Int64 = length(ι_grid)

    # Housing grids
    p_grid::Vector{Float64} = η_grid ./κ_η
    P_bar::Float64 = 1 # Don't do anything - just let prices grow  [For now, deflate using the ratio of CPI in 1972 to CPI in 1992 
                                          # CPI taken from https://fred.stlouisfed.org/series/CPIAUCSL   * 41.8/140.3]
    np::Int64 = nη

    # Punishment value 
    pun::Float64 = -10^6 # The value agents face if they default. 
    # State / Choice Grids 
    X_min::Float64 = -100000.0
    X_max::Float64 = 100000.0
    nX::Int64 = 201
    X_grid::Vector{Float64} = collect(range(X_min, length = nX, stop = X_max))

    H_min::Float64 = 20000.0
    H_max::Float64 = 600000.0
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
    
    tol::Float64 = 100.0          # stop optimizing once the candidate bracket is ≤ $100 wide
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

end

function build_solutions(para) 

    # Value function has an extra level for housing due to the initial housing state of 0.0 
    # It has an extra level for age due to bequest. 
    val_func    = zeros(Float64,2,2,para.nη, para.nH + 1, para.nX, para.T + 1 ) 
    c_pol_func  = zeros(Float64,2,2,para.nη, para.nH + 1, para.nX, para.T ) 
    H_pol_func  = zeros(Float64,2,2,para.nη, para.nH + 1, para.nX, para.T ) 
    D_pol_func  = zeros(Float64,2,2,para.nη, para.nH + 1, para.nX, para.T ) 
    FC_pol_func = zeros(Float64,2,2,para.nη, para.nH + 1, para.nX, para.T ) 
    α_pol_func  = zeros(Float64,2,2,para.nη, para.nH + 1, para.nX, para.T ) 

    sols = Solutions(val_func, c_pol_func, H_pol_func, D_pol_func, FC_pol_func, α_pol_func)

    return sols
end 

function Initialize_Model() 

    para = Model_Parameters()
    sols = build_solutions(para)

    return para, sols 
end

###########################################
# Functions 
###########################################
# Function which when given an annual rate returns the compounded rate over T periods. 
include("compound.jl") 

# Function which when given a vector of processes and an N for each process, returns a transition matrix and grid 
include("rouwenhorst.jl")

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

    return 10^20 *(    ( c^(1-θ) * H_prime^θ )^( 1 - γ )   ) / (1 - γ)
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

                P = P_bar * exp(b * (T) + p_grid[η_index])
    
                # Agents are forced to sell their house when they die
                W = X - δ * H * P +  (1-λ) *  P * H     
                
                # Account for agents essentially taking on impractical levels of debt that could leave them endowing nothing. 
                if W < 0 
                    V[:, :, η_index, H_index, X_index] .+= pun 
                else

                V[:, :, η_index, H_index, X_index] .+= 10^20 * ( W^(1-γ) )/(1-γ)  
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
    extrap = extrapolate(scaled_interp, Interpolations.Flat())
    return  extrap
end

#########################################
# Solve the Model!
#########################################
para, sols = Initialize_Model()
@time Solve_Retiree_Problem(para, sols)
@time Solve_Worker_Problem(para, sols)
#########################################
# Simulate the Model! 
#########################################
S = 10000
bonds,stocks,stock_share,stock_market_entry,IFC_paid,housing,cash_on_hand,debt,consumption,wealth,bequest,persistent,transitory,stock_market_shock = @time simulate_model(para,sols,S)
#########################################
# Checks
#########################################
@unpack_Model_Parameters para 
@unpack val_func, c_pol_func, H_pol_func, D_pol_func, FC_pol_func, α_pol_func = sols

start_age = 25 
end_age = 70

age_grid = collect(range(start_age, length = 10, stop = end_age))

plot(1:T+1, val_func[1,1,3,1,1,1:T+1])

# Value function across X
plot(sols.val_func[1,1,1,1,:,6])
plot!(sols.val_func[2,1,1,1,:,6])

# Consumption
plot(sols.c_pol_func[1,1,1,:,:,1]')
plot!(sols.c_pol_func[2,1,1,1,:,1])

# Housing 
plot(sols.H_pol_func[1,1,1,:,:,1]')
plot!(sols.H_pol_func[2,1,1,1,:,1])

# Debt 
plot(sols.D_pol_func[1,1,3,:,:,1]')
plot!(sols.D_pol_func[2,1,1,:,:,10])

# Stock share 
plot(X_grid, sols.α_pol_func[1,1,1,1,:,:]) 

# Stock market entry payment 
plot(X_grid, sols.FC_pol_func[1,1,1,:,:,10]') 

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
consumption_path = mean(consumption, dims = 1)[1:T]
wealth_path = mean(wealth, dims = 1)[1:T]
stock_path = mean(stocks, dims = 1)[1:T]
debt_path = mean(debt, dims = 1)[1:T]
housing_path = mean(housing, dims = 1)[1:T]

plot(age_grid, consumption_path)
plot(age_grid, wealth_path)
plot(age_grid, wealth_path)
plot(age_grid,stock_path)
plot(age_grid,debt_path)
plot(age_grid,housing_path)

histogram(cash_on_hand[:,7])