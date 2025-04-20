###########################################
# Solves the Lifecycle Portfolio Choice Model of 
# Cocco (2005, RFS) "Portfolio Choice in the Presence of Housing"
###########################################
# Packages 
###########################################
using Parameters,DelimitedFiles, CSV, Plots, Distributions,LaTeXStrings, Statistics, DataFrames, LinearAlgebra, Optim, Interpolations, Base.Threads, Roots
###########################################
indir = "C:/Users/zacha/Documents/Research Ideas/Housing and Portfolio Choice/Replication"

indir_parameters = "C:/Users/zacha/Documents/Research Ideas/Housing and Portfolio Choice/Replication/parameters"
cd(indir)
###########################################
# Parameters
###########################################
cd(indir_parameters)

@with_kw struct Model_Parameters
    # Normalize all dollar variables by Z
    Z = 1

    # Number of gridpoints for random variables 
    g = 3

    # Gridpoints for cash on hand 

    # Parameters with type annotations
    # Variance Parameters 
    σ_η::Float64 = 0.019^2  # Variance of persistent component of earnings
    σ_ι::Float64 = 0.1674^2 # Variance of stock market innovation
    σ_ω::Float64 = 0.136^2  # Start with the no-college case 
    σ_p::Float64 = 0.062^2  # Variance of house prices

    # Correlations between processes 
    κ_ω::Float64 = 0.00     # Correlation between house prices and transitory component
    κ_η::Float64 = σ_η/σ_p  # Regression coefficient of cyclical fluctuations in house prices on persistent component (correlation is 1)
    ρ_ϵ_ι::Float64 = 0.0    # Correlation between persistent component of income and the stock market
    φ::Float64 = 0.748      # Autocorrelation in the persistent component 

    # One time stock market entry cost 
    F::Float64 = 1000.0 / Z

    # Returns / interest rates 
    R_D::Float64 =  compound(1 + 0.04, 5)                  # Mortgage interest rate 
    R_F::Float64 =  compound(1 + 0.02, 5)                  # Risk-free rate
    R_S::Float64 = 0.1                                     # Expected return on stocks 
    μ::Float64 = log(compound(1 + 0.1, 5) - σ_η/2)        # Expected log-return on stocks

    # Housing parameters
    d::Float64 = 0.15                               # Down-Payment proportion 
    π::Float64 = compound(1 + 0.03, 5) - 1          # Moving shock probability 
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
    η_grid::Vector{Float64} = rouwenhorst(σ_η, φ, 3)[1] .- log(Z) # Normalize labor income by Z so need to subtract log(Z) from the log-process. 
    T_η::Matrix{Float64} = rouwenhorst(σ_η, φ, 3)[2]
    nη::Int64 = length(η_grid)

    # Transitory Earnings
    ω_grid::Vector{Float64} = rouwenhorst(σ_ω, 0.0, 3)[1] .- log(Z)
    T_ω::Matrix{Float64} = rouwenhorst(σ_ω, 0.0, 3)[2]
    nω::Int64 = length(ω_grid)

    # In the paper, there is no initial level of housing and LW_1 = 0 
    # But what is η_0? He doesn't say... 
    η_0::Float64 = 0.0 # For now, assume that persistent earnings starts at the unconditional mean

    # Deterministic earnings
    # For now, use what my own estimates from the PSID. 
    κ::Matrix{Any} = hcat(CSV.File("life_cycle_income_1.csv").age_group, CSV.File("life_cycle_income_1.csv").age_dummies .- - log(Z)) 

    # Stock market grids
    ι_grid::Vector{Float64} = rouwenhorst(σ_ι, 0.0, 3)[1] .- log(Z)
    T_ι::Matrix{Float64} = rouwenhorst(σ_ι, 0.0, 3)[2]
    nι::Int64 = length(ι_grid)

    # Housing grids
    p_grid::Vector{Float64} = η_grid ./κ_η
    P_bar::Float64 = 30500.0 * 140.3/(Z * 41.8) # For now, use an estimate of house prices in 1972 and inflate using the ratio of CPI in 1992 to CPI in 1972
                                          # 1972 house prices taken from https://www.huduser.gov/periodicals/ushmc/winter2001/histdat08.htm 
                                          # CPI taken from https://fred.stlouisfed.org/series/CPIAUCSL 
    np::Int64 = nη

    # State / Choice Grids 
    X_min::Float64 = 0.01
    X_max::Float64 = 300000.0/ Z
    nX::Int64 = 40
    X_grid::Vector{Float64} = collect(range(X_min, length = nX, stop = X_max))

    c_min::Float64 = 0.0001
    c_max::Float64 = 3*X_max
    nc::Int64 = 100
    c_grid::Vector{Float64} = collect(range(c_min, length = nc, stop = c_max))

    H_min::Float64 = 0.2
    H_max::Float64 = 5.0
    nH::Int64 = 10
    H_grid::Vector{Float64} = collect(range(H_min, length = nH, stop = H_max))

    D_min::Float64 = 0.0
    D_max::Float64 = 3*X_max
    nD::Int64 = 20
    D_grid::Vector{Float64} = collect(range(D_min, length = nD, stop = D_max))

    α_min::Float64 = 0.0
    α_max::Float64 = 1.0
    nα::Int64 = 3
    α_grid::Vector{Float64} = collect(range(α_min, length = nα, stop = α_max))

    Inv_Move_grid::Vector{Int64} = [0, 1]

    FC_grid::Vector{Int64}       = [0, 1]

    IFC_grid::Vector{Int64}      = [0, 1]
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

    # Last two fields represent whether the agent has to move and whether they have previously paid the stock market entry fee
    val_func = zeros(Float64,para.T + 1,para.nH,para.nX,para.nη,2,2 ) 
    c_pol_func  = zeros(Float64,para.T,para.nH,para.nX,para.nη,2,2 ) 
    H_pol_func  = zeros(Float64,para.T,para.nH,para.nX,para.nη,2,2 ) 
    D_pol_func  = zeros(Float64,para.T,para.nH,para.nX,para.nη,2,2 ) 
    FC_pol_func = zeros(Float64,para.T,para.nH,para.nX,para.nη,2,2 ) 
    α_pol_func  = zeros(Float64,para.T,para.nH,para.nX,para.nη,2,2 ) 

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

#########################################################
# Functions 
#########################################################
function flow_utility_func(c::Float64, H::Float64, para::Model_Parameters)
    @unpack γ, θ = para

    return 10^20 *(    ( c^(1-θ) * H^θ )^( 1 - γ )   ) / (1 - γ)
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

    # Loop over aggregate income states
    for η_index in 1:nη
        η =  η_grid[η_index]

        P = P_bar * exp(b * (T) + p_grid[η_index])

        # Loop over Housing States 
        for H_index in 1:nH 
            H = H_grid[H_index]
        
            # Loop over Cash-on-hand states
            for X_index in 1:nX
                X = X_grid[X_index]
    
                # Agents are forced to sell their house when they die
                    
                W = X - δ * H * P +  (1-λ) *  P * H

                V[H_index, X_index, η_index, :, :] .+= 10^20 * ( W^(1-γ) )/(1-γ)  
            end 
        end 
    end 

    return V
end 

# Creates a linear interpolation function using a mapping from grid x1 to outcome F. 
# Cubic-spline interpolation within-grid
# Allows for extrapolation outside the grid (Flat extrapolation)
function linear_interp(F::Array{Float64, 1}, x1::Vector{Float64})
    x1_grid = range(minimum(x1), maximum(x1), length=length(x1))

    interp = interpolate(F, BSpline(Cubic()))
    extrap = extrapolate(interp, Interpolations.Flat())
    return  extrap
end

#########################################
# Solve the Model!
#########################################
para, sols = Initialize_Model()
@time Solve_Retiree_Problem(para, sols)
@time Solve_Worker_Problem(para, sols)

#########################################
# Checks
#########################################
@unpack_Model_Parameters para 
@unpack val_func, c_pol_func, H_pol_func, D_pol_func, FC_pol_func, α_pol_func = sols

plot(H_grid, sols.val_func[9,:,90,1,1,1])
plot(X_grid, sols.val_func[9,2,:,1,1,1])
plot(X_grid,sols.D_pol_func[10,5,:,1,1,1])
plot(sols.c_pol_func[9,2,:,1,1,1])
plot!(sols.c_pol_func[10,2,:,1,2,1])
plot(sols.H_pol_func[10,10,:,1,1,2])
plot!(sols.H_pol_func[10,10,:,1,2,2])
plot(sols.α_pol_func[9,10,:,1,1,1]) 

# CHeck constraints
D = sols.D_pol_func[9,2,2,1,1,1]
c = sols.c_pol_func[9,2,2,1,1,1]
FC = Int64(sols.FC_pol_func[9,2,2,1,1,1])
α = sols.α_pol_func[9,2,2,1,1,1]
H_prime = sols.H_pol_func[9,2,2,1,1,1]

# Compute X_prime
j = 9
X = X_grid[2]
H = H_grid[2]
Inv_Move = 1
η_index = 1
η_prime = η_grid[η_index]
P = P_bar * exp(b * (j-1) + p_grid[η_index])

ι_index = 1
ι_prime = ι_grid[ι_index]
H_prime = H_grid[2]

S_and_B  = budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)

find_zero(c -> budget_constraint(X, H, P, Inv_Move,c, H_prime, D, FC, para), 100.0)

# Check if an optimizer helps 
function value_given_c(c, H_prime, D, α, state, para)
    # Compute flow utility
    u = flow_utility_func(c, H_prime, para)

    # Compute asset tomorrow (returns, debt, etc.)
    A_next = compute_next_assets(c, H, D, α, state, para)

    # Interpolate or compute expected continuation value
    EV = expected_value_next(A_next, ...)

    return -(u + para.β * EV)  # Negative since GSS finds minimum
end
R_prime = exp(ι_prime + μ)
Y_Prime = κ[j, 2]

S = α * S_and_B
B = (1-α) * S_and_B
# Compute next period's liquid wealth
X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime

P =  P_bar * exp(b * (10-1) + p_grid[η_index])
X = X_grid[1]
S_and_B  = budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)

# Need to change: 
# Potentially order of where the debt choice / debt constraint is checked 
# Where the if-else conditions are will also change because of pruning from from IFC = 1 then FC = 0 
# The Price law of motion needs to be made multiplicative 
# Need to add β to value function iteration step 
#Solve_Worker_Problem(para, sols)

#= 
# Instead of grid-searching, optimize. 
compute_objective(c::Float64, H_prime::Float64, D::Float64, α::Float64, FC::Int64,
                  H::Float64, X::Float64, η::Float64, Inv_Move::Int64, IFC::Int64, para, sols)

    # Impose consumption must be non-negative
    if c <= 0
        return -Inf
    end

    # Impose housing must exceed the minimum house size must be non-negative
    if H_prime < H_min
        return -Inf
    end

    # Impose the stock share must be non-negative
    if α < 0
        return -Inf
    end

    S_and_B  =  budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)

    # Skip if implied stock and bond spending must be negative
    if S_and_B <= 0
        return -Inf
    end

    # Skip if debt exceeds the collateral constraint. 
    if debt_constraint(D, H, P, para) <= 0
        return -Inf
    end  

    # If the person has not paid the entry cost and does not pay the entry cost today 
    # they must invest 0 in stocks. 
    if FC == 0 && IFC == 0 
        if α > 0 
            return -Inf 
        end 


        # Compute Stock and Bond Positions 
        S = α * S_and_B 
        B = (1-α) * S_and_B 

        val = flow_utility_func(c, H, para)

        # Find the continuation value 
        # Loop over random variables 
        for η_prime_index in 1:nη
        η_prime = η_grid[η_prime_index]

            for ω_prime_index in 1:nω
                ω_prime = ω_grid[ω_prime_index]

                for ι_prime_index in 1:nι 
                    ι_prime = ι_grid[ι_prime_index]

                    R_prime = exp(ι_prime + μ)
                    Y_Prime = κ[T, 2] * exp(η_prime + ω_prime)

                    # Compute next period's liquid wealth
                    X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime

                    val += (1-π)  * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *                                # Not forced to Move 
                            interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (0 * 2) + FC + 1](X_prime) +
                            π     * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *                                # Forced to move
                            interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (1 * 2) + FC + 1](X_prime)
                end 
            end 
        end 
                                                

    # If !(IFC == 0 && FC == 0)
    else 
        # Loop over Risky-share choices
        for α_index in 1:nα 
            α = α_grid[α_index]

            # Compute Stock and Bond Positions 
            S = α * S_and_B 
            B = (1-α) * S_and_B 

            val = flow_utility_func(c, H, para)

            # Find the continuation value 
            # Loop over random variables 
            for η_prime_index in 1:nη
                η_prime = η_grid[η_prime_index]

                for ω_prime_index in 1:nω
                    ω_prime = ω_grid[ω_prime_index]

                    for ι_prime_index in 1:nι  
                        ι_prime = ι_grid[ι_prime_index]

                        R_prime = exp(ι_prime + μ)
                        Y_Prime = exp(η_prime + ω_prime + κ[T, 2])

                        # Compute next period's liquid wealth
                        X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime

                                                                
                        val += (1-π) * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *
                                interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (0 * 2) + FC + 1](X_prime) +
                                π * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *
                                interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (1 * 2) + FC + 1](X_prime)
                    end 
                end 
            end 
        end 
    end 

    return val 
end 

=# 