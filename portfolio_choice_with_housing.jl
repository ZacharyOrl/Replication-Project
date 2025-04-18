###########################################
# Solves the Lifecycle Portfolio Choice Model of 
# Cocco (2005, RFS) "Portfolio Choice in the Presence of Housing"
###########################################
# Packages 
###########################################
using Parameters, CSV, DelimitedFiles, CSV, Plots,Distributions,LaTeXStrings, Statistics, DataFrames, LinearAlgebra, Optim, Interpolations
###########################################
indir = "C:/Users/zacha/Documents/Research Ideas/Housing and Portfolio Choice/Replication"

indir_parameters = "C:/Users/zacha/Documents/Research Ideas/Housing and Portfolio Choice/Replication/parameters"
cd(indir)
###########################################
# Functions 
###########################################
# Function which when given an annual rate returns the compounded rate over T periods. 
include("compound.jl") 

# Function which when given a vector of processes and an N for each process, returns a transition matrix and grid 
include("rouwenhorst.jl")

# Function which computes an initial distribution over a given grid 
include("compute_initial_dist.jl")
###########################################
# Parameters
###########################################
cd(indir_parameters)
using Parameters

@with_kw struct Model_Parameters
    # Normalize all dollar variables by Z
    Z = 10000

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
    R_D::Float64 = compound(0.04, 5)                  # Mortgage interest rate 
    R_F::Float64 = compound(0.02, 5)                  # Risk-free rate
    R_S::Float64 = 0.1                                # Expected return on stocks 
    μ::Float64 = compound(log(R_S + 1.0) - σ_η/2, 5)  # Expected log-return on stocks

    # Housing parameters
    d::Float64 = 0.15              # Down-Payment proportion 
    π::Float64 = 0.244             # Moving shock probability 
    δ::Float64 = 0.01              # Housing Depreciation
    λ::Float64 = 0.08              # House-sale cost 
    b::Float64 = compound(0.01, 5) # Real house price growth over 5 years  - matching the way it is presented in the paper 

    # Utility function parameters
    θ::Float64 = 0.1              # Utility from housing services relative to consumption
    γ::Float64 = 5.0              # Risk-aversion parameter 
    β::Float64 = compound(0.96,5) # Discount Rate 

    # Lifecycle Parameters 
    T::Int64 = 10 # Each T represents five years of life from 25 to 75
    TR::Int64 = 9 # The final two time periods represent retirement 

    # Grids & Transition Matrices
    # Persistent Earnings
    η_grid::Vector{Float64} = rouwenhorst(σ_η, φ, 3)[1] .- log(Z) # Normalize labor income by 1000 so need to subtract log(1000) from log-process. 
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
    p_grid::Vector{Float64} = κ_η .* η_grid 
    P_bar::Float64 = 30500.0 * 140.3/(Z * 41.8) # For now, use an estimate of house prices in 1972 and inflate using the ratio of CPI in 1992 to CPI in 1972
                                          # 1972 house prices taken from https://www.huduser.gov/periodicals/ushmc/winter2001/histdat08.htm 
                                          # CPI taken from https://fred.stlouisfed.org/series/CPIAUCSL 
    np::Int64 = nη

    # State / Choice Grids 
    c_min::Float64 = 0.0001
    c_max::Float64 = 1000000.0/Z
    nc::Int64 = 20
    c_grid::Vector{Float64} = collect(range(c_min, length = nc, stop = c_max))

    H_min::Float64 = 0.2
    H_max::Float64 = 10.0
    nH::Int64 = 5
    H_grid::Vector{Float64} = collect(range(H_min, length = nH, stop = H_max))

    D_min::Float64 = 0.0
    D_max::Float64 = 0.8*c_max / Z
    nD::Int64 = 20
    D_grid::Vector{Float64} = collect(range(D_min, length = nD, stop = D_max))

    α_min::Float64 = 0.0
    α_max::Float64 = 1.0
    nα::Int64 = 20
    α_grid::Vector{Float64} = collect(range(α_min, length = nα, stop = α_max))

    X_min::Float64 = 0.0
    X_max::Float64 = 1000000.0/ Z
    nX::Int64 = 20
    X_grid::Vector{Float64} = collect(range(X_min, length = nX, stop = X_max))

    Inv_Move_grid::Vector{Int64} = [0, 1]
    FC_grid::Vector{Int64} = [0, 1]
    IFC_grid::Vector{Int64} = [0, 1]
end
#initialize value function and policy functions
mutable struct Solutions

    val_func::Array{Float64,6} # 6 states, it turns out that the retired's value function still depends on η even after retirement,as it pins down housing.
    # For each t and state, there are five choices
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

#########################################################
# Functions 
#########################################################
function flow_utility_func(c::Float64, H::Float64, para::Model_Parameters)
    @unpack γ, θ = para

    return (   ( c^(1-θ) * H^θ )^( 1 - γ )   ) / (1 - γ)
end 

# Takes as input all states and choices necessary to pin down the budget constraint
# and outputs the sum of stocks and bonds (LHS of the budget constraint in Cocco)
function budget_constraint(X::Float64, H::Float64, P::Float64, Inv_Move::Int64, c::Float64, 
                           H_prime::Float64, D::Float64, FC::Int64, para::Model_Parameters)
    @unpack δ, F, λ = para

    # If there is no house trade
    if Inv_Move == 1
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

    # Loop over Housing States 
    for H_index in 1:nH 
        H = H_grid[H_index]
    
        # Loop over Cash-on-hand states
        for X_index in 1:nX
            X = X_grid[X_index]
    
            # Loop over aggregate income states
            for η_index in 1:nη
                η =  η_grid[η_index]

                P = P_bar + exp(b * (T) + p_grid[η_index])
    
                # Agents are forced to sell their house when they die
                    
                W = X - δ * H * P +  (1-λ) *  P * H

                V[H_index, X_index, η_index, :, :] .+= ( W^(1-γ) )/(1-γ)  
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

# Solves the decision problem, outputs results back to the sols structure. 
function Solve_Retiree_Problem(para::Model_Parameters, sols::Solutions)
    @unpack_Model_Parameters para 
    @unpack val_func, c_pol_func, H_pol_func, D_pol_func, FC_pol_func, α_pol_func = sols
    println("Solving the Retiree's Problem")

    # Compute the bequest value of wealth
    val_func[T+1,:,:,:,:,:] = compute_bequest_value(val_func[T+1, :, :, :, :, :], para)

    println("Begin solving the model backwards")
    for j in T:-1:TR  # Backward induction

        println("Age is ", 25 + 5*j)
        
        # Generate interpolation functions for cash-on hand given each possible combination of the other states tomorrow 
        interp_functions = Vector{Any}(undef, nH * nη * 2 * 2)
        for H_index in 1:nH
            for η_prime_index in 1:nη
                for Inv_Move in 0:1
                    for FC in 0:1
                        index = ((H_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (Inv_Move * 2) + FC + 1
                        interp_functions[index] = linear_interp(val_func[j+1, H_index, :, η_prime_index, Inv_Move + 1, FC + 1], X_grid)
                    end 
                end 
            end 
        end 

        # Loop over Housing States 
        for H_index in 1:nH 
            H = H_grid[H_index]

            # Loop over Cash-on-hand states
            for X_index in 1:nX
                X = X_grid[X_index]

                # Loop over aggregate income states
                for η_index in 1:nη
                    η = η_grid[η_index]
                    P = P_bar + exp(b * (j-1) + p_grid[η_index])

                    # Loop over whether the agent was forced to move 
                    for Inv_Move_index in 1:2
                        Inv_Move = Inv_Move_grid[Inv_Move_index]

                        # Loop over whether the agent has already paid their stock market entry cost 
                        for IFC_index in 1:2
                            IFC = IFC_grid[IFC_index]
                            candidate_max = -Inf  

                            # Loop over consumption choices 
                            for c_index in 1:nc 
                                c = c_grid[c_index]

                                # Loop over Housing choices 
                                for H_prime_index in 1:nH 
                                    H_prime = H_grid[H_prime_index]

                                    # Loop over Debt choices 
                                    for D_index in 1:nD
                                        D = D_grid[D_index]

                                        # Loop over enter/not enter choices 
                                        for FC_index in 1:2
                                            FC = FC_grid[FC_index]

                                            S_and_B  =  budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)

                                            # Skip if implied stock and bond spending must be negative
                                            if S_and_B <= 0
                                                continue
                                            end

                                            # Skip if debt exceeds the collateral constraint. 
                                            if debt_constraint(D, H, P, para) <= 0
                                                continue 
                                            end  

                                            # If the person has not paid the entry cost and does not pay the entry cost today 
                                            # they must invest 0 in stocks. 
                                            if FC == 0 && IFC == 0 
                                                α_index = 1 
                                                α = α_grid[α_index]

                                                # Compute Stock and Bond Positions 
                                                S = α * S_and_B 
                                                B = (1-α) * S_and_B 

                                                val = flow_utility_func(c, H, para)
                                                # Find the continuation value 
                                                # Loop over random variables 
                                                for η_prime_index in 1:nη
                                                    η_prime = η_grid[η_prime_index]

                                                    for ι_prime_index in 1:nι 
                                                        ι_prime = ι_grid[ι_prime_index]

                                                        R_prime = exp(ι_prime + μ)
                                                        Y_Prime = exp(κ[j, 2])
                                                        P_Prime = P_bar + exp(b * (j-1) + p_grid[η_prime_index])

                                                        # Compute next period's liquid wealth
                                                        X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime

                                                        val += (1-π) * T_η[η_index, η_prime_index]  * T_ι[1, ι_prime_index] *                                # Not forced to Move 
                                                                interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (0 * 2) + FC + 1](X_prime) +
                                                                π     * T_η[η_index, η_prime_index] * T_ι[1, ι_prime_index] *                                # Forced to move
                                                                interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (1 * 2) + FC + 1](X_prime)

                                                    end 
                                                end 
                                                
                                                # Update value function
                                                if val > candidate_max 
                                                    val_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]    = val
                                                    
                                                    c_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = c 
                                                    H_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = H_prime
                                                    D_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = D
                                                    FC_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1] = FC
                                                    α_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = α
                                                    
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


                                                        for ι_prime_index in 1:nι  
                                                            ι_prime = ι_grid[ι_prime_index]

                                                            R_prime = exp(ι_prime + μ)
                                                            Y_Prime = exp(κ[j, 2])
                                                            P_Prime = P_bar + exp(b * (j-1) + p_grid[η_prime_index])

                                                            # Compute next period's liquid wealth
                                                            X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime

                                                                
                                                            val += (1-π) * T_η[η_index, η_prime_index]  * T_ι[1, ι_prime_index] *
                                                                interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (0 * 2) + FC + 1](X_prime) +
                                                                    π * T_η[η_index, η_prime_index]   * T_ι[1, ι_prime_index] *
                                                                interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (1 * 2) + FC + 1](X_prime)
                                                         
                                                        end 
                                                    end 
                                                    # Update value function
                                                    if val > candidate_max 
                                                        val_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]    = val

                                                        c_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = c 
                                                        H_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = H_prime
                                                        D_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = D
                                                        FC_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1] = FC
                                                        α_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = α
                                                    end 
                                                end 
                                            end 
                                        end 
                                    end 
                                end 
                            end 
                        end 
                    end 
                end 
            end 
        end 
    end 
end

function Solve_Worker_Problem(para::Model_Parameters, sols::Solutions)
    @unpack_Model_Parameters para 
    @unpack val_func, c_pol_func, H_pol_func, D_pol_func, FC_pol_func, α_pol_func = sols
    println("Solving the Worker's Problem")
    println("Begin solving the model backwards")

    for j in TR -1:-1:1  # Backward induction
        println("Age is ", 25 + 5*j)

        # Generate interpolation functions for cash-on hand given each possible combination of the other states tomorrow 
        interp_functions = Vector{Any}(undef, nH * nη * 2 * 2)
        for H_index in 1:nH
            for η_prime_index in 1:nη
                for Inv_Move in 0:1
                    for FC in 0:1
                        index = ((H_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (Inv_Move * 2) + FC + 1
                        interp_functions[index] = linear_interp(val_func[j+1, H_index, :, η_prime_index, Inv_Move + 1, FC + 1], X_grid)
                    end 
                end 
            end 
        end 
        
        # Loop over Housing States 
        for H_index in 1:nH 
            H = H_grid[H_index]

            # Loop over Cash-on-hand states
            for X_index in 1:nX
                X = X_grid[X_index]

                # Loop over aggregate income states
                for η_index in 1:nη
                    η = η_grid[η_index]
                    P = P_bar + exp(b * (j-1) + p_grid[η_index])

                    # Loop over whether the agent was forced to move 
                    for Inv_Move_index in 1:2
                        Inv_Move = Inv_Move_grid[Inv_Move_index]

                        # Loop over whether the agent has already paid their stock market entry cost 
                        for IFC_index in 1:2
                            IFC = IFC_grid[IFC_index]
                            candidate_max = -Inf  

                            # Loop over consumption choices 
                            for c_index in 1:nc 
                                c = c_grid[c_index]

                                # Loop over Housing choices 
                                for H_prime_index in 1:nH 
                                    H_prime = H_grid[H_prime_index]

                                    # Loop over Debt choices 
                                    for D_index in 1:nD
                                        D = D_grid[D_index]

                                        # Loop over enter/not enter choices 
                                        for FC_index in 1:2
                                            FC = FC_grid[FC_index]

                                            S_and_B  =  budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)

                                            # Skip if implied stock and bond spending must be negative
                                            if S_and_B <= 0
                                                continue
                                            end

                                            # Skip if debt exceeds the collateral constraint. 
                                            if debt_constraint(D, H, P, para) <= 0
                                                continue 
                                            end  

                                            # If the person has not paid the entry cost and does not pay the entry cost today 
                                            # they must invest 0 in stocks. 
                                            if FC == 0 && IFC == 0 
                                                α_index = 1 
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
                                                            Y_Prime = exp(η_prime + ω_prime + κ[j, 2])
                                                            P_Prime = P_bar + exp(b * (j-1) + p_grid[η_prime_index])

                                                            # Compute next period's liquid wealth
                                                            X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime

                                                            val += (1-π) * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *                                # Not forced to Move 
                                                                   interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (0 * 2) + FC + 1](X_prime) +
                                                                   π     * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *                                # Forced to move
                                                                   interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (1 * 2) + FC + 1](X_prime)
                                                        end 
                                                    end 
                                                end 
                                                
                                                # Update value function
                                                if val > candidate_max 
                                                    val_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]    = val
                                                    
                                                    c_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = c 
                                                    H_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = H_prime
                                                    D_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = D
                                                    FC_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1] = FC
                                                    α_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = α
                                                    
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
                                                                Y_Prime = exp(η_prime + ω_prime + κ[j, 2])
                                                                P_Prime = P_bar + exp(b * (j-1) + p_grid[η_prime_index])

                                                                # Compute next period's liquid wealth
                                                                X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime

                                                                
                                                                val += (1-π) * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *
                                                                    interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (0 * 2) + FC + 1](X_prime) +
                                                                     π * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *
                                                                    interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (1 * 2) + FC + 1](X_prime)
                                                            end 
                                                        end 
                                                    end 
                                                    # Update value function
                                                    if val > candidate_max 
                                                        val_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]    = val

                                                        c_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = c 
                                                        H_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = H_prime
                                                        D_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = D
                                                        FC_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1] = FC
                                                        α_pol_func[j, H_index, X_index, η_index, Inv_Move + 1, IFC + 1]  = α
                                                    end 
                                                end 
                                            end 
                                        end 
                                    end 
                                end 
                            end 
                        end 
                    end 
                end 
            end 
        end 
    end 
end

para, sols = Initialize_Model()
Solve_Retiree_Problem(para, sols)
Solve_Worker_Problem(para, sols)

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
                    Y_Prime = exp(η_prime + ω_prime + κ[T, 2])
                    P_Prime = P_bar + exp(b * (j-1) + p_grid[η_prime_index])

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
                        P_Prime = P_bar + exp(b * (j-1) + p_grid[η_prime_index])

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