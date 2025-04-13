###########################################
# Estimates the Lifecycle Portfolio Choice Model of 
# Cocco (2005, RFS) "Portfolio Choice in the Presence of Housing"
###########################################
# Packages 
###########################################
using Parameters, CSV, DelimitedFiles, CSV, Plots,Distributions,LaTeXStrings,Statistics, DataFrames
###########################################
# Put your path to the dataset "estimation_results.csv" below 
indir = "$outdir_parameters" 

# Put the outdirectory below
outdir_images = "$outdir_images" 
outdir_parameters = "$outdir_parameters" 
###########################################
# Functions 
###########################################
# Function which when given an annual rate returns the compounded rate over T periods. 
use("compound.jl") 

# Function which when given a vector of processes and an N for each process, returns a transition matrix and grid 
use("tauchen.jl")

# Function which computes an initial distribution over a given grid 
use("compute_initial_dist.jl")
###########################################
# Parameters
###########################################
cd(indir)
@with_kw struct Model_Parameters @deftype Float64

    # Parameters 

    # Variance Parameters 
    σ_n = 0.019^2
    σ_ι = 0.1674^2
    σ_ω = 0.136^2 # Start with the no-college case 
    σ_p = 0.062^2

    # Correlations between processes 
    κ_ω = 0.00 # Regression coefficient of house price deviations on idiosyncratic labor income
    κ_η = σ_n/σ_p # Proportion of aggregate shock that translates into a log house price shock
    ρ_ϵ_ι = 0.0 # Correlation between aggregate income shocks and stock returns
    φ = 0.748 # Autocorrelation of η

    # One time stock market entry cost 
    F = 1000.0

    # Returns / interest rates 
    R_D = compound(0.04,5) # HELOC interest rate (net, not gross)
    R_F = compound(0.02,5) # Risk-Free rate 
    R_S = 0.1 # Mean Stock Return
    μ = compound(log(R_S + 1.0) - σ_η/2,5)  # Expected return on risky assets

    # Housing parameters
    d = 0.15 # Down-Payment proportion 
    π = 0.244 # Moving shock probability 
    δ = 0.01 # Housing Depreciation
    λ = 0.08 # House-sale cost 
    b = 0.01 * 5 # Real house price growth over 5 years  - matching the way it is presented in the paper 

    # Utility function parameters
    θ = 0.1 # Utility from housing services relative to consumption
    γ = 5.0 # Risk-aversion parameter 
    β = compound(0.96,5) # Discount Rate 

    # Lifecycle Parameters 
    T = 10 # Each T represents five years of life from 25 to 75 
    TR = 9 # The final two time periods represent retirement 

    # Grids

    # Income Process
    η_grid::Array{Float64,1},T_η::Matrix{Float64} = tauchen(σ_η)
    ω_grid::Array{Float64,1},T_ω::Matrix{Float64}  = tauchen(σ_ϵ)

    nω::Int64 = length(ω_grid)
    nη::Int64 = length(η_grid)

    # In the paper, there is no initial level of housing and LW_1 = 0 
    # But what is η_0? He doesn't say... 
    η_0_dist::Array{Float64,1}  = compute_initial_dist(η_grid,σ_η_0)

    # Load the lifecycle component of income: Should be for T years (includes retirement)
    κ::Matrix{Float64} = hcat(CSV.File("life_cycle_income.csv").age,CSV.File("life_cycle_income.csv").deterministic_component)

    # Stock Market Process 
    ι_grid::Array{Float64,1},T_ι::Matrix{Float64} = tauchen(σ_ι) # Only need to do one by one discretization as ρ_ϵ_ι = 0.0 

    nι::Int64 = length(ι_grid)

    # House Price Process 
    p_grid::Array{Float64,1} = κ_η .* η_grid
    
    np = nη

    # States 

    # Consumption
    C_min = 0.0001
    C_max = 1000000 # Maximum consumption allowed

    nc::Int64 = 20 # Number of consumption grid points 

    # Use equally-spaced grids like the paper 
    c_grid::Array{Float64,1} = collect(range(c_min, length = nc, stop = c_max))

    # Housing 
    H_min = 20000.0 # Minimum housing size 
    H_max = 1000000

    nH::Int64 = 50 
    H_grid::Array{Float64,1} = collect(range(H_min, length = nH, stop = H_max))

    # Debt-grid 
    D_min = 0.0
    D_max = 0.8*H_max # Start with some value

    nD::Int64 = 20
    D_grid::Array{Float64,1} = collect(range(D_min, length = nD, stop = D_max))

    # Risky Portfolio share grid 
    α_min = 0.0 
    α_max = 1.0 

    nα::Int64 = 20 
    α_grid::Array{Float64,1} = collect(range(α_min, length = nα, stop = α_max))

    # Cash on hand grid
    X_min = 0.0
    X_max = 1000000 

    nX::Int64 = 10 
    X_grid::Array{Float64,1} = collect(range(X_min, length = nX, stop = X_max))

    # Involuntary-Move 
    Inv_Move = [0 1]

    # Enter the Stock Market for the first time 
    FC = [0 1]

end 

#initialize value function and policy functions
mutable struct Solutions

    val_func::Array{Float64,6} # 6 states, it turns out that the retired's value function still depends on η even after retirement,as it pins down housing.
    pol_func::Array{Float64,6} # For each t and state, there are five choices

end

function build_solutions(para) 

    # Last two fields represent whether the agent has to move and whether they have previously paid the stock market entry fee
    val_func = zeros(Float64,para.T,para.nH,para.nX,para.nη,2,2 ) 

    # Last field represents whether the agent chooses to pay the stock market entry fee
    pol_func = zeros(Float64,para.nc,para.nH,para.nD,para.nα,2 )

    sols = Solutions(val_func,pol_func)

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
    #=
        Computes Cobb-Douglas Flow Utility 
 
        Args
        c (Float64): consumption choice
        H (Float64): Housing choice
        para (Model_Parameters): parameters that define the model, specifically the utility function
    =#

    @unpack γ, θ = para
    return (   ( c^(1-θ) * H^θ )^( 1 - γ )   ) / (1 - γ)
end 

function budget_constraint(X::Float64, H::Float64, para::Model_Parameters)
    @unpack γ, θ = para
    return (   ( c^(1-θ) * H^θ )^( 1 - γ )   ) / (1 - γ)
end 

function compute_bequest_value(V::Array{Float64,5}, para::Model_Parameters)
    # Given a realized state in T+1 and a , computes the bequest agent's bequest utility.    
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

                P = exp(b * (T+1) + p_grid[η_index])
    
                # Agents are forced to sell their house when they die
                    
                W = X - δ * H * P +  (1-λ) *  P * H

                V[H_index, X_index, η_index, :, :,] .+= β * ( W^(1-γ) )/(1-γ)  
            end 
        end 
    end 

    return V
end 

function bilinear_interp(F::Array{Float64, 2}, x1::Vector{Float64}, x2::Vector{Float64})
    #=
    Bilinear interpolation for 2D grid with flat extrapolation

    Args
    F (Array): 2D grid of function values evaluated on grid points
    x1 (Vector): grid points for first dimension - must be evenly spaced
    x2 (Vector): grid points for second dimension - must be evenly spaced

    Returns
    interp (Function): bilinear interpolation function
    =#

    # need to get range object for Interpolations.jl
    x1_grid = range(minimum(x1), maximum(x1), length=length(x1))
    x2_grid = range(minimum(x2), maximum(x2), length=length(x2))

    interp = interpolate(F, BSpline(Linear()))
    extrap = extrapolate(interp, Interpolations.Flat())
    return scale(extrap, x1_grid, x2_grid)
end

function Solve_Problem(para::Model_Parameters, sols::Solutions)
    # Solves the decision problem, outputs results back to the sols structure. 

    @unpack_Model_Parameters para 
    @unpack val_func, pol_func = sols

    V = zeros(T+1, nH, nX, nη, 2, 2) 
    pol = zeros(T+1 , nc, nH, nD, nα, 2)

    # Compute the bequest value of wealth
    V[T+1,:,:,:,:,:] = compute_bequest_value(V[T+1, :, :, :, :, :], para)

    println("Begin solving the model backwards")
    for j in T:-1:TR  # Backward induction
        println("Solving the Retiree's Problem")
        println("Age is ", 75 - 5*j)
        
        # Generate interpolation functions
        interp_functions = Vector{Function}(undef, 5)

        # It seems that my states DO NOT pin down next period's value for my problem...
        # Specifically, ω and r influence next period's X value

        for η_prime in 1: nη
            for ω_prime in 1:nω
                for r_prime in 1:nr # Given some η, ω, r, the fixed entry cost and the moving shock and my choice, I know the states

                    EV[j+1, :, :, η_prime, Inv_Move, IFC] .+= (1+ r_prime)
                    interp[η_prime   ] = bilinear_interp(V[j+1, :, :, η_prime, Inv_Move, IFC], H_grid, X_grid)

        # Loop over Housing States 
        for H_index in 1:nH 
            H = H_grid[H_index]

            # Loop over Cash-on-hand states
            for X_index in 1:nX
                X = X_grid[X_index]

                 # Loop over aggregate income states
                for η_index in 1:nη
                    η =  η_grid[η_index]

                    # Loop over whether the agent was forced to move 
                    for Inv_Move = 0:1

                        # Loop over whether the agent has already paid their stock market entry cost 
                        for IFC = 0:1 
                            candidate_max = -Inf  

                            # Loop over consumption choices 
                            for c_index in 1:nc 
                                c = c_grid[c_index]

                                # Loop over Housing choices 
                                for H_prime_index = 1:nH 
                                    H_prime = H_grid[H_prime_index]

                                    # Loop over Debt choices 
                                    for D_index in 1:nD
                                        D = D_grid[D_index]

                                        # Loop over Risky-share choices
                                        for α_index in 1:nα 
                                            α = α_grid[α_index]

                                            # Loop over enter / not enter choices 
                                            for FC in 0:1 
                                                
                                                if budget_constraint(c, H_prime, D, α) < budget 
                                                    val = flow_utility_func(c,H)
                                                    # Loop over random variables 
                                                    for η_prime in 1: nη
                                                        for ω_prime in 1:nω
                                                            for r_prime in 1:nr 

                                                                val += 




                        






                # 
                candidate_max = -Inf                     
                Y = exp(ϵ + ζ + κ[j,2])  # Construct the income process

                # Use that ap(a) is a weakly increasing function. 
                start_index = 1 
                for index_a in 1:na
                    a = a_grid[index_a]
                    coh =  (1 + r) * a + Y
                    for index_ap in start_index:na 
                        ap = a_grid[index_ap]
                        c = coh - ap 

                        if c > 0  # Feasibility check
                            val = (c^(1 - σ)) / (1 - σ)

                            for e_prime in 1:nϵ
                                for z_prime in 1:nζ
                                    val += β * T_ϵ[e,e_prime] * T_ζ[z,z_prime] * V_next[index_ap, e_prime, z_prime, j + 1]
                                end
                            end

                            if val > candidate_max  # Check for max
                                candidate_max = val
                                pol_next[index_a, e, z, j] = ap
                                start_index = index_ap
                                V_next[index_a, e, z, j] = candidate_max
                            end   
                        end
                    end 
                end 
            end 
        end
    end

 
    sols.val_func .= V_next[:,:,:,1:N]
    sols.pol_func .= pol_next[:,:,:,1:N]
end