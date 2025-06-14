#############################################################################################

# This file holds a series of auxilary functions for use in solving and simulating the model: 
# Function list: 
# 1: The instantaneous utility function
# 2: The budget constraint 
# 3: The budget constraint, conditional on not moving 
# 4: The budget constraint, conditional on moving 
# 5: The value of bequest 
# 6: Bilinear Interpolation function 
# 7: Bilinear Interpolation function, applied to the policy functions and allowing for linear extrapolation. 
# 8: The expected value of future labor earnings in each state. 
# 9: The time series of shocks for each age cohort ( age is as of 1989 as per Cocco)
# 10: Each agent has a full simulated lifecycle in simulations. I pull only a single "age" from each lifecycle according to Cocco's weights.
# 11: Compounds an annual rate over T periods. 
# 12: Computes the stationary distribution of a Markov chain. 
#############################################################################################
# 1.
#############################################################################################

# This function takes the choice of consumption and housing today and returns the flow utility value. 
function flow_utility_func(c::Float64, Own_prime::Int64, s::Float64, para::Model_Parameters)
    @unpack γ, θ = para

    if Own_prime == 0
       return  (    ( c^(1-θ) )^( 1 - γ )   ) / (1 - γ) 
    else 
        return  (    ( c^(1-θ) * s^θ )^( 1 - γ )   ) / (1 - γ)
    end 
end 

#############################################################################################
# 2.
#############################################################################################
# If Inv_Move == 0 & Own == Own_prime, then we use the no-move budget constraint 
function no_move_budget_constraint(X::Float64, Own::Int64, P::Float64, c::Float64, Own_prime::Int64, 
    M::Float64, M_prime::Float64,FC::Int64, F::Float64, para::Model_Parameters)

    @unpack δ, λ, d, r_p = para
    D = M_prime - M # Net borrowing

    if Own_prime == 0.0 # Renter case 
        S_and_B = X - c - FC*F - δ * P * Own - r_p * P + D 
    else # Owner case: 
        S_and_B = X - c - FC*F - δ * P * Own + D
    end 

    return S_and_B
end 

#############################################################################################
# 4.
#############################################################################################
# If Inv_Move != 0 and/or Own != Own_prime, then we use the move budget constraint 
function move_budget_constraint(X::Float64, Own::Int64, P::Float64, c::Float64, Own_prime::Int64, M::Float64, M_prime::Float64, FC::Int64, F::Float64, para::Model_Parameters)
    @unpack δ, λ, d, r_p = para
    D = M_prime - M # Net borrowing

    if Own_prime == 0 # Renter case 
        S_and_B = X - c - FC*F - δ * P * Own + (1-λ) * P * Own - r_p * P + D
    else # Owner case 
        S_and_B = X - c - FC*F - δ * P * Own + D + (1-λ)* P * Own - P * Own_prime * (1+λ)
    end 

    return S_and_B
end 
#############################################################################################
# 5.
#############################################################################################
# Computes the value of the bequest in T+1 
function compute_bequest_value(V::Array{Float64,6}, para::Model_Parameters)
    @unpack_Model_Parameters para 

    # Loop over Cash-on-hand states
    for X_index in 1:nX
        X = X_grid[X_index]

        for M_index in 1:nM
            M = M_grid[M_index]

            # Loop over Ownership States 
            for Own_index in 1:2
                Own = Own_grid[Own_index]
                
                # Loop over aggregate income states
                for η_index in 1:nη
                    η =  η_grid[η_index]

                    P = P_bar * exp(b * (T+1) + p_grid[η_index])
        
                    # Agents are forced to sell their house when they die and repay their mortgage. 
                    W = X - δ * P +  (1-λ) * P *  Own - M 
                    
                    # Account for agents essentially taking on impractical levels of debt that could leave them endowing nothing. 
                    if W < 0 
                        V[:, :, η_index, Own_index, M_index, X_index] .+= pun 
                    else

                    V[:, :, η_index, Own_index, M_index, X_index] .+= ( W^(1-γ) )/(1-γ)  
                    end 
                end 
            end
        end 
    end 

    return V
end 

#############################################################################################
# 6.
#############################################################################################

function linear_interp(F::Array{Float64, 1}, x1::Vector{Float64})
    #= linear interpolation for 1D grid - no extrapolation
    Arguments:  F (Array): 1D grid of function values evaluated on grid points
                x1 (Vector): grid points for first dimension - must be evenly spaced
    Returns:    interp (Function): linear interpolation function =#
    x1_grid = range(minimum(x1), maximum(x1), length=length(x1))

    interp = interpolate(F, BSpline(Linear()))
    return Interpolations.scale(interp, x1_grid)
end

#############################################################################################
# 7. Interpolate the policy functions after solving the Model
# Allows for extrapolation as some rounding is necessary
# (due to moving and stock market entry being discrete).
#############################################################################################
function interpolate_policy_funcs(sols::Solutions,para::Model_Parameters)
    @unpack_Model_Parameters para 
    @unpack val_func,c_pol_func, M_pol_func, FC_pol_func, α_pol_func, Move_pol_func, Own_pol_func = sols

       # Generate interpolation functions for cash-on hand given each possible combination of the other states
       c_interp_functions = Array{Any}(undef, 2 * 2 * nη *  2 * nM,T)
       M_interp_functions = Array{Any}(undef, 2 * 2 * nη *  2 * nM,T) 
       FC_interp_functions = Array{Any}(undef, 2 * 2 * nη * 2 * nM,T) 
       α_interp_functions = Array{Any}(undef, 2 * 2 * nη *  2 * nM,T) 
       Move_interp_functions = Array{Any}(undef, 2 * 2 * nη *  2 * nM,T) 
       Own_interp_functions = Array{Any}(undef, 2 * 2 * nη *  2 * nM,T) 

        for n = 1:T
            for Inv_Move_index in 1:2
                for IFC_index in 1:2
                    for η_index in 1:nη
                        for Own_index in 1:2
                            for M_index in 1:nM

                                # Compute linear index 
                                index = lin[Inv_Move_index, IFC_index, η_index, Own_index, M_index]
                                # Create interpolated policy functions
                                c_interp_functions[index,n]     = extrapolate(linear_interp(c_pol_func[Inv_Move_index, IFC_index, η_index, Own_index, M_index,:, n], X_grid), Line())
                                M_interp_functions[index,n]     = extrapolate(linear_interp(M_pol_func[Inv_Move_index, IFC_index, η_index, Own_index, M_index,:, n], X_grid), Line())
                                FC_interp_functions[index,n]    = extrapolate(linear_interp(FC_pol_func[Inv_Move_index, IFC_index, η_index, Own_index, M_index,:, n],X_grid), Line())
                                α_interp_functions[index,n]     = extrapolate(linear_interp(α_pol_func[Inv_Move_index, IFC_index, η_index, Own_index, M_index,:, n], X_grid), Line())
                                Move_interp_functions[index,n]  = extrapolate(linear_interp(Move_pol_func[Inv_Move_index, IFC_index, η_index, Own_index, M_index,:, n], X_grid), Line())
                                Own_interp_functions[index,n]   = extrapolate(linear_interp(Own_pol_func[Inv_Move_index, IFC_index, η_index, Own_index, M_index,:, n], X_grid), Line())
                            end 
                        end
                    end
                end
            end
        end

    return c_interp_functions, M_interp_functions, FC_interp_functions, α_interp_functions, Move_interp_functions, Own_interp_functions
end 

#############################################################################################
# 8. Compute future expected earnings for each aggregate state and age . 
# Discount future labor income by 5% each year. 
#############################################################################################
function compute_expected_earnings(ω_grid::Vector{Float64},T_ω::Matrix{Float64},κ::Matrix{Any}, para::Model_Parameters; S = 100000)
    @unpack_Model_Parameters para

    perm_dists = [Categorical(T_η[i, :]) for i in 1:nη]
    transitory_dist = Categorical(T_ω[1,:])
    output = zeros(T,3)
    check = zeros(S)
    Threads.@threads for n = 1:T # Current period 
        for η_index_init in 1:nη  # Current aggregate state 
            Y_future = zeros(S,T-n)
            summed_Y_future = zeros(S)

            # Simulate one draw of the time series of future labor income 

            for s = 1:S 
                η_index = η_index_init
                for t = n+1:T # Future period
                    steps_ahead = 5 * (t - n) # Number of years in the future 
                    if t <= TR - 1
                        # Update the aggregate state 
                        η_index = rand(perm_dists[η_index])

                        # The new transitory state 
                        ω_index = rand(transitory_dist)
                        # Find the next value of labor income. 
                        if t == 8
                            check[s] = η_index
                        end 
                        Y_future[s,t-n] = compound(0.95,steps_ahead) * κ[t,2] * exp(η_grid[η_index] + ω_grid[ω_index])
                    else 
                        Y_future[s,t-n] = compound(0.95,steps_ahead) * κ[t,2]
                    end 
                end 
                # Sum up labor income  
                summed_Y_future[s] = sum(Y_future[s,:])
            end 

            # Compute expected future labor income as the mean 
            output[n,η_index_init] = mean(summed_Y_future)
        end 
    end 


    return output 
end 
#############################################################################################
# 9. Generate the time series of aggregate shocks each cohort experiences 
# in the exercise which matches the η shocks to the data of house price variation around trend. 
#############################################################################################
function generate_aggregate_ts(para::Model_Parameters)
    @unpack_Model_Parameters para
    persistent_index = [2,2,2,2,2,2,1,3,2,2]

    indices = zeros(T,T)
    for t = 1:T 
        indices[t,:] = vcat(persistent_index[T-t+1:T], 2 * ones(T - t))
    end 

    return Int.(indices)
end 

#############################################################################################
# 10.  Ensures only simulation values corresponding to the intended age are retained 
# after simulating.
#############################################################################################
function filter_age(mat::Matrix,age::Vector{Float64})
    age = Int64.(age)
    # Returns a Vector{Float64}
    S, T = size(mat)
    length(age) == S || throw(ArgumentError("length(age) must equal $S (rows in matrix)"))
    idx = CartesianIndex.(1:S, age)

    return mat[idx]
end 

#############################################################################################
# 11.  Returns the compounded factor over `T` periods given an annual rate `rate`
#############################################################################################
function compound(rate::Float64, T::Int)
    return (rate)^T
end
#############################################################################################
# 12.  Computes the stationary distribution of a Markov Chain
#############################################################################################
function stationary_distribution(η_grid::Vector{Float64},T_η::Matrix{Float64}; S = 100000, burn_in = 2000)
    nη = length(η_grid)
    # Find the stationary distribution of aggregate income process η
    perm_dists = [Categorical(T_η[i, :]) for i in 1:nη]

    # Need to choose some initial state to start the markov-chain 
    initial_dist = perm_dists[1]

    persistent = zeros(S)
    index_persistent = rand(initial_dist)

    # Persistent and Transitory components 
    persistent[1] = η_grid[index_persistent]

    for s = 2:S

        index_persistent = rand(perm_dists[index_persistent]) # Draw the new permanent component based upon the old one.         

        # Outputs 
        persistent[s] = η_grid[index_persistent]
    end 

    chain = persistent[(burn_in +1) : S]
    state_counts = countmap(chain)                 # Dict( state => frequency )
    T            = length(chain)
    Stationary_Distribution = [ state_counts[η_grid[s]] / T for s in 1:nη ]
    return Stationary_Distribution
end 


#############################################################################################
# 13.  Mortgage COnstraint
#############################################################################################
function mortgage_constraint(M_index::Int64, M_prime_index::Int64, Own_index::Int64, Own_prime_index::Int64,P::Float64, para::Model_Parameters)
    @unpack_Model_Parameters para
    
    M = M_grid[M_index]
    M_prime = M_grid[M_prime_index]

    # Holding a home
    if Own_index == 2 && Own_prime_index == 2
        out = M - M_prime # Can't accrue any more debt and must pay interest. 
    # Buying a home
    elseif Own_prime_index == 2 && Own_index == 1
       out =  (1 - d) * P  - M_prime  # Mortgage cannot exceed (1-d)% of the home value

    # Selling a home
    elseif Own_prime_index == 1 && Own_index == 2
       out =  - M_prime  # Must pay off mortgage when selling home

    # Renting
    elseif Own_prime_index == 1 && Own_index == 1
       out =  -(M + M_prime)  # Renters will not hold a mortgage
    end 

    return out
end 



