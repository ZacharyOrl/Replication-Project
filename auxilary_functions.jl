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

#############################################################################################
# 1.
#############################################################################################

# This function takes the choice of consumption and housing today and returns the flow utility value. 
function flow_utility_func(c::Float64, H_prime::Float64, para::Model_Parameters)
    @unpack γ, θ = para

    return  (    ( c^(1-θ) * H_prime^θ )^( 1 - γ )   ) / (1 - γ)
end 

#############################################################################################
# 2.
#############################################################################################
# Takes as input all states and choices necessary to pin down the budget constraint
# and outputs the sum of stocks and bonds (LHS of the budget constraint in Cocco)
function budget_constraint(X::Float64, H::Float64, P::Float64, Inv_Move::Int64, c::Float64, 
                           H_prime::Float64, LTV::Float64, FC::Int64, para::Model_Parameters)
    @unpack δ, F, λ = para

    # If not forced to move and there is no house trade
    if (Inv_Move == 0) && (H_prime == H)
        S_and_B = X - c - FC*F - δ * P * H + LTV * P * H_prime 

    # Otherwise:
    else 
        S_and_B = X - c - FC*F - δ * P * H + LTV * P * H_prime + (1-λ)* P * H - P * H_prime
    end 

    return S_and_B
end 

#############################################################################################
# 3.
#############################################################################################
# If Inv_Move == 0 & H == H_prime, then we use the no-move budget constraint 
function no_move_budget_constraint(X::Float64, H::Float64, P::Float64, c::Float64, H_prime::Float64, LTV::Float64, FC::Int64, para::Model_Parameters)
    @unpack δ, F, λ, d = para

    S_and_B = X - c - FC*F - δ * P * H + LTV * P * H_prime

    return S_and_B
end 

#############################################################################################
# 4.
#############################################################################################
# If Inv_Move != 0 and/or H != H_prime, then we use the move budget constraint 
function move_budget_constraint(X::Float64, H::Float64, P::Float64, c::Float64, H_prime::Float64, LTV::Float64, FC::Int64, para::Model_Parameters)
    @unpack δ, F, λ, d = para


    S_and_B = X - c - FC*F - δ * P * H + LTV * P * H_prime + (1-λ)* P * H - P * H_prime

    return S_and_B
end 
#############################################################################################
# 5.
#############################################################################################
# Computes the value of the bequest in T+1 
function compute_bequest_value(V::Array{Float64,5}, para::Model_Parameters)
    @unpack_Model_Parameters para 

    # Loop over Cash-on-hand states
    for X_index in 1:nX
        X = X_grid[X_index]

        # Loop over Housing States 
        for H_index in 1:nH
            H = H_grid[H_index]
            
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

#############################################################################################
# 6.
#############################################################################################

function bilinear_interp(F::Array{Float64, 2}, x1::Vector{Float64}, x2::Vector{Float64})
    #= Bilinear interpolation for 2D grid - no extrapolation
    # Linear because cubic spline wasn't working well for a pretty sparse grid. 
    Arguments:  F (Array): 2D grid of function values evaluated on grid points
                x1 (Vector): grid points for first dimension - must be evenly spaced
                x2 (Vector): grid points for second dimension - must be evenly spaced
    Returns:    interp (Function): bilinear interpolation function =#
    x1_grid = range(minimum(x1), maximum(x1), length=length(x1))
    x2_grid = range(minimum(x2), maximum(x2), length=length(x2))

    interp = interpolate(F, BSpline(Linear()))
    return Interpolations.scale(interp, x1_grid, x2_grid)
end

#############################################################################################
# 7. Interpolate the policy functions after solving the Model
# Allows for extrapolation as some rounding is necessary
# (due to moving and stock market entry being discrete).
#############################################################################################
function interpolate_policy_funcs(sols::Solutions,para::Model_Parameters)
    @unpack_Model_Parameters para 
    @unpack val_func,c_pol_func, LTV_pol_func, H_pol_func, FC_pol_func, α_pol_func, Move_pol_func = sols

       # Generate interpolation functions for cash-on hand given each possible combination of the other states
       c_interp_functions = Array{Any}(undef, 2 * 2 * nη *  nH,T)
       LTV_interp_functions = Array{Any}(undef, 2 * 2 * nη *  nH,T) 
       H_interp_functions = Array{Any}(undef, 2 * 2 * nη *  nH,T) 
       FC_interp_functions = Array{Any}(undef, 2 * 2 * nη * nH,T) 
       α_interp_functions = Array{Any}(undef, 2 * 2 * nη *  nH,T) 
       Move_interp_functions = Array{Any}(undef, 2 * 2 * nη *  nH,T) 

        for n = 1:T
            for Inv_Move_index in 1:2
                for IFC_index in 1:2
                    for η_index in 1:nη

                    # Compute linear index 
                    index = lin[Inv_Move_index, IFC_index, η_index]
                    # Create interpolated policy functions
                    c_interp_functions[index,n]     = extrapolate(bilinear_interp(c_pol_func[Inv_Move_index, IFC_index, η_index, :, :, n], H_grid, X_grid), Line())
                    LTV_interp_functions[index,n]   = extrapolate(bilinear_interp(LTV_pol_func[Inv_Move_index, IFC_index, η_index, :, :, n], H_grid, X_grid), Line())
                    H_interp_functions[index,n]     = extrapolate(bilinear_interp(H_pol_func[Inv_Move_index, IFC_index, η_index, :, :, n], H_grid, X_grid), Line())
                    FC_interp_functions[index,n]    = extrapolate(bilinear_interp(FC_pol_func[Inv_Move_index, IFC_index, η_index, :, :, n], H_grid, X_grid), Line())
                    α_interp_functions[index,n]     = extrapolate(bilinear_interp(α_pol_func[Inv_Move_index, IFC_index, η_index, :, :, n], H_grid, X_grid), Line())
                    Move_interp_functions[index,n]  = extrapolate(bilinear_interp(Move_pol_func[Inv_Move_index, IFC_index, η_index, :, :, n], H_grid, X_grid), Line())
                    end
                end
            end
        end

    return c_interp_functions, LTV_interp_functions, H_interp_functions, FC_interp_functions, α_interp_functions, Move_interp_functions
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



