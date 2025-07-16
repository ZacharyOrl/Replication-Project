#############################################################################################

# This file holds a series of auxilary functions for use in solving and simulating the model: 
# Function list: 
# 1: The instantaneous utility function
# 2: The budget constraint, conditional on not moving 
# 3: The budget constraint, conditional on moving 
# 4: The value of bequest 
# 5: Bilinear Interpolation function 
# 6: Bilinear Interpolation function, applied to the policy functions and allowing for linear extrapolation. 
# 7: The expected value of future labor earnings in each state. 
# 8: The time series of shocks for each age cohort ( age is as of 1989 as per Cocco)
# 9: Each agent has a full simulated lifecycle in simulations. I pull only a single "age" from each lifecycle according to Cocco's weights.
# 10: Compounds an annual rate over T periods. 
# 11: Computes the stationary distribution of a Markov chain. 
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
# If Inv_Move == 0 & H == H_prime, then we use the no-move budget constraint 
function no_move_budget_constraint(X::Float64, H::Float64, P::Float64, c::Float64, H_prime::Float64, LTV::Float64, FC::Int64, para::Model_Parameters)
    @unpack δ, F, λ, d = para

    S_and_B = X - c - FC*F - δ * P * H + LTV * P * H_prime

    return S_and_B
end 

#############################################################################################
# 3.
#############################################################################################
# If Inv_Move != 0 and/or H != H_prime, then we use the move budget constraint 
function move_budget_constraint(X::Float64, H::Float64, P::Float64, c::Float64, H_prime::Float64, LTV::Float64, FC::Int64, para::Model_Parameters)
    @unpack δ, F, λ, d = para


    S_and_B = X - c - FC*F - δ * P * H + LTV * P * H_prime + (1-λ)* P * H - P * H_prime

    return S_and_B
end 
#############################################################################################
# 4.
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
                if W <= 0 
                    V[:, :, η_index, H_index, X_index] .+= pun 
                else

                V[:, :, η_index, H_index, X_index] .+=  weight * ( W^(1-γ) )/(1-γ)  
                end 
            end 
        end 
    end 

    return V
end 

#############################################################################################
# 6. Interpolate the policy functions after solving the Model
# Allows for extrapolation as some rounding is necessary
# (due to moving and stock market entry being discrete).
#############################################################################################
function interpolate_policy_funcs(sols::Solutions,para::Model_Parameters)
    @unpack_Model_Parameters para 
    @unpack val_func,c_pol_func, LTV_pol_func, H_pol_func, FC_pol_func, α_pol_func, Move_pol_func = sols

       # Generate interpolation functions for cash-on hand given each possible combination of the other states
       c_interp_functions = Array{Any}(undef, 2 * 2 * nη * nH ,T)
       LTV_interp_functions = Array{Any}(undef, 2 * 2 * nη * nH ,T) 
       H_interp_functions = Array{Any}(undef, 2 * 2 * nη * nH ,T) 
       FC_interp_functions = Array{Any}(undef, 2 * 2 * nη * nH,T) 
       α_interp_functions = Array{Any}(undef, 2 * 2 * nη * nH ,T) 
       Move_interp_functions = Array{Any}(undef, 2 * 2 * nη * nH,T) 

        for n = 1:T
            for H_index = 1:nH
                for Inv_Move_index in 1:2
                    for IFC_index in 1:2
                        for η_index in 1:nη

                        # Compute linear index 
                        index = lin[Inv_Move_index, IFC_index, η_index, H_index]

                        # Create interpolated policy functions
                        c_interp_functions[index,n]     = Spline1D(X_grid,c_pol_func[Inv_Move_index, IFC_index, η_index, H_index, :, n], k = 1)
                        LTV_interp_functions[index,n]   = x -> clamp(Spline1D(X_grid,LTV_pol_func[Inv_Move_index, IFC_index, η_index, H_index, :, n],k = 1)(x),0.0,0.85)
                        H_interp_functions[index,n]     = Spline1D(X_grid,H_pol_func[Inv_Move_index, IFC_index, η_index, H_index, :, n], k = 1)
                        FC_interp_functions[index,n]    = Spline1D(X_grid,FC_pol_func[Inv_Move_index, IFC_index, η_index, H_index, :, n], k = 1)
                        α_interp_functions[index,n]     = x -> clamp(Spline1D(X_grid,α_pol_func[Inv_Move_index, IFC_index, η_index, H_index, :, n], k = 1)(x),0.0,1.0)
                        Move_interp_functions[index,n]  = x -> clamp(Spline1D(X_grid,Move_pol_func[Inv_Move_index, IFC_index, η_index, H_index, :, n], k = 1)(x),0.0,1.0)
                        end
                    end
                end
            end
        end

    return c_interp_functions, LTV_interp_functions, H_interp_functions, FC_interp_functions, α_interp_functions, Move_interp_functions
end 

#############################################################################################
# 7. Compute future expected earnings for each aggregate state and age . 
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
# 8. Generate the time series of aggregate shocks each cohort experiences 
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
# 9.  Ensures only simulation values corresponding to the intended age are retained 
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
# 10.  Returns the compounded factor over `T` periods given an annual rate `rate`
#############################################################################################
function compound(rate::Float64, T::Int)
    return (rate)^T
end
#############################################################################################
# 11.  Computes the stationary distribution of a Markov Chain
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
# 11.  Discrete Choice Routine
#############################################################################################
#=
function discrete_choice_routine(T::Int64,c::Float64, LTV::Float64, X::Float64, H::Float64, P::Float64, index::Int64,
                                H_interp_functions::Array{Any}, Move_interp_functions::Array{Any},FC_interp_functions::Array{Any},para::Model_Parameters)
    @unpack_Model_Parameters para

    # Compute the unadjusted policies 
    moved = Move_interp_functions[index,T](X)
    housing = H_interp_functions[index,T](X)
    stock_market_entry = FC_interp_functions[index,T](X)

    # 8 possible rounding combinations 
    # But you don't need to round housing if 
    # Rounding not needed for moving and stock market entry implies no problem 
    if round(moved) == moved && round(stock_market_entry) == stock_market_entry 
        if moved == 0 
            H_out = H 
        else 
            H_out = housing 
        end 

        return moved, H_out ,stock_market_entry 
    end 

    # Rounding only need for stock market entry - 
    # randomize entry, subject to satisfying the budget constraint 
    if round(moved) == moved && round(stock_market_entry) != stock_market_entry 

        if moved == 0 
            H_out = H 

            if rand() > stock_market_entry && no_move_budget_constraint(X,H,P,c,H_out,LTV,1,para) > 0.0
                stock_market_entry = 1
            else 
                stock_market_entry = 0 
            end  

        else 
            H_out = housing 

            if rand() > stock_market_entry && move_budget_constraint(X,H,P,c,H_out,LTV,1,para) > 0.0
                stock_market_entry = 1
            else 
                stock_market_entry = 0 
            end  
        end 


        return moved, H_out,stock_market_entry 
    end 

    # Rounding needed for moving but not stock market entry and not for the housing value conditional on moving
    if round(moved) != moved && round(stock_market_entry) == stock_market_entry && typeof(findfirst(==(housing),H_grid)) != Nothing
        
        if rand() > moved && move_budget_constraint(X,H,P,c,housing,LTV,stock_market_entry,para) > 0.0
            moved_out = 1 
        else 

        # Rounding needed for housing 
        if typeof(findfirst(==(housing),H_grid)) == Nothing

            # Choose between the two closest values of the first feasible housing gridpoint that is not ordered 
            # after "housing"
            k = searchsortedlast(H_grid[2:nH], housing)
            H_choice_low = H_grid[k]
            H_choice_high = H_grid[k+1]

            # Rounding needed for stock market entry 
            if round(stock_market_entry) != stock_market_entry
                
                
                S_and_B_1 = no_move_budget_constraint(X, H, P, c, H, 
                                        LTV, stock_market_entry, para)


                if moved[s,n] == 0 
                    housing[s,n] = H 
                    H_prime_index = H_index
                else 
                    housing[s,n]  =  round_to_grid(H_interp_functions[index,n](cash_on_hand[s, n]), H_grid[2:nH])
                    H_prime_index = searchsortedlast(H_grid[1:nH], housing[s,n])
                end 

                # Need to adjust stock market entry so it is on the grid 
                stock_market_entry[s,n] = round_to_grid(FC_interp_functions[index,n](cash_on_hand[s, n]), FC_grid)

                # Compute savings 
                if moved[s,n] == 0 
                    S_and_B = no_move_budget_constraint(cash_on_hand[s,n], H, P, consumption[s,n], housing[s,n], 
                                        LTV[s,n], stock_market_entry[s,n], para)
                end 

                if moved[s,n] == 1 
                    S_and_B =   move_budget_constraint(cash_on_hand[s,n], H, P, consumption[s,n], housing[s,n], 
                                        LTV[s,n], stock_market_entry[s,n], para)
    end 
end 

=#

function map_X(value::Float64, para::Model_Parameters)
    @unpack_Model_Parameters para

    n = length(X_grid)
    step = (X_max - X_min) / (n - 1)
    idx  = (value - X_min) / step + 1
    return clamp(idx, 1, n)
end