#############################################################################################

# This file holds a series of auxilary functions for use in solving and simulating the model: 
# Function list: 
# 1: The instantaneous utility function
# 2: The budget constraint, conditional on not moving 
# 3: The budget constraint, conditional on moving 
# 4: The value of bequest 

# 5: The expected value of future labor earnings in each state (for the simulation)
# 6: The time series of shocks for each age cohort ( age is as of 1989 as per Cocco)
# 7: Each agent has a full simulated lifecycle in simulations. I pull only a single "age" from each lifecycle according to Cocco's weights.
# 8: Compounds an annual gross rate over T periods. 

# 9: Computes the stationary distribution of a Markov chain. 
# 10: A function which map cash-in-hand values for interpolation (taken from Campbell - Clara - Cocco , 2020)
# 11: A function which rounds cash-in-hand values to the cash-in-hand grid for the simulation (taken from Campbell - Clara - Cocco, 2020)
# 12: Helper function which combines the output of a simulation run. 

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
# 12.  Maps an X value for interpolation 
#############################################################################################

function map_X(value::Float64, para::Model_Parameters)
    @unpack_Model_Parameters para

    n = length(X_grid)
    step = (X_max - X_min) / (n - 1)
    idx  = (value - X_min) / step + 1
    return clamp(idx, 1, n)
end
#############################################################################################
# 12.  # Round a value onto a grid (for the simulation)
#############################################################################################

# Closest 
function round_to_grid(val, grid::AbstractVector)
    k = searchsortedlast(grid, val)          # grid[k] ≤ val < grid[k+1]

    if k == 0                                # below grid
        return grid[1]
    elseif k == length(grid)                 # above grid (or at the top knot)
        return grid[end]
    else
        # pick the nearer of grid[k] and grid[k+1]
        return (val - grid[k] < grid[k+1] - val) ? grid[k] : grid[k + 1]
    end
end

# Round down 
#=
# Round a value onto a grid 
function round_to_grid(val, grid::Vector)
    k = searchsortedlast(grid, val)
    return grid[ max(k,1)]    # handles val below the grid
end 
=#

#############################################################################################
# 13: Small helper function which combines the outputs of a simulation run 
#############################################################################################
function sim_to_matrix(sim::Sim_Results)

    return hcat(
        sim.bonds,           sim.stocks,            sim.stock_share,
        sim.stock_market_entry,                     sim.IFC_paid,       
        sim.housing,         sim.moved,             sim.Inv_Move_shock,
        sim.cash_on_hand,    sim.expected_earnings,
        sim.debt,            sim.LTV,               sim.consumption,       
        sim.wealth,          sim.bequest,           sim.income,                       
        sim.persistent,      sim.transitory,        sim.stock_market_shock,
        sim.age,     sim.education
        )           
    
end
