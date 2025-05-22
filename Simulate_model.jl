function simulate_model(para,sols,S::Int64, edu::Int64)
    # Simulates the solved model S times, returns assets, consumption, income, persistent shock and transitroy shock by age. 
    # Applies analysis weights based upon the education group. 

    @unpack ι_grid, η_grid,p_grid, H_state_grid, FC_grid, X_grid,  T_η, T_ι, nη, nH, π_η, T, TR, π_m, P_bar, b, μ, R_F, R_D, δ, λ, g, X_max, X_min, lin = para
    @unpack val_func,c_pol_func, D_pol_func, H_pol_func, FC_pol_func, α_pol_func, κ, σ_ω = sols

    # Generate Transitory earnings grid based on σ_w of group
    ω_grid::Vector{Float64} = rouwenhorst(σ_ω, 0.0, g)[1] 
    T_ω::Matrix{Float64} = rouwenhorst(σ_ω, 0.0, g)[2]
    nω::Int64 = length(ω_grid)

    c_interp_functions, D_interp_functions, H_interp_functions, FC_interp_functions, α_interp_functions = interpolate_policy_funcs(sols,para)

    # Education group 

    # Distribution over the initial permanent component
    initial_dist = 1.0

    # Distribution over the transitory component (use that it isn't persistent, so won't vary over time)
    transitory_dist = Categorical(T_ω[1,:])

    # Distribution over Stock Market Shock 
    stock_dist = Categorical(T_ι[1,:])

    # State-contingent distributions over the aggregate states
    perm_dists = [Categorical(T_η[i, :]) for i in 1:nη]

    # Stationary distribution over initial aggregate state
    initial_dist = Categorical(π_η)

    # Outputs
    bonds = zeros(S,T+1) 
    stocks = zeros(S,T+1) 
    stock_share = zeros(S, T+1)
    stock_market_entry = zeros(Int64, S,T+1)
    IFC_paid = zeros(S,T+1)

    housing = zeros(S,T+1)
    moved = zeros(S,T+1)
    cash_on_hand = zeros(S,T+1) 

    debt = zeros(S,T+1)
    consumption = zeros(S,T+1) 
    wealth = zeros(S,T+1) # Savings + Housing - Debt 
    bequest = zeros(S)

    persistent = zeros(S,T+1)
    transitory = zeros(S,T+1)
    stock_market_shock = zeros(S, T+1)

    weights_var = zeros(S, T+1)
    education   = ones(S, T+1) * edu
    for s = 1:S
        η_index = rand(initial_dist) # Draw the initial aggregate state from its stationary distribution 
        ι_index = rand(stock_dist)
        ω_index   = rand(transitory_dist)

        # Save values of shocks 
        persistent[s,1] = η_grid[η_index]
        transitory[s,1] = ω_grid[ω_index]
        stock_market_shock[s,1] = ι_grid[ι_index]

        # Initialize whether the agent has to involuntary move
        if rand() > π_m
            Inv_Move_index = 1 
            Inv_Move = 0 
        else 
            Inv_Move_index = 2
            Inv_Move = 1
        end 

        # Initialize the price index - perfectly correlated with the labor market state 
        P = P_bar * exp(b + p_grid[η_index])

        # Start with 0 assets, 0 debt, 0 housing, not having entered the stock market
        H_index = 1 
        IFC_index = 1

        cash_on_hand[s,1]   = 0.0 + κ[1,2] * exp(η_grid[η_index] + ω_grid[ω_index])
        if cash_on_hand[s,1] > X_max
            println(cash_on_hand[s,n])
        end 
        # This is purely an input into the budget constraint and so is not saved.
        H = H_state_grid[H_index]

        # Overall index 
        index = lin[Inv_Move_index, IFC_index, η_index, H_index]

        # Compute choices 
        consumption[s,1] = c_interp_functions[index,1](cash_on_hand[s,1])
        debt[s,1] = D_interp_functions[index,1](cash_on_hand[s,1])

        # Need to adjust housing and other choices so they are on the grid: 
        housing[s,1]  =  H_state_grid[floor_grid_index(H_interp_functions[index,1](cash_on_hand[s,1]), H_state_grid)]
        moved[s,1] = (housing[s,1] != H_state_grid[H_index])

        # Need to adjust stock market entry so it is on the grid 
        stock_market_entry[s,1] = FC_grid[floor_grid_index(FC_interp_functions[index,1](cash_on_hand[s,1]), FC_grid)]
        if stock_market_entry[s,1] == 0 && IFC_index == 1
            stock_share[s,1] = 0.0
        else 
            stock_share[s,1] = α_interp_functions[index,1](cash_on_hand[s,1])
        end 
   
        # Find next period's indices
        H_prime_index = findfirst(x -> x == housing[s,1], H_state_grid)
        IFC_prime_index = max(stock_market_entry[s,1],IFC_index - 1) + 1
        IFC_paid[s,1] = IFC_prime_index - 1

        # Compute savings 
        S_and_B = budget_constraint(cash_on_hand[s,1], H, P, Inv_Move, consumption[s,1], housing[s,1], 
                                    debt[s,1], stock_market_entry[s,1], para)
                                    
        stocks[s,1] = stock_share[s,1] *  S_and_B 
        bonds[s,1] = (1.0 - stock_share[s,1]) *  S_and_B       

        # Compute wealth 
        wealth[s,1] = stocks[s,1] + bonds[s,1] + P * housing[s,1] - debt[s,1]

        # Simulate working age 
        for n = 2:TR - 1  
            # Draw new values for the labor income shocks 
            η_index = rand(perm_dists[η_index]) # Draw the new permanent component based upon the old one. 
            ι_index = rand(stock_dist)
            ω_index   = rand(transitory_dist)

            # Save values of shocks 
            persistent[s,n] = η_grid[η_index]
            transitory[s,n] = ω_grid[ω_index]
            stock_market_shock[s,n] = ι_grid[ι_index]

            # Initialize whether the agent has to involuntary move
            if rand() > π_m
                Inv_Move_index = 1 
                Inv_Move = 0 
            else 
                Inv_Move_index = 2 
                Inv_Move = 1
            end 

            # Turn the indices of the choices last period to the states today.
            H_index =   H_prime_index
            H = H_state_grid[H_index]

            IFC_index = IFC_prime_index

            # Compute cash on hand 
            P = P_bar * exp(b * (n) + p_grid[η_index])
            R_S = exp(stock_market_shock[s,n] + μ)
            Y = κ[n,2] * exp( η_grid[η_index] + ω_grid[ω_index])

            if cash_on_hand[s,n] > X_max
                println(cash_on_hand[s,n])
            end 

            cash_on_hand[s,n] = Y + R_S * stocks[s,n-1] + R_F * bonds[s,n-1] - R_D * debt[s,n-1]

            # Overall index 
            index = lin[Inv_Move_index, IFC_index, η_index, H_index]

            # Compute choices 
            consumption[s,n] = c_interp_functions[index,n](cash_on_hand[s,n])
            debt[s,n] = D_interp_functions[index,n](cash_on_hand[s,n])
            # Need to adjust housing and other choices so they are on the grid: 
            housing[s,n]  =  H_state_grid[floor_grid_index(H_interp_functions[index,n](cash_on_hand[s,n]), H_state_grid)]
            moved[s,n] = (housing[s,n] != housing[s,n-1])

            # Need to adjust stock market entry so it is on the grid 
            stock_market_entry[s,n] = FC_grid[floor_grid_index(FC_interp_functions[index,n](cash_on_hand[s,n]), FC_grid)]
            if stock_market_entry[s,n] == 0 && IFC_index == 1
                stock_share[s,n] = 0.0
            else 
                stock_share[s,n] = α_interp_functions[index,n](cash_on_hand[s,n])
            end 
   
            # Find next period's indices
            H_prime_index = findfirst(x -> x == housing[s,n], H_state_grid)
            IFC_prime_index = max(stock_market_entry[s,n],IFC_index - 1) + 1
            IFC_paid[s,n] = IFC_prime_index - 1

            # Compute savings 
            S_and_B = budget_constraint(cash_on_hand[s,n], H, P, Inv_Move, consumption[s,n], housing[s,n], 
                                        debt[s,n], stock_market_entry[s,n], para)
                                    
            stocks[s,n] = stock_share[s,n] *  S_and_B 
            bonds[s,n] = (1.0 - stock_share[s,n]) *  S_and_B       

            # Compute wealth 
            wealth[s,n] = stocks[s,n]  + bonds[s,n] + P * housing[s,n] - debt[s,n]

        end 

        # Simulate retirement
        for n = TR:T
            # Draw new values for the labor income shocks 
            η_index = rand(perm_dists[η_index]) # Draw the new permanent component based upon the old one. 
            ι_index = rand(stock_dist)

            # Save values of shocks 
            persistent[s,n] = η_grid[η_index]
            stock_market_shock[s,n] = ι_grid[ι_index]

            # Initialize whether the agent has to involuntary move
            if rand() > π_m
                Inv_Move_index = 1 
                Inv_Move = 0 
            else 
                Inv_Move_index = 2 
                Inv_Move = 1
            end 

            # Turn the indices of the choices last period to the states today.
            H_index =  H_prime_index
            H = H_state_grid[H_index]'

            IFC_index = IFC_prime_index

            # Compute cash on hand 
            P = P_bar * exp(b * (n) + p_grid[η_index])
            R_S = exp(stock_market_shock[s,n] + μ)
            Y = κ[n,2]
            cash_on_hand[s,n] = Y + R_S * stocks[s,n-1] + R_F * bonds[s,n-1] - R_D * debt[s,n-1]
        
            if cash_on_hand[s,n] > X_max
                println("Y",Y, " cash on hand ",cash_on_hand[s,n], " bonds ", bonds[s,n-1], " stocks ", stocks[s,n-1])
            end 
            # Overall index 
            index = lin[Inv_Move_index, IFC_index, η_index, H_index]

            # Compute choices 
            consumption[s,n] = c_interp_functions[index,n](cash_on_hand[s,n])
            debt[s,n] = D_interp_functions[index,n](cash_on_hand[s,n])
            # Need to adjust housing and other choices so they are on the grid: 
            housing[s,n]  =  H_state_grid[floor_grid_index(H_interp_functions[index,n](cash_on_hand[s,n]), H_state_grid)]
            moved[s,n] = (housing[s,n] != housing[s,n-1])

            # Need to adjust stock market entry so it is on the grid 
            stock_market_entry[s,n] = FC_grid[floor_grid_index(FC_interp_functions[index,n](cash_on_hand[s,n]), FC_grid)]
            if stock_market_entry[s,n] == 0 && IFC_index == 1
                stock_share[s,n] = 0.0
            else 
                stock_share[s,n] = α_interp_functions[index,n](cash_on_hand[s,n])
            end 
   
            # Find next period's indices
            H_prime_index = findfirst(x -> x == housing[s,n], H_state_grid)
            IFC_prime_index = max(stock_market_entry[s,n],IFC_index - 1) + 1
            IFC_paid[s,n] = IFC_prime_index - 1

            # Compute savings 
            S_and_B = budget_constraint(cash_on_hand[s,n], H, P, Inv_Move, consumption[s,n], housing[s,n], 
                                        debt[s,n], stock_market_entry[s,n], para)
                                    
            stocks[s,n] = stock_share[s,n] *  S_and_B 
            bonds[s,n] = (1.0 - stock_share[s,n]) *  S_and_B       

            # Compute wealth 
            wealth[s,n] = stocks[s,n] + bonds[s,n] + P * housing[s,n] - debt[s,n]

        end 

        # Simulate bequest
        η_index = rand(perm_dists[η_index]) # Draw the new permanent component based upon the old one. 
        ι_index = rand(stock_dist)

        # Save values of shocks 
        persistent[s,T+1] = η_grid[η_index]
        stock_market_shock[s,T+1] = ι_grid[ι_index]

        # Compute cash on hand 
        P = P_bar * exp(b * (T+1) + p_grid[η_index])
        R_S = exp(stock_market_shock[s,T+1] + μ)
        cash_on_hand[s,T+1] = R_S * stocks[s,T+1] + R_F * bonds[s,T] - R_D * debt[s,T]
        
        # Agents are forced to sell their house when they die
        bequest[s] = cash_on_hand[s,T+1] - δ * housing[s,T] * P +  (1-λ) *  P * housing[s,T]    
    end 

    # Add the age-weights for the education group 
    for n = 1:T 
        #weights_var[:,n] .= copy(weights[n,edu])
    end 

    return SimResults(
        bonds, stocks, stock_share, stock_market_entry, IFC_paid,
        housing, moved, cash_on_hand,
        debt, consumption, wealth, bequest,
        persistent, transitory, stock_market_shock, weights_var, education
    )
end

function interpolate_policy_funcs(sols::Solutions,para::Model_Parameters)
    @unpack_Model_Parameters para 
    @unpack val_func,c_pol_func, D_pol_func, H_pol_func, FC_pol_func, α_pol_func = sols

       # Generate interpolation functions for cash-on hand given each possible combination of the other states
       c_interp_functions = Array{Any}(undef, 2 * 2 * nη *  (nH+1),T)
       D_interp_functions = Array{Any}(undef, 2 * 2 * nη *  (nH+1),T) 
       H_interp_functions = Array{Any}(undef, 2 * 2 * nη *  (nH+1),T) 
       FC_interp_functions = Array{Any}(undef, 2 * 2 * nη * (nH+1),T) 
       α_interp_functions = Array{Any}(undef, 2 * 2 * nη *  (nH+1),T) 

        for n = 1:T
            for Inv_Move_index in 1:2
                for IFC_index in 1:2
                    for η_index in 1:nη
                        for H_index in 1:(nH + 1)
                            # Compute linear index 
                            index = lin[Inv_Move_index, IFC_index, η_index, H_index]
                            # Create interpolated policy functions
                            c_interp_functions[index,n]  = linear_interp(c_pol_func[Inv_Move_index, IFC_index, η_index, H_index, :, n], X_grid)
                            D_interp_functions[index,n]  = linear_interp(D_pol_func[Inv_Move_index, IFC_index, η_index, H_index, :, n], X_grid)
                            H_interp_functions[index,n]  = linear_interp(H_pol_func[Inv_Move_index, IFC_index, η_index, H_index, :, n], X_grid)
                            FC_interp_functions[index,n] = linear_interp(FC_pol_func[Inv_Move_index, IFC_index, η_index, H_index, :, n], X_grid)
                            α_interp_functions[index,n]  = linear_interp(α_pol_func[Inv_Move_index, IFC_index, η_index, H_index, :, n], X_grid)
                        end
                    end
                end
            end
        end

    return c_interp_functions, D_interp_functions, H_interp_functions, FC_interp_functions, α_interp_functions
end 


#=
    floor_grid_index(x, grid)::Int

Return the index of the greatest element in `grid` that is ≤ `x`.
`grid` must be sorted in ascending order.
If `x` is below the smallest grid value, 1 is returned.
=#
function floor_grid_index(x, grid)
    hi = searchsortedfirst(grid, x) 
    if hi > length(grid)
        hi = length(grid)
    end 
    if grid[hi] > x     # first element ≥ x
        return (hi == 1) ? 1 : hi - 1        # step back one slot
    else 
        return hi 
    end 
end

#=
    SimResults

All data produced by one call to `simulate_model`.
=#
struct SimResults
    bonds               ::Matrix{Float64}
    stocks              ::Matrix{Float64}
    stock_share         ::Matrix{Float64}
    stock_market_entry  ::Matrix{Int}
    IFC_paid            ::Matrix{Int}

    housing             ::Matrix{Float64}
    moved               ::Matrix{Float64}
    cash_on_hand        ::Matrix{Float64}

    debt                ::Matrix{Float64}
    consumption         ::Matrix{Float64}
    wealth              ::Matrix{Float64}
    bequest             ::Vector{Float64}

    persistent          ::Matrix{Float64}
    transitory          ::Matrix{Float64}
    stock_market_shock  ::Matrix{Float64}

    weights_var         ::Matrix{Float64}
    education           ::Matrix{Float64}
end
