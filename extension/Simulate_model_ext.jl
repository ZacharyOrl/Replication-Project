function simulate_model(para,sols,S::Int64, edu::Int64)
    # Simulates the solved model S times, returns assets, consumption, income, persistent shock and transitroy shock by age. 
    # Applies analysis weights based upon the education group. 

    @unpack ι_grid, M_grid, η_grid,p_grid, Own_grid, FC_grid, Move_grid, X_grid,  T_η, T_ι, nη, π_η, T, TR, π_m, P_bar, b, μ, R_F, R_D, R_D_R, δ, λ, g, X_max, X_min, lin, wts = para
    @unpack val_func,c_pol_func, M_pol_func, Own_pol_func, FC_pol_func, α_pol_func, Move_pol_func, κ, σ_ω, F = sols
    
    # Set the number of simulations for this education group 
    wt_tot = sum(wts[:,2])
    tot_sims = Int(round(S * wt_tot))

    # Generate Transitory earnings grid based on σ_w of group
    ω_grid::Vector{Float64} = rouwenhorst(σ_ω, 0.0, g)[1] 
    T_ω::Matrix{Float64} = rouwenhorst(σ_ω, 0.0, g)[2]
    nω::Int64 = length(ω_grid)

    c_interp_functions, M_interp_functions, FC_interp_functions, α_interp_functions, Move_interp_functions, Own_interp_functions = interpolate_policy_funcs(sols,para)
    expected_earnings_vals =  compute_expected_earnings(ω_grid,T_ω,κ, para)
    # Education group

    # Distribution over the transitory component (use that it isn't persistent, so won't vary over time)
    transitory_dist = Categorical(T_ω[1,:])

    # Distribution over Stock Market Shock 
    stock_dist = Categorical(T_ι[1,:])

    # State-contingent distributions over the aggregate states
    perm_dists = [Categorical(T_η[i, :]) for i in 1:nη]

    # Stationary distribution over initial aggregate state
    initial_dist = Categorical(π_η)

    # Outputs
    bonds = zeros(tot_sims,T+1) 
    stocks = zeros(tot_sims,T+1) 
    stock_share = zeros(tot_sims, T+1)
    stock_market_entry = zeros(Int64, tot_sims,T+1)
    IFC_paid = zeros(tot_sims,T+1)
    Inv_Move_shock = zeros(tot_sims, T+1)
    own = zeros(Int64, tot_sims, T+1)

    moved = zeros(tot_sims,T+1)
    cash_on_hand = zeros(tot_sims,T+1)
    expected_earnings = zeros(tot_sims, T+1) 

    debt = zeros(tot_sims,T+1)

    consumption = zeros(tot_sims,T+1) 
    wealth = zeros(tot_sims,T+1) # Savings + Housing - Debt 
    bequest = zeros(tot_sims)

    income = zeros(tot_sims,T+1)
    persistent = zeros(tot_sims,T+1)
    transitory = zeros(tot_sims,T+1)
    stock_market_shock = zeros(tot_sims, T+1)

    age = zeros(tot_sims)

    sim_start = 1

    for a = 1:T
        stop = Int(sim_start + round(S*wts[a,edu]) - 1) 
        
         for s = sim_start: stop
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
                Inv_Move_shock[s,1] = 0 
            else 
                Inv_Move_index = 2
                Inv_Move_shock[s,1] = 1
            end 

            # Initialize the price index - perfectly correlated with the labor market state 
            P = P_bar * exp(b + p_grid[η_index])

            expected_earnings[s,1] = expected_earnings_vals[1,η_index] 

            # Start with 0 assets, 0 debt, renting, not having entered the stock market
            Own_index = 1 
            IFC_index = 1
            M_index = 1

            income[s,1] = κ[1,2] * exp(η_grid[η_index] + ω_grid[ω_index])
            cash_on_hand[s,1]   = 0.0 + income[s,1]

            if cash_on_hand[s,1] > X_max
                println(cash_on_hand[s,1])
            end 
            # This is purely an input into the budget constraint and so is not saved.
            Own = Own_grid[Own_index]

            # Overall index 
            index = lin[Inv_Move_index, IFC_index, η_index, Own_index, M_index]

            # Compute choices 
            consumption[s,1] = c_interp_functions[index,1](cash_on_hand[s,1])

            # Compute whether the agent moved - adjusting so it is on grid. 
            moved[s,1] = floor(Move_interp_functions[index,1](cash_on_hand[s,1]))
            
            if moved[s,1] == 0
                own[s,1] = 0

                M_prime_index = 0
                debt[s,1] = 0
            else     
                own[s,1]  = floor(Own_interp_functions[index,1](cash_on_hand[s,1]))

                # Place mortgage choice on the grid 
                M_interp_val = M_interp_functions[index,1](cash_on_hand[s,1])
                M_prime_index = argmin(abs.(M_grid .- M_interp_val))
                debt[s,1] = M_grid[M_prime_index]
            end 

            # Need to adjust stock market entry so it is on the grid 
            stock_market_entry[s,1] = floor(FC_interp_functions[index,1](cash_on_hand[s,1]))
            
            if stock_market_entry[s,1] == 0 && IFC_index == 1
                stock_share[s,1] = 0.0
            else 
                stock_share[s,1] = α_interp_functions[index,1](cash_on_hand[s,1])
            end 
    
            # Find next period's indices
            IFC_prime_index = max(stock_market_entry[s,1],IFC_index - 1) + 1
            IFC_paid[s,1] = IFC_prime_index - 1

            # Compute savings 

            if moved[s,1] == 0 
                S_and_B = no_move_budget_constraint(cash_on_hand[s,1], Own, P, consumption[s,1], own[s,1], 
                0.0, debt[s,1],stock_market_entry[s,1], F, para)
            end 

            if moved[s,1] == 1 
                S_and_B =   move_budget_constraint(cash_on_hand[s,1], Own, P, consumption[s,1], own[s,1], 
                                        0.0, debt[s,1], stock_market_entry[s,1], F, para)
            end 
                                        
            stocks[s,1] = stock_share[s,1] *  S_and_B 
            bonds[s,1] = (1.0 - stock_share[s,1]) *  S_and_B       

            # Compute wealth 
            wealth[s,1] = stocks[s,1] + bonds[s,1] + P * own[s,1] - debt[s,1]

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
                    Inv_Move_shock[s,n] = 0 
                else 
                    Inv_Move_index = 2 
                    Inv_Move_shock[s,n] = 1
                end 

                # Turn the values of the choices last period to the states today.
                Own = own[s,n-1]
                Own_index = Own + 1 

                M_index = argmin(abs.(M_grid .- debt[s,n-1]))
                M = M_grid[M_index]

                IFC_index = IFC_prime_index

                # Compute cash on hand 
                P = P_bar * exp(b * (n) + p_grid[η_index])
                R_S = exp(stock_market_shock[s,n] + μ)
                income[s,n] = κ[n,2] * exp( η_grid[η_index] + ω_grid[ω_index])
                expected_earnings[s,n] = expected_earnings_vals[n,η_index] 

                cash_on_hand[s,n] = income[s,n] + R_S * stocks[s,n-1] + R_F * bonds[s,n-1] - R_D * M

                # Overall index 
                index = lin[Inv_Move_index, IFC_index, η_index, Own_index, M_index]

                # Compute choices 
                consumption[s,n] = c_interp_functions[index,n](cash_on_hand[s,n])

                # Compute whether the agent moved - adjusting so it is on grid. 
                moved[s,n] = floor(Move_interp_functions[index,n](cash_on_hand[s,n]))

                if moved[s,n] == 0
                    own[s,n] = own[s,n-1] 
                else 
                    own[s,n]  =  floor(Own_interp_functions[index,n](cash_on_hand[s,n])) 
                end 

                # Find mortgage level: forcing mortgage to 0 if renting 
                if own[s,n] == 0 
                    M_prime_index = 0 
                    debt[s,n] = 0.0 
                else 
                    # Place mortgage choice on the grid 
                    M_interp_val = M_interp_functions[index,n](cash_on_hand[s,n])
                    M_prime_index = argmin(abs.(M_grid .- M_interp_val))
                    debt[s,n] = M_grid[M_prime_index]
                end 

                # Need to adjust stock market entry so it is on the grid 
                stock_market_entry[s,n] = floor(FC_interp_functions[index,n](cash_on_hand[s,n]))
                if stock_market_entry[s,n] == 0 && IFC_index == 1
                    stock_share[s,n] = 0.0
                else 
                    stock_share[s,n] = α_interp_functions[index,n](cash_on_hand[s,n])
                end 
    
                # Find next period's indices
                IFC_prime_index = max(stock_market_entry[s,n],IFC_index - 1) + 1
                IFC_paid[s,n] = IFC_prime_index - 1

                # Compute savings 
                if moved[s,n] == 0 
                    S_and_B = no_move_budget_constraint(cash_on_hand[s,n], Own, P, consumption[s,n], own[s,n], 
                                        debt[s,n-1],debt[s,n], stock_market_entry[s,n], F, para)
                end 

                if moved[s,n] == 1 
                    S_and_B =   move_budget_constraint(cash_on_hand[s,n], Own, P, consumption[s,n], own[s,n], 
                                        debt[s,n-1],debt[s,n], stock_market_entry[s,n], F, para)
                end 
                                        
                stocks[s,n] = stock_share[s,n] *  S_and_B 
                bonds[s,n] = (1.0 - stock_share[s,n]) *  S_and_B      

                # Compute wealth 
                wealth[s,n] = stocks[s,n]  + bonds[s,n] + P * own[s,n] - debt[s,n]

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
                    Inv_Move_shock[s,n] = 0 
                else 
                    Inv_Move_index = 2 
                    Inv_Move_shock[s,n] = 1
                end 

                # Turn the choices last period to the states today.
                Own = own[s,n-1]
                Own_index = Own + 1 

                M_index = argmin(abs.(M_grid .- debt[s,n-1]))
                M = M_grid[M_index]

                IFC_index = IFC_prime_index

                # Compute cash on hand 
                P = P_bar * exp(b * (n) + p_grid[η_index])
                R_S = exp(stock_market_shock[s,n] + μ)
                income[s,n] = κ[n,2]
                expected_earnings[s,n] = expected_earnings_vals[n,η_index] 

                cash_on_hand[s,n] = income[s,n] + R_S * stocks[s,n-1] + R_F * bonds[s,n-1] - R_D_R * debt[s,n-1]
            
                if cash_on_hand[s,n] > X_max
                    println("Income is ",income[s,n], " cash on hand ",cash_on_hand[s,n], " bonds ", bonds[s,n-1], " stocks ", stocks[s,n-1])
                end 
                # Overall index 
                index = lin[Inv_Move_index, IFC_index, η_index, Own_index, M_index]

                # Compute choices 
                consumption[s,n] = c_interp_functions[index,n](cash_on_hand[s,n])

                # Compute whether the agent moved - adjusting so it is on grid. 
                moved[s,n] = floor(Move_interp_functions[index,n](cash_on_hand[s,n]))

                if moved[s,n] == 0
                    own[s,n] = own[s,n-1] 
                else 
                    own[s,n]  =  floor(Own_interp_functions[index,n](cash_on_hand[s,n])) 
                end 

                # Find mortgage level: forcing mortgage to 0 if renting 
                if own[s,n] == 0 
                    M_prime_index = 0 
                    debt[s,n] = 0 
                else 
                    # Place mortgage choice on the grid 
                    M_interp_val = M_interp_functions[index,n](cash_on_hand[s,n])
                    M_prime_index = argmin(abs.(M_grid .- M_interp_val))
                    debt[s,n] = M_grid[M_prime_index]
                end 

                # Need to adjust stock market entry so it is on the grid 
                stock_market_entry[s,n] = floor(FC_interp_functions[index,n](cash_on_hand[s,n]))
                if stock_market_entry[s,n] == 0 && IFC_index == 1
                    stock_share[s,n] = 0.0
                else 
                    stock_share[s,n] = α_interp_functions[index,n](cash_on_hand[s,n])
                end 
    
                # Find next period's indices
                IFC_prime_index = max(stock_market_entry[s,n],IFC_index - 1) + 1
                IFC_paid[s,n] = IFC_prime_index - 1

                # Compute savings 
                if moved[s,n] == 0 
                    S_and_B = no_move_budget_constraint(cash_on_hand[s,n], Own, P, consumption[s,n], own[s,n], 
                                        debt[s,n-1],debt[s,n], stock_market_entry[s,n],F, para)
                end 

                if moved[s,n] == 1 
                    S_and_B =   move_budget_constraint(cash_on_hand[s,n], Own, P, consumption[s,n], own[s,n], 
                                        debt[s,n-1],debt[s,n], stock_market_entry[s,n], F, para)
                end 
                                        
                stocks[s,n] = stock_share[s,n] *  S_and_B 
                bonds[s,n] = (1.0 - stock_share[s,n]) *  S_and_B  

                # Compute wealth 
                wealth[s,n] = stocks[s,n] + bonds[s,n] + P * own[s,n] - debt[s,n]

            end 

            # Simulate bequest
            η_index = rand(perm_dists[η_index]) # Draw the new permanent component based upon the old one. 
            ι_index = rand(stock_dist)

            # Save values of shocks 
            persistent[s,T+1] = η_grid[η_index]
            stock_market_shock[s,T+1] = ι_grid[ι_index]

            # Mortgage debt 
            M_index = argmin(abs.(M_grid .- debt[s,T]))
            M = M_grid[M_index]

            # Compute cash on hand 
            P = P_bar * exp(b * (T+1) + p_grid[η_index])
            R_S = exp(stock_market_shock[s,T+1] + μ)
            cash_on_hand[s,T+1] = R_S * stocks[s,T] + R_F * bonds[s,T] - R_D_R * M 
            
            # Agents are forced to sell their house when they die
            bequest[s] = cash_on_hand[s,T+1] - δ * own[s,T] * P +  (1-λ) *  P * own[s,T]   - M  
            age[s] = a
        end 
        sim_start = stop + 1
    end

    return bonds, stocks, stock_share, stock_market_entry, IFC_paid,
        own, moved,Inv_Move_shock, cash_on_hand,expected_earnings,
        debt, consumption, wealth, bequest, income,
        persistent, transitory, stock_market_shock
end

function sim_to_matrix(sim::Sim_Results)

    return hcat(
        sim.bonds,              sim.stocks,             sim.stock_share,
        sim.stock_market_entry, sim.IFC_paid,           sim.own,               
        sim.moved,              sim.Inv_Move_shock,     sim.cash_on_hand,      
        sim.expected_earnings,  sim.debt,                
        sim.consumption,        sim.wealth,
        sim.bequest,            sim.income,                       
        sim.persistent,         sim.transitory,        sim.stock_market_shock,
        sim.age
        )           
    
end