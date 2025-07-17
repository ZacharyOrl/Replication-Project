function simulate_model(para,sols,S::Int64, edu::Int64)
    # Simulates the solved model S times, returns assets, consumption, income, persistent shock and transitroy shock by age. 
    # Applies analysis weights based upon the education group. 

    @unpack_Model_Parameters para
    @unpack val_func,c_pol_func, LTV_pol_func, H_pol_func, FC_pol_func, α_pol_func, Move_pol_func, κ, σ_ω = sols
    
    # Set the number of simulations for this education group 
    wt_tot = sum(wts[:,edu])
    tot_sims = Int(round(S * wt_tot))

    # Generate Transitory earnings grid based on σ_w of group
    ω_grid::Vector{Float64} = tauchen_hussey(0.0,sqrt(σ_ω),0.0,g)[1] 
    T_ω::Matrix{Float64} = tauchen_hussey(0.0,sqrt(σ_ω),0.0,g)[2]
    nω::Int64 = length(ω_grid)

    expected_earnings_vals =  compute_expected_earnings(ω_grid,T_ω,κ, para)

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

    housing = zeros(tot_sims,T+1)
    moved = zeros(tot_sims,T+1)
    cash_on_hand = zeros(tot_sims,T+1)
    expected_earnings = zeros(tot_sims, T+1) 

    debt = zeros(tot_sims,T+1)
    LTV = zeros(tot_sims,T+1)

    consumption = zeros(tot_sims,T+1) 
    wealth = zeros(tot_sims,T+1) # Savings + Housing - Debt 
    bequest = zeros(tot_sims)

    income = zeros(tot_sims,T+1)
    persistent = zeros(tot_sims,T+1)
    transitory = zeros(tot_sims,T+1)
    stock_market_shock = zeros(tot_sims, T+1)
    education   = ones(tot_sims, T+1) * edu

    age = zeros(tot_sims)

    sim_start = 1

    # Age Loop 
    for a = 1:T 
        # Loop up to the number of simulations for the age education group 
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

            # Start with 0 assets, 0 debt, 0 housing, not having entered the stock market
            H_index = 1 
            IFC_index = 1

            income[s,1] = κ[1,2] * exp(η_grid[η_index] + ω_grid[ω_index])

            X   = 0.0 + income[s,1]
            cash_on_hand[s,1] = round_to_grid(X, X_grid)
            X_index =  searchsortedlast(X_grid, cash_on_hand[s,1])

            # Check 
            if X > X_max
                println(cash_on_hand[s,1])
            end 

            # This is purely an input into the budget constraint and so is not saved.
            H = H_grid[H_index]

            # Compute choices 
            consumption[s,1] = c_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
            LTV[s,1] = LTV_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]

            # Compute whether the agent moved - adjusting so it is on grid. 
            moved[s,1] = 1.0

            housing[s,1]  =  H_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
            H_prime_index = searchsortedlast(H_grid[1:nH], housing[s,1])

            stock_market_entry[s,1] = FC_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
            stock_share[s,1] = α_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
    
            # Find next period's indices
            IFC_prime_index = max(stock_market_entry[s,1],IFC_index - 1) + 1
            IFC_paid[s,1] = IFC_prime_index - 1

            # Compute savings 

            # Everyone moves in the first period 
            S_and_B =   move_budget_constraint(cash_on_hand[s,1], H, P, consumption[s,1], housing[s,1], 
                                        LTV[s,1], stock_market_entry[s,1], para)
                       
            stocks[s,1] = stock_share[s,1] *  S_and_B 
            bonds[s,1] = (1.0 - stock_share[s,1]) *  S_and_B       

            debt[s,1] = P * LTV[s,1] * housing[s,1]

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
                    Inv_Move_shock[s,n] = 0 
                else 
                    Inv_Move_index = 2 
                    Inv_Move_shock[s,n] = 1
                end 

                # Turn the values of the choices last period to the states today.
                H = housing[s,n-1]
                H_index = H_prime_index

                IFC_index = IFC_prime_index

                # Compute cash on hand 
                P = P_bar * exp(b * (n) + p_grid[η_index])
                R_S = exp(stock_market_shock[s,n] + μ)
                income[s,n] = κ[n,2] * exp( η_grid[η_index] + ω_grid[ω_index])
                expected_earnings[s,n] = expected_earnings_vals[n,η_index] 

                X = income[s,n] + R_S * stocks[s,n-1] + R_F * bonds[s,n-1] - R_D * debt[s,n-1]
                cash_on_hand[s,n] = round_to_grid(X,X_grid)
                X_index =  searchsortedlast(X_grid, cash_on_hand[s,n])

                # Compute choices 
                consumption[s,n] = c_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                LTV[s,n] = LTV_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                moved[s,n] = Move_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]

                housing[s,n]  =  H_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                H_prime_index = searchsortedlast(H_grid, housing[s,n])

                # Need to adjust stock market entry so it is on the grid 
                stock_market_entry[s,n] = FC_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]

                stock_share[s,n] = α_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                
                # Make sure no investors are entering the stock market without paying the fee  
                if  stock_market_entry[s,n] == 0 && IFC_paid[s,n-1] == 0 && stock_share[s,n] > 0 
                    println("Error in stock choice")
                end 

                # Find next period's indices
                IFC_prime_index = max(stock_market_entry[s,n],IFC_index - 1) + 1
                IFC_paid[s,n] = IFC_prime_index - 1

                # Compute savings 
                if moved[s,n] == 0 
                    S_and_B = no_move_budget_constraint(cash_on_hand[s,n], H, P, consumption[s,n], housing[s,n], 
                                        LTV[s,n], stock_market_entry[s,n], para)
                end 

                if moved[s,n] == 1 
                    S_and_B =   move_budget_constraint(cash_on_hand[s,n], H, P, consumption[s,n], housing[s,n], 
                                        LTV[s,n], stock_market_entry[s,n], para)
                end 
                                        
                stocks[s,n] = stock_share[s,n] *  S_and_B 
                bonds[s,n] = (1.0 - stock_share[s,n]) *  S_and_B      
                
                debt[s,n] = P * LTV[s,n] * housing[s,n]

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
                    Inv_Move_shock[s,n] = 0 
                else 
                    Inv_Move_index = 2 
                    Inv_Move_shock[s,n] = 1
                end 

                # Turn the choices last period to the states today.
                H = housing[s, n-1]
                H_index = H_prime_index

                IFC_index = IFC_prime_index

                # Compute cash on hand 
                P = P_bar * exp(b * (n) + p_grid[η_index])
                R_S = exp(stock_market_shock[s,n] + μ)
                income[s,n] = κ[n,2]
                expected_earnings[s,n] = expected_earnings_vals[n,η_index]

                X = income[s,n] + R_S * stocks[s,n-1] + R_F * bonds[s,n-1] - R_D * debt[s,n-1]
                cash_on_hand[s,n] = round_to_grid(X,X_grid)
                X_index =  searchsortedlast(X_grid, cash_on_hand[s,n])
                
                # Check 
                if X > X_max
                    println(" X too high, Income is ",income[s,n], " cash on hand ",cash_on_hand[s,n], " bonds ", bonds[s,n-1], " stocks ", stocks[s,n-1])
                end 

                # Compute choices 
                consumption[s,n] = c_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                LTV[s,n] = LTV_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]

                moved[s,n] = Move_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                housing[s,n]  =  H_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                H_prime_index = searchsortedlast(H_grid, housing[s,n])

                stock_market_entry[s,n] = FC_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                stock_share[s,n] = α_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]

                # Find next period's indices
                IFC_prime_index = max(stock_market_entry[s,n],IFC_index - 1) + 1
                IFC_paid[s,n] = IFC_prime_index - 1

                # Compute savings 
                if moved[s,n] == 0 
                    S_and_B = no_move_budget_constraint(cash_on_hand[s,n], H, P, consumption[s,n], housing[s,n], 
                                        LTV[s,n], stock_market_entry[s,n], para)
                end 

                if moved[s,n] == 1 
                    S_and_B =   move_budget_constraint(cash_on_hand[s,n], H, P, consumption[s,n], housing[s,n], 
                                        LTV[s,n], stock_market_entry[s,n], para)
                end 
                                        
                stocks[s,n] = stock_share[s,n] *  S_and_B 
                bonds[s,n] = (1.0 - stock_share[s,n]) *  S_and_B       

                debt[s,n] = P * housing[s,n] * LTV[s,n]

                # Compute wealth 
                wealth[s,n] = stocks[s,n] + bonds[s,n] + P * housing[s,n] - debt[s,n]

                # Check to see if 
                if X > X_max
                    println(" X too high, Income is ",income[s,n], " cash on hand ",cash_on_hand[s,n], " bonds ", bonds[s,n-1], " stocks ", stocks[s,n-1])
                end 


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
            X = R_S * stocks[s,T] + R_F * bonds[s,T] - R_D * debt[s,T]
            cash_on_hand[s,T+1] = round_to_grid(X,X_grid)
            
            # Agents are forced to sell their house when they die
            bequest[s] = cash_on_hand[s,T+1] - δ * housing[s,T] * P +  (1-λ) *  P * housing[s,T]    
            age[s] = a
        end 
        sim_start = stop + 1
    end

    return Sim_Results(
        filter_age(bonds,age), filter_age(stocks,age), filter_age(stock_share,age), filter_age(stock_market_entry,age), filter_age(IFC_paid,age),
        filter_age(housing,age), filter_age(moved,age),filter_age(Inv_Move_shock,age), filter_age(cash_on_hand,age),filter_age(expected_earnings,age),
        filter_age(debt,age), filter_age(LTV,age), filter_age(consumption,age), filter_age(wealth,age), bequest, filter_age(income,age),
        filter_age(persistent,age), filter_age(transitory,age), filter_age(stock_market_shock,age), 
        age, filter_age(education,age))
end

# Same as the simulation except shocks are constant for cohorts in the same year. 
# Essentially a repeat of simulate_model except the persistent shocks are not randomly drawn. 
function simulate_model_constant_shocks(para,sols,S::Int64, edu::Int64)

    @unpack ι_grid, η_grid,p_grid, H_grid, FC_grid, Move_grid, X_grid,  T_η, T_ι, nη, nH, π_η, T, TR, π_m, P_bar, b, μ, R_F, R_D, δ, λ, g, X_max, X_min, H_min, lin, wts = para
    @unpack val_func,c_pol_func, LTV_pol_func, H_pol_func, FC_pol_func, α_pol_func, Move_pol_func, κ, σ_ω = sols
    
    # Set the number of simulations for this education group 
    wt_tot = sum(wts[:,edu])
    tot_sims = Int(round(S * wt_tot))

    # Generate Transitory earnings grid based on σ_w of group
    ω_grid::Vector{Float64} = tauchen_hussey(0.0,σ_ω,0.0,g)[1] 
    T_ω::Matrix{Float64} = tauchen_hussey(0.0,σ_ω,0.0,g)[2]
    nω::Int64 = length(ω_grid)

    expected_earnings_vals =  compute_expected_earnings(ω_grid,T_ω,κ, para)
    # Education group

    # Distribution over the transitory component (use that it isn't persistent, so won't vary over time)
    transitory_dist = Categorical(T_ω[1,:])

    # Distribution over Stock Market Shock 
    stock_dist = Categorical(T_ι[1,:])

    # Generate indices of the persistent shocks experienced in each five year group 
    # for cohorts from age 25 in 1985-89 to age 70 in 1985-89.
    # 1970-1989 shocks are coded to match the house prices time series 
    # pre - 1970 shocks are set at 0.0. 
    perm_dists = generate_aggregate_ts(para)

    # Stationary distribution over initial aggregate state
    initial_dist = Categorical(π_η)

    # Outputs
    bonds = zeros(tot_sims,T+1) 
    stocks = zeros(tot_sims,T+1) 
    stock_share = zeros(tot_sims, T+1)
    stock_market_entry = zeros(Int64, tot_sims,T+1)
    IFC_paid = zeros(tot_sims,T+1)
    Inv_Move_shock = zeros(tot_sims,T+1)

    housing = zeros(tot_sims,T+1)
    moved = zeros(tot_sims,T+1)
    cash_on_hand = zeros(tot_sims,T+1)
    expected_earnings = zeros(tot_sims, T+1) 

    debt = zeros(tot_sims,T+1)
    LTV = zeros(tot_sims,T+1)

    consumption = zeros(tot_sims,T+1) 
    wealth = zeros(tot_sims,T+1) # Savings + Housing - Debt 
    bequest = zeros(tot_sims)

    income = zeros(tot_sims, T+1)
    persistent = zeros(tot_sims,T+1)
    transitory = zeros(tot_sims,T+1)
    stock_market_shock = zeros(tot_sims, T+1)
    education   = ones(tot_sims, T+1) * edu

    age = zeros(tot_sims)

    sim_start = 1
    for a = 1:T
        stop = Int(sim_start + round(S*wts[a,edu]) - 1) 
        for s = sim_start:stop
            η_index = perm_dists[a,1]
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

            # Start with 0 assets, 0 debt, 0 housing, not having entered the stock market
            H_index = 1 
            IFC_index = 1

            income[s,1] = κ[1,2] * exp(η_grid[η_index] + ω_grid[ω_index])
            X   = 0.0 + income[s,1]
            cash_on_hand[s,1] = round_to_grid(X, X_grid)
            X_index =  searchsortedlast(X_grid, cash_on_hand[s,1])

            # Check 
            if X > X_max
                println(cash_on_hand[s,1])
            end 

            # This is purely an input into the budget constraint and so is not saved.
            H = H_grid[H_index]

            # Compute choices 
            consumption[s,1] = c_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
            LTV[s,1] = LTV_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]

            # Compute whether the agent moved - adjusting so it is on grid. 
            moved[s,1] = 1.0

            housing[s,1]  =  H_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
            H_prime_index = searchsortedlast(H_grid[1:nH], housing[s,1])

            stock_market_entry[s,1] = FC_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
            stock_share[s,1] = α_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
    
            # Find next period's indices
            IFC_prime_index = max(stock_market_entry[s,1],IFC_index - 1) + 1
            IFC_paid[s,1] = IFC_prime_index - 1

            # Compute savings 

            if moved[s,1] == 1 
                S_and_B =   move_budget_constraint(cash_on_hand[s,1], H, P, consumption[s,1], housing[s,1], 
                                        LTV[s,1], stock_market_entry[s,1], para)
            end 
                                        
            stocks[s,1] = stock_share[s,1] *  S_and_B 
            bonds[s,1] = (1.0 - stock_share[s,1]) *  S_and_B       

            debt[s,1] = P * LTV[s,1] * housing[s,1]

            # Compute wealth 
            wealth[s,1] = stocks[s,1] + bonds[s,1] + P * housing[s,1] - debt[s,1]

            # Simulate working age 
            for n = 2:TR - 1  
                # Draw new values for the labor income shocks 
                η_index = perm_dists[a,n]
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
                H = housing[s,n-1]
                H_index = H_prime_index 
                IFC_index = IFC_prime_index

                # Compute cash on hand 
                P = P_bar * exp(b * (n) + p_grid[η_index])
                R_S = exp(stock_market_shock[s,n] + μ)
                income[s,n] = κ[n,2] * exp( η_grid[η_index] + ω_grid[ω_index])
                expected_earnings[s,n] = expected_earnings_vals[n,η_index] 

                X = income[s,n] + R_S * stocks[s,n-1] + R_F * bonds[s,n-1] - R_D * debt[s,n-1]
                cash_on_hand[s,n] = round_to_grid(X,X_grid)
                X_index =  searchsortedlast(X_grid, cash_on_hand[s,n])

                # Compute choices 
                consumption[s,n] = c_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                LTV[s,n] = LTV_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                moved[s,n] = Move_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]

                housing[s,n]  =  H_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                H_prime_index = searchsortedlast(H_grid, housing[s,n])

                # Need to adjust stock market entry so it is on the grid 
                stock_market_entry[s,n] = FC_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]

                stock_share[s,n] = α_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                
                # Make sure no investors are entering the stock market without paying the fee  
                if  stock_market_entry[s,n] == 0 && IFC_paid[s,n-1] == 0 && stock_share[s,n] > 0 
                    println("Error in stock choice")
                end 
    
                # Find next period's indices
                IFC_prime_index = max(stock_market_entry[s,n],IFC_index - 1) + 1
                IFC_paid[s,n] = IFC_prime_index - 1

                # Compute savings 
                if moved[s,n] == 0 
                    S_and_B = no_move_budget_constraint(cash_on_hand[s,n], H, P, consumption[s,n], housing[s,n], 
                                        LTV[s,n], stock_market_entry[s,n], para)
                end 

                if moved[s,n] == 1 
                    S_and_B =   move_budget_constraint(cash_on_hand[s,n], H, P, consumption[s,n], housing[s,n], 
                                        LTV[s,n], stock_market_entry[s,n], para)
                end 
                                        
                stocks[s,n] = stock_share[s,n] *  S_and_B 
                bonds[s,n] = (1.0 - stock_share[s,n]) *  S_and_B      
                
                debt[s,n] = P * LTV[s,n] * housing[s,n]

                # Compute wealth 
                wealth[s,n] = stocks[s,n]  + bonds[s,n] + P * housing[s,n] - debt[s,n]

            end 

            # Simulate retirement
            for n = TR:T
                # Draw new values for the labor income shocks 
                η_index = perm_dists[a,n]
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
                H = housing[s, n-1]
                H_index = H_prime_index 

                IFC_index = IFC_prime_index

                # Compute cash on hand 
                P = P_bar * exp(b * (n) + p_grid[η_index])
                R_S = exp(stock_market_shock[s,n] + μ)
                income[s,n] = κ[n,2]
                expected_earnings[s,n] = expected_earnings_vals[n,η_index] 
                
                X = income[s,n] + R_S * stocks[s,n-1] + R_F * bonds[s,n-1] - R_D * debt[s,n-1]
                cash_on_hand[s,n] = round_to_grid(X,X_grid)
                X_index =  searchsortedlast(X_grid, cash_on_hand[s,n])
                
                # Check 
                if X > X_max
                    println(" X too high, Income is ",income[s,n], " cash on hand ",cash_on_hand[s,n], " bonds ", bonds[s,n-1], " stocks ", stocks[s,n-1])
                end 

                # Compute choices 
                consumption[s,n] = c_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                LTV[s,n] = LTV_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]

                moved[s,n] = Move_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                housing[s,n]  =  H_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                H_prime_index = searchsortedlast(H_grid, housing[s,n])

                stock_market_entry[s,n] = FC_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
                stock_share[s,n] = α_pol_func[Inv_Move_index, IFC_index, η_index, H_index, X_index, n]
    
                # Find next period's indices
                IFC_prime_index = max(stock_market_entry[s,n],IFC_index - 1) + 1
                IFC_paid[s,n] = IFC_prime_index - 1

                # Compute savings 
                if moved[s,n] == 0 
                    S_and_B = no_move_budget_constraint(cash_on_hand[s,n], H, P, consumption[s,n], housing[s,n], 
                                        LTV[s,n], stock_market_entry[s,n], para)
                end 

                if moved[s,n] == 1 
                    S_and_B =   move_budget_constraint(cash_on_hand[s,n], H, P, consumption[s,n], housing[s,n], 
                                        LTV[s,n], stock_market_entry[s,n], para)
                end 
                                        
                stocks[s,n] = stock_share[s,n] *  S_and_B 
                bonds[s,n] = (1.0 - stock_share[s,n]) *  S_and_B       

                debt[s,n] = P * housing[s,n] * LTV[s,n]

                # Compute wealth 
                wealth[s,n] = stocks[s,n] + bonds[s,n] + P * housing[s,n] - debt[s,n]

            end 

            # Simulate bequest
            # η_index = rand(perm_dists[1,1])   # No need to draw as bequest isn't an analysis object. 
            ι_index = rand(stock_dist)

            # Save values of shocks 
            persistent[s,T+1] = persistent[s,T]
            stock_market_shock[s,T+1] = ι_grid[ι_index]

            # Compute cash on hand 
            P = P_bar * exp(b * (T+1) + p_grid[η_index])
            R_S = exp(stock_market_shock[s,T+1] + μ)

            X = R_S * stocks[s,T] + R_F * bonds[s,T] - R_D * debt[s,T]
            cash_on_hand[s,T+1] = round_to_grid(X,X_grid)
            
            # Agents are forced to sell their house when they die
            bequest[s] = cash_on_hand[s,T+1] - δ * housing[s,T] * P +  (1-λ) *  P * housing[s,T]    

            age[s] = a
            
        end 

        sim_start = stop + 1
    end 

    return Sim_Results(
        filter_age(bonds,age), filter_age(stocks,age), filter_age(stock_share,age), filter_age(stock_market_entry,age), filter_age(IFC_paid,age),
        filter_age(housing,age), filter_age(moved,age),filter_age(Inv_Move_shock,age), filter_age(cash_on_hand,age),filter_age(expected_earnings,age),
        filter_age(debt,age), filter_age(LTV,age), filter_age(consumption,age), filter_age(wealth,age), bequest, filter_age(income,age),
        filter_age(persistent,age), filter_age(transitory,age), filter_age(stock_market_shock,age), 
        age, filter_age(education,age))
end
