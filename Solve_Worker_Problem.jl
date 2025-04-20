# Solves the decision problem, outputs results back to the sols structure. 
function Solve_Worker_Problem(para::Model_Parameters, sols::Solutions)
    @unpack_Model_Parameters para 
    @unpack val_func, c_pol_func, H_pol_func, D_pol_func, FC_pol_func, α_pol_func = sols
    println("Solving the Worker's's Problem")

    println("Begin solving the model backwards")
    for j in TR-1:-1:1  # Backward induction

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
        Threads.@threads for X_index in 1:nX
            X = X_grid[X_index]

            # Loop over Cash-on-hand states
            for H_index in 1:nH
                H = H_grid[H_index]

                # Loop over aggregate income states
                for η_index in 1:nη
                    η = η_grid[η_index]
                    P = P_bar * exp(b * (j-1) + p_grid[η_index])

                    # Loop over whether the agent was forced to move 
                    for Inv_Move_index in 1:2
                        Inv_Move = Inv_Move_grid[Inv_Move_index]

                        # Loop over whether the agent has already paid their stock market entry cost 
                        for IFC_index in 1:2
                            IFC = IFC_grid[IFC_index]

                            candidate_max = -Inf  

                            # If an agent has already paid the stock market entry fee, they won't pay it again. 
                            if IFC == 1
                                FC_index = 1
                                FC = FC_grid[FC_index]

                                # Loop over Debt choices 
                                for D_index in 1:nD
                                    D = D_grid[D_index]

                                    # Loop over Housing choices 
                                    for H_prime_index in 1:nH 
                                        H_prime = H_grid[H_prime_index]
                                    
                                        # Skip if debt exceeds the collateral constraint. 
                                        if debt_constraint(D, H_prime, P, para) <= 0
                                            continue 
                                        end  

                                        # Loop over consumption choices 
                                        for c_index in 1:nc 
                                            c = c_grid[c_index]

                                            S_and_B  = budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)

                                            # Skip if implied stock and bond spending must be negative
                                            if S_and_B <= 0
                                                continue
                                            end

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
                                                            Y_Prime = κ[j, 2] * exp(η_prime + ω_prime)

                                                            # Compute next period's liquid wealth
                                                            X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime
                                                                
                                                            val += β * ((1-π) * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *
                                                                    interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (0 * 2) + FC + 1](X_prime) +
                                                                    π * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *
                                                                    interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (1 * 2) + FC + 1](X_prime)
                                                                    )
                                                        end 
                                                    end 
                                                end 
                                                # Update value function
                                                if val > candidate_max 
                                                    val_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]    = val
                                                    #println("Value is: ", val)

                                                    c_pol_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]  = c 
                                                    H_pol_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]  = H_prime
                                                    D_pol_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]  = D
                                                    FC_pol_func[j, H_index, X_index, η_index, Inv_Move_index,IFC_index] = FC
                                                    α_pol_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]  = α

                                                    candidate_max = val 
                                                end 
                                            end 
                                        end 
                                    end
                                end 
                            # They have not already paid the entry fee
                            else 

                                # Loop over Debt choices 
                                for D_index in 1:nD
                                    D = D_grid[D_index]

                                    # Loop over Housing choices 
                                    for H_prime_index in 1:nH 
                                        H_prime = H_grid[H_prime_index]
                                                                            
                                    # Skip if debt exceeds the collateral constraint. 
                                    if debt_constraint(D, H_prime, P, para) <= 0
                                        continue 
                                    end  

                                        # Loop over consumption choices 
                                        for c_index in 1:nc 
                                            c = c_grid[c_index]
                                    
                                            # Loop over enter/not enter choices 
                                            for FC_index in 1:2
                                                FC = FC_grid[FC_index]
                                    
                                                S_and_B  = budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)
                                    
                                                # Skip if implied stock and bond spending must be negative
                                                if S_and_B <= 0
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
                                                                Y_Prime = κ[j, 2] * exp(η_prime + ω_prime)

                                                                # Compute next period's liquid wealth
                                                                X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime
                                                                
                                                                val += β * ((1-π) * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *
                                                                    interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (0 * 2) + FC + 1](X_prime) +
                                                                     π * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *
                                                                    interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (1 * 2) + FC + 1](X_prime)
                                                                        )
                                                            end 
                                                        end 
                                                    end 
                                                                                        
                                                    # Update value function
                                                    if val > candidate_max 
                                                        val_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]    = val
                                                        #println("X is: ",X, " IFC is: ",IFC,"η is: ",η, " H is: ", H," Inv_Move is: ", H," Value is: ", val, )
                                                        c_pol_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]  = c 
                                                        H_pol_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]  = H_prime
                                                        D_pol_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]  = D
                                                        FC_pol_func[j, H_index, X_index, η_index, Inv_Move_index,IFC_index] = FC
                                                        α_pol_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]  = α
                                                        
                                                        candidate_max = val 
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
                                                                    Y_Prime = κ[j, 2] * exp(η_prime + ω_prime)

                                                                    # Compute next period's liquid wealth
                                                                    X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime
                                                                    
                                                                    val += β * ((1-π) * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *
                                                                        interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (0 * 2) + FC + 1](X_prime) +
                                                                        π * T_η[η_index, η_prime_index] * T_ω[1, ω_prime_index] * T_ι[1, ι_prime_index] *
                                                                        interp_functions[((H_prime_index - 1) * nη * 2 * 2) + ((η_prime_index - 1) * 2 * 2) + (1 * 2) + FC + 1](X_prime)
                                                                            )
                                                                end 
                                                            end 
                                                        end 
                                                        # Update value function
                                                        if val > candidate_max 
                                                            val_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]    = val
                                                            #println("X is: ",X, " IFC is: ",IFC,"η is: ",η, " H is: ", H," Inv_Move is: ", H," Value is: ", val)
                                                            c_pol_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]  = c 
                                                            H_pol_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]  = H_prime
                                                            D_pol_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]  = D
                                                            FC_pol_func[j, H_index, X_index, η_index, Inv_Move_index,IFC_index] = FC
                                                            α_pol_func[j, H_index, X_index, η_index, Inv_Move_index, IFC_index]  = α
                                                            candidate_max = val 
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
                end # η loop
            end # X Loop 
        end # H Loop
    end # T loop
end 