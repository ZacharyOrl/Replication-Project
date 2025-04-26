# Solves the decision problem, outputs results back to the sols structure. 
function Solve_Retiree_Problem(para::Model_Parameters, sols::Solutions)
    @unpack_Model_Parameters para 
    @unpack val_func, c_pol_func, H_pol_func, D_pol_func, FC_pol_func, α_pol_func = sols
    println("Solving the Retiree's Problem")

    # Compute the bequest value of wealth
    val_func[:, :, :, :, :, T+1] = compute_bequest_value(val_func[:, :, :, :, :, T+1], para)

    println("Begin solving the model backwards")
    for j in T:-1:TR  # Backward induction

        println("Age is ", 20 + 5*j)
        
       # Generate interpolation functions for cash-on hand given each possible combination of the other states tomorrow 
        interp_functions = Vector{Any}(undef, 2 * 2 * nη * nH) 
        for Inv_Move_index in 1:2
            for IFC_index in 1:2
                for η_index in 1:nη
                    for H_index in 1:nH
                        # Compute linear index 
                        index =  (H_index - 1) * (η_index * 4) + (η_index - 1) * 4 + (Inv_Move_index - 1) * 2 + (IFC_index - 1) + 1
                         # Access val_func with dimensions [Inv_Move, IFC, η, H, X, j]
                        interp_functions[index] = linear_interp(val_func[Inv_Move_index, IFC_index, η_index, H_index, :, j+1], X_grid)
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

                            candidate_max = pun

                            # If an agent has already paid the stock market entry fee, they won't pay it again. 
                            if IFC == 1
                                FC_index = 1
                                FC = FC_grid[FC_index]

                                # Loop over Housing choices 
                                for H_prime_index in 1:nH 
                                    H_prime = H_grid[H_prime_index]

                                    # Loop over Risky-share choices
                                    for α_index in 1:nα 
                                        α = α_grid[α_index]

                                        # Debt and bills are perfect substitutes. If you are already in the stock market, any α < 1 implies D = 0.0 
                                        if  α < 1.0
                                            D  = 0.0 
                                            c  = optimize_retiree_c(j, H, P, X, η_index, Inv_Move, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, para)
                                            val = compute_retiree_value(j, H, P,  X, η_index, Inv_Move, c, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, para)
                                        else
                                            D, c, val = optimize_retiree_d(j, H, P, X, η_index, Inv_Move, α, H_prime, H_prime_index, IFC, FC, interp_functions, para)
                                        end 

                                        # Update value function
                                        if val > candidate_max 
                                            val_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]     = val

                                            c_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = c 
                                            H_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = H_prime
                                            D_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = D
                                            FC_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]  = FC
                                            α_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = α

                                            candidate_max = val 
                                        end 


                                    end
                                end 
                            end 
                            # They have not already paid the entry fee
                            if IFC == 0

                                # Loop over enter/not enter choices 
                                for FC_index in 1:2
                                    FC = FC_grid[FC_index]

                                    # Loop over Housing choices 
                                    for H_prime_index in 1:nH 
                                        H_prime = H_grid[H_prime_index]                                
                                    
                                        # If the person has not paid the entry cost and does not pay the entry cost today 
                                        # they must invest 0 in stocks. 
                                        if FC == 0 
                                            α_index = 1 
                                            α = α_grid[α_index]
                                    
                                            D, c, val = optimize_retiree_d(j, H, P, X, η_index, Inv_Move, α, H_prime, H_prime_index, IFC, FC, interp_functions, para)
                                                                                        
                                            # Update value function
                                            if val > candidate_max 
                                                val_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]     = val

                                                c_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = c 
                                                H_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = H_prime
                                                D_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = D
                                                FC_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]  = FC
                                                α_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = α
                                                candidate_max = val 
                                            end 
                                        # If FC == 1 && IFC == 0 
                                        else 
                                        # Loop over Risky-share choices
                                            for α_index in 1:nα 
                                                α = α_grid[α_index]

                                                # Debt and bills are perfect substitutes. 
                                                # If α is not equal to zero or 1, then you must hold positive values of both stocks and bonds and hence have no debt. 

                                                if  α < 1.0
                                                    D  = 0.0 
                                                    c  = optimize_retiree_c(j, H, P, X, η_index, Inv_Move, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, para)
                                                    val = compute_retiree_value(j, H, P,  X, η_index, Inv_Move, c, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, para)
                                                else
                                                    D, c, val = optimize_retiree_d(j, H, P, X, η_index, Inv_Move, α, H_prime, H_prime_index, IFC, FC, interp_functions, para)
                                                end 

                                                # Update value function
                                                if val > candidate_max 
                                                    val_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]     = val
                                                    #println("X is: ",X, " IFC is: ",IFC,"η is: ",η, " H is: ", H," Inv_Move is: ", H," Value is: ", val)
                                                    c_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = c 
                                                    H_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = H_prime
                                                    D_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = D
                                                    FC_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]  = FC
                                                    α_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = α
                                        
                                                    candidate_max = val 
                                                end 
                                            end 
                                        end 
                                    end 
                                end 
                            end

                            if candidate_max <= pun
                                val = pun
                                val_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j] = val
                            end 
                        end 
                    end 
                end # η loop
            end  # X Loop 
        end # H Loop
    end # T loop
end


######################################
# Optimization Functions 
######################################
# Compute expectation function given both c and D
function compute_retiree_value(j::Int64, H::Float64, P::Float64, X::Float64, η_index::Int64, Inv_Move::Int64, c::Float64, α::Float64, H_prime::Float64, H_prime_index::Int64, D::Float64, IFC::Int64, FC::Int64, interp_functions::Vector{Any}, para::Model_Parameters )
    @unpack_Model_Parameters para
    S_and_B = budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)

    IFC_prime_index = max(IFC,FC) + 1
    # Compute Stock and Bond Positions 
    S = α * S_and_B 
    B = (1-α) * S_and_B 

    val = flow_utility_func(c, H, para)

    # Labor Income tomorrow
    Y_Prime = κ[j+1, 2]

    # Find the continuation value 
    # Loop over random variables 
    for ι_prime_index in 1:nι
        ι_prime = ι_grid[ι_prime_index]

        R_prime = exp(ι_prime + μ)

        # Compute next period's liquid wealth
        X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime
        for η_prime_index in 1:nη
            val += ( β * ( (1-π) * T_η[η_index, η_prime_index]  * T_ι[1, ι_prime_index] *
                    interp_functions[(H_prime_index - 1) * (η_prime_index * 4) + (η_prime_index - 1) * 4 + (1 - 1) * 2 + (IFC_prime_index - 1) + 1](X_prime) +
                        π * T_η[η_index, η_prime_index]   * T_ι[1, ι_prime_index] *
                    interp_functions[(H_prime_index - 1) * (η_prime_index * 4) + (η_prime_index - 1) * 4 + (2 - 1) * 2 + (IFC_prime_index - 1) + 1](X_prime)   
                    )   )
            
        end 
    end 

   return val 
end 

# Optimize value function over c given a choice of D
function optimize_retiree_c(j::Int64, H::Float64, P::Float64, X::Float64, η_index::Int64, Inv_Move::Int64, α::Float64, H_prime::Float64, H_prime_index::Int64, 
                            D::Float64, IFC::Int64, FC::Int64, interp_functions::Vector{Any}, para::Model_Parameters)
    @unpack_Model_Parameters para
    # Find maximum feasible consumption
    budget(c) =  budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)
    c_max_case = find_zero(budget, X, Roots.Order1())

    if c_max_case < 0
        return (c_opt = 0.0)

    else
        # Optimize using Brent's method
        result = optimize(c -> -compute_retiree_value(j, H, P,  X, η_index, Inv_Move, c, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, para), 0.0, c_max_case, Brent())
        
        return (c_opt = Optim.minimizer(result))
    end 
end

# Find the value of the problem given a choice of D taking the optimizing choice of c function as given. 
function objective_D(D::Float64, j::Int64, H::Float64, P::Float64, X::Float64, η_index::Int64, Inv_Move::Int64, α::Float64, H_prime::Float64, 
                        H_prime_index::Int64, IFC::Int64, FC::Int64, interp_functions::Vector{Any}, para::Model_Parameters)
    @unpack_Model_Parameters para

    # Get optimal c for this D
    c = optimize_retiree_c(j, H, P, X, η_index, Inv_Move,α, H_prime, H_prime_index, D, IFC, FC, interp_functions, para)
    
    # Compute value
    val = compute_retiree_value(j, H, P,  X, η_index, Inv_Move, c, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, para)
    return val  # Negative for maximization
end

function optimize_retiree_d(j::Int64, H::Float64, P::Float64, X::Float64, η_index::Int64, Inv_Move::Int64, α::Float64, H_prime::Float64, H_prime_index::Int64, 
                             IFC::Int64, FC::Int64, interp_functions::Vector{Any}, para::Model_Parameters)

    @unpack_Model_Parameters para
    # Find maximum feasible consumption
    debt_limit(D) =  debt_constraint(D, H_prime, P, para)
    D_max_case = find_zero(debt_limit, X, Roots.Order1())

    if D_max_case < 0
        return (D_opt = 0.0,c_opt = 0.0, val_opt = pun)
    else
    
    # Optimize using Brent's method
    result = optimize(D -> -objective_D(D, j, H, P,  X, η_index, Inv_Move, α, H_prime, H_prime_index, IFC, FC, interp_functions, para), 0.0, D_max_case, Brent())
    D_opt = Optim.minimizer(result)

    c_opt = optimize_retiree_c(j, H, P, X, η_index, Inv_Move,α, H_prime, H_prime_index, D_opt, IFC, FC, interp_functions, para)
    val_opt = -Optim.minimum(result)
        return (D_opt, c_opt, val_opt)
    end 
end