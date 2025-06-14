# Solves the decision problem, outputs results back to the sols structure. 
function Solve_Retiree_Problem(para::Model_Parameters, sols::Solutions)
    @unpack_Model_Parameters para 
    @unpack val_func, c_pol_func, M_pol_func, FC_pol_func, α_pol_func, Move_pol_func, Own_pol_func, κ, s, F = sols
    
    #println("Solving the Retiree's Problem")

    # Compute the bequest value of wealth
    val_func[:, :, :, :, :, :, T+1] = compute_bequest_value(val_func[:, :, :, :, :, :, T+1], para)

    #println("Begin solving the model backwards")
    for j in T:-1:TR  # Backward induction

        #println("Age is ", 25 - 50/T + (50/T)*j)
        
       # Generate interpolation functions for cash-on hand given each possible combination of the other states tomorrow 
        interp_functions = Vector{Any}(undef, 2 * 2 * nη * 2 * nM) 

        for Inv_Move_index in 1:2
            for IFC_index in 1:2
                for η_index in 1:nη
                    for Own_index in 1:2
                        for M_index in 1:nM
                            # Compute linear index 
                            index = lin[Inv_Move_index, IFC_index, η_index, Own_index, M_index]
                            # Access val_func with dimensions [Inv_Move, IFC, η, Own, X, j]
                            interp_functions[index] =  linear_interp(val_func[Inv_Move_index, IFC_index, η_index, Own_index, M_index, :, j+1], X_grid)
                        end 
                    end
                end
            end
        end

        # Loop over cash on hand states
        Threads.@threads for X_index in 1:nX
            X = X_grid[X_index]

            # Loop over current mortgage debt 
            for M_index in 1:nM
                M = M_grid[M_index]

                # Loop over housing states
                for Own_index in 1:2
                    Own = Own_grid[Own_index]

                    # Loop over aggregate income states
                    for η_index in 1:nη
                        η = η_grid[η_index]
                        P = P_bar * exp(b * (j) + p_grid[η_index])

                        # Loop over whether the agent was forced to move 
                        for Inv_Move_index in 1:2
                            Inv_Move = Inv_Move_grid[Inv_Move_index]

                            # Loop over whether the agent has already paid their stock market entry cost 
                            for IFC_index in 1:2
                                IFC = IFC_grid[IFC_index]

                                candidate_max = pun

                                # Loop over M_prime choices
                                for M_prime_index = 1:nM 
                                    M_prime = M_grid[M_prime_index]

                                    # Loop over stock share choices
                                    for α_index = 1:nα
                                        α = α_grid[α_index]

                                        # Loop over stock market entry choices 
                                        for FC_index = 1:2 
                                            FC = FC_grid[FC_index]

                                            # Loop over Rent or Own choice 
                                            for Own_prime_index = 1:2 
                                                Own_prime = Own_grid[Own_prime_index]

                                                # Impose conditions 

                                                # Check for violations of the borrowing constraint: 
                                                if mortgage_constraint(M_index, M_prime_index, Own_index, Own_prime_index, P, para) < 0 
                                                    continue 
                                                end 

                                                # If the agent has already entered the stock market, they won't do it again. 
                                                if IFC == 1 && FC == 1
                                                    continue 
                                                end 

                                                # Not possible to have positive stock market share if not in the stock market. 
                                                if IFC == 0 && FC == 0 && α > 0.0 
                                                    continue 
                                                end 

                                                if Inv_Move == 0 && Own == Own_prime
                                                    c,val = optimize_retiree_no_move(j, Own, P, X, η_index, α, M_index, M_prime_index, IFC,FC, κ, interp_functions, s, F, para)
                                                    Move = 0 
                                                end 

                                                if Inv_Move == 0 && Own != Own_prime
                                                    c,val = optimize_retiree_move(j, Own, Own_prime, P, X, η_index, α, M_index, M_prime_index, IFC,FC, κ, interp_functions, s, F, para)
                                                    Move = 1
                                                end 

                                                # If Inv_Mov == 1 then the agent must optimize conditional on moving only
                                                if Inv_Move == 1 
                                                    c,val = optimize_retiree_move(j, Own, Own_prime, P, X, η_index, α, M_index, M_prime_index, IFC,FC, κ, interp_functions, s, F, para) 
                                                    # The agent's optimal choice is always to move in this case. 
                                                    Move = 1
                                                end 
                                                    
                                                # Update value function
                                                if val > candidate_max 
                                                    val_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]     = val
                                                    #println("Value is: ", val)

                                                    c_pol_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]   = c 
                                                    M_pol_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]   = M_prime
                                                    FC_pol_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]  = FC
                                                    α_pol_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]   = α
                                                    Move_pol_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]   = Move
                                                    Own_pol_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]   = Own_prime
                                                    #println( " X", X, " Own ", Own, " IFC ", IFC, " c " ,c," Own_prime ",Own_prime)

                                                    candidate_max = val 
                                                end 
                                            end 
                                        end
                                    end
                                end
                                 
                                if candidate_max <= pun
                                    val_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j] = pun
                                end 
                                 
                            end # IFC Loop
                        end # Inv_Move loop
                    end  # η loop
                end # Own Loop
            end # M Loop 
        end # X-Loop
    end # T loop
end 

######################################
# Optimization Functions 
######################################
###############################################################################
# 1. VALUE FUNCTION
###############################################################################
function retiree_value(j::Int, Own::Int64, P::Float64, X::Float64,
                       η_index::Int, c::Float64, α::Float64, Own_prime::Int64, M_index::Int64,
                       M_prime_index::Int64, IFC::Int, FC::Int,
                       κ::Matrix{Any}, interp_functions::Vector{Any},
                       constraint::Function, s::Float64, F::Float64, para::Model_Parameters)

    @unpack_Model_Parameters para 

    M = M_grid[M_index]
    M_prime = M_grid[M_prime_index]

    S_and_B = constraint(X, Own, P, c, Own_prime, M, M_prime, FC,F, para)

    Own_prime_index = Own_prime + 1

    IFC_prime_index = max(IFC, FC) + 1          # 1 = no stock entry, 2 = entered

    S  = α       * S_and_B
    B  = (1 - α) * S_and_B

    val = flow_utility_func(c, Own_prime, s, para)

    # labour‐income next period (κ holds exogenous paths)
    Y_prime = κ[j + 1, 2]

    R_prime_max = exp(ι_grid[g] + μ)
    R_prime_min = exp(ι_grid[1]  + μ)

    X_prime_ub  = R_prime_max * S + R_F * B - R_D_R * M_prime + Y_prime
    X_prime_lb  = R_prime_min * S + R_F * B - R_D_R * M_prime + Y_prime

    if X_prime_lb < X_min || X_prime_ub > X_max || S_and_B < 0
        return pun
    end

    # continuation value
    for ι_prime_index in 1:nι
        ι_prime  = ι_grid[ι_prime_index]
        R_prime  = exp(ι_prime + μ)
        X_prime  = R_prime * S + R_F * B - R_D_R * M_prime + Y_prime

        for η_prime_index in 1:nη
            index_no_move = lin[1, IFC_prime_index, η_prime_index, Own_prime_index, M_prime_index]
            index_move    = lin[2, IFC_prime_index, η_prime_index, Own_prime_index, M_prime_index]

            v_no_move = interp_functions[index_no_move]( X_prime)
            v_move    = interp_functions[index_move]( X_prime)

            val += β * T_η[η_index, η_prime_index] * T_ι[1, ι_prime_index] *
                   ((1 - π_m) * v_no_move + π_m * v_move)
        end
    end

    return val
end

###############################################################################
# 2. NO-MOVE OPTIMISATION
###############################################################################
function optimize_retiree_no_move(j::Int, Own::Int64, P::Float64, X::Float64,
                                  η_index::Int, α::Float64, M_index::Int64, M_prime_index::Int64,
                                  IFC::Int, FC::Int, κ::Matrix{Any},
                                  interp_functions::Vector{Any}, s::Float64, F::Float64,
                                  para::Model_Parameters)

    @unpack_Model_Parameters para

    Own_prime = Own

    M = M_grid[M_index]
    M_prime = M_grid[M_prime_index]
    D = M_prime - M

    c_max = X - FC*F - δ * P * Own + D

    if c_max < 0 
        return 0.0, pun
    end

    result = optimize(
        c -> -retiree_value(j, Own, P, X, η_index, c, α, Own_prime, M_index,
                            M_prime_index, IFC, FC, κ, interp_functions,
                            no_move_budget_constraint, s, F, para),
        0.0, c_max, Brent(); abs_tol = tol)

        c = Optim.minimizer(result)
        v = -Optim.minimum(result)
    return c, v
end

###############################################################################
# 3. MOVE: INNER OPTIMISATION OVER c FOR A GIVEN OWN_PRIME
###############################################################################
function optimize_retiree_c(j::Int, Own::Int64, P::Float64, X::Float64, η_index::Int,
                            α::Float64, Own_prime::Int64, M_index::Int64, M_prime_index::Int64, IFC::Int, FC::Int,
                            κ::Matrix{Any}, interp_functions::Vector{Any}, s::Float64, F::Float64,
                            para::Model_Parameters)

    @unpack_Model_Parameters para

    M = M_grid[M_index]
    M_prime = M_grid[M_prime_index]
    D = M_prime - M

    c_max = X - FC*F - δ * P * Own + D + (1-λ)* P * Own - P * (1 + λ) * Own_prime 

    if c_max < 0 
        return 0.0
    end

    result = optimize(
        c -> -retiree_value(j, Own, P, X, η_index, c, α, Own_prime, M_index,
                            M_prime_index, IFC, FC, κ, interp_functions,
                            move_budget_constraint, s , F, para),
        0.0, c_max, Brent(); abs_tol = tol)

    return Optim.minimizer(result)
end

###############################################################################
# 4. OBJECTIVE IN OWN_PRIME (NESTED c SEARCH INSIDE)
###############################################################################
function objective_own_prime(Own_prime::Int64, j::Int, Own::Int64, P::Float64, X::Float64,
                            η_index::Int, α::Float64, M_index::Int64, M_prime_index::Int64, IFC::Int, FC::Int,
                            κ::Matrix{Any}, interp_functions::Vector{Any}, s::Float64, F::Float64,
                            para::Model_Parameters)
    @unpack_Model_Parameters para

    M = M_grid[M_index]
    M_prime = M_grid[M_prime_index]
    D = M_prime - M

    c_max = X - FC*F - δ * P * Own + D + (1-λ)* P * Own - (1 + λ) * P * Own_prime

    if c_max < 0.0
         c_star = 0.0
    else 

        c_star = optimize_retiree_c(j, Own, P, X, η_index, α, Own_prime, M_index,
                                M_prime_index, IFC, FC, κ, interp_functions, s, F, para)
    end

    return retiree_value(j, Own, P, X, η_index, c_star, α, Own_prime, M_index,
                         M_prime_index, IFC, FC, κ, interp_functions, 
                         move_budget_constraint, s, F, para)
end

###############################################################################
# 5. MOVE OPTIMISER (OUTER SEARCH OVER OWN′)
###############################################################################
function optimize_retiree_move(j::Int, Own::Int64, Own_prime::Int64, P::Float64, X::Float64,
                               η_index::Int, α::Float64, M_index::Int64, M_prime_index::Int64,
                               IFC::Int, FC::Int, κ::Matrix{Any},
                               interp_functions::Vector{Any}, s::Float64, F::Float64,
                               para::Model_Parameters)
    @unpack_Model_Parameters para

    v = objective_own_prime(Own_prime, j, Own, P, X, η_index, α, M_index, M_prime_index, IFC, FC,
                                 κ, interp_functions, s, F, para)

  

    c_opt  = optimize_retiree_c(j, Own, P, X, η_index, α, Own_prime, M_index, M_prime_index,
                                     IFC, FC, κ, interp_functions, s, F, para)
 
    return c_opt, v
end



