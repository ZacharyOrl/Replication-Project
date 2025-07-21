# Solves the decision problem, outputs results back to the sols structure. 
function Solve_Worker_Problem(para::Model_Parameters, sols::Solutions)
    @unpack_Model_Parameters para 
    @unpack val_func, c_pol_func, H_pol_func, LTV_pol_func, FC_pol_func, α_pol_func, Move_pol_func, S_and_B_pol_func, κ, σ_ω = sols
    println("Solving the Worker's Problem")

    # Generate Transitory earnings grid based on σ_w of group
    ω_grid::Vector{Float64} = tauchen_hussey(0.0,sqrt(σ_ω),0.0,g)[1] 
    T_ω::Matrix{Float64} = tauchen_hussey(0.0,sqrt(σ_ω),0.0,g)[2]
    nω::Int64 = length(ω_grid)

    println("Begin solving the model backwards")
    for j in TR-1:-1:1  # Backward induction

        println("Age is ", 25 - 50/T + (50/T)*j)
        
       # Generate interpolation functions for cash-on hand given each possible combination of the other states tomorrow 
        interp_functions = Vector{Any}(undef, 2 * 2 * nη * nH) 

        for H_index = 1:nH
            for Inv_Move_index = 1:2
                for IFC_index = 1:2
                    for η_index = 1:nη
                        # Compute linear index 
                        index = lin[Inv_Move_index, IFC_index, η_index, H_index]
                        # Access val_func with dimensions [Inv_Move, IFC, η, H, X, j]

                        interp = interpolate( val_func[Inv_Move_index, IFC_index, η_index, H_index, :, j+1], BSpline(Linear()),OnGrid())
                        interp_functions[index]        = extrapolate(interp, Interpolations.Flat())

                    end 
                end
            end
        end 

        # Loop over cash on hand states
        Threads.@threads for X_index in 1:nX
            X = X_grid[X_index]

            # Loop over housing states
            for H_index in 1:nH
                H = H_grid[H_index]

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

                            # Loop over LTV choices
                            for LTV_index = 1:nLTV 
                                LTV = LTV_grid[LTV_index]

                                # Loop over stock share choices
                                for α_index = 1:nα
                                    α = α_grid[α_index]

                                    # Loop over stock market entry choices 
                                    for FC_index = 1:2 
                                        FC = FC_grid[FC_index]

                                        # Loop over housing choices 
                                        for H_prime_index in 2:nH # start from index 2 as index 1 is the 0 housing state. 
                                            H_prime = H_grid[H_prime_index]

                                             # Record whether the agent is moving
                                             # and find maximum consumption 
                                            if H_prime != H || Inv_Move == 1
                                                Move = 1
                                                c_max = X - FC*F - δ * P * H + LTV * P * H_prime + (1-λ)* P * H - P * H_prime
                                            else 
                                                Move = 0
                                                c_max = X - FC*F - δ * P * H + LTV * P * H_prime 
                                            end  
                                            
                                            if c_max < 0 
                                                continue 
                                            end 

                                            # Loop over consumption share choices
                                            for c_share_index in 1:nC 
                                                c_share = ξ_shaped_grid[c_share_index]

                                                c = c_share * c_max 

                                                # Impose conditions 
                                                # If the agent has already entered the stock market, they won't do it again. 
                                                if IFC == 1 && FC == 1
                                                    continue 
                                                end 

                                                # Not possible to have positive stock market share if not in the stock market. 
                                                if IFC == 0 && FC == 0 && α > 0.0 
                                                    continue 
                                                end 

                                                # Debt and bills are perfect substitutes. 
                                                # If in the market and if LTV is positive, α must be 1.0 
                                                if α < 1.0 && LTV > 0.0 && (FC == 1 || IFC == 1)
                                                    continue 
                                                end 

                                                # Skip the initial housing state unless in the first period.                                               # Skip the first housing state unless in the first period 
                                                if j != 1 && H == H_grid[1]
                                                    val = pun
                                                    continue 
                                                end 

                                                
                                                # If in the first period, skip every other housing state
                                                if j == 1 && H != H_grid[1]
                                                    val = pun
                                                    continue 
                                                end 

                                                # If Move == 0 then compute the value subject to the no move budget 
                                                if Move == 0
                                                    val = worker_value(j, ω_grid, T_ω, nω, H, P, X, η_index, c, 
                                                                        α, H_prime_index, LTV, IFC, FC, κ, 
                                                                        interp_functions, no_move_budget_constraint, para)
                                                end 

                                                # If Move == 1, then compute the value subject to the move budget. 
                                                if Move == 1
                                                    val = worker_value(j, ω_grid, T_ω, nω, H, P, X, η_index, c, 
                                                                        α, H_prime_index, LTV, IFC, FC, κ, 
                                                                        interp_functions, move_budget_constraint, para)
                                                end 
     

                                                # Update value function
                                                if val > candidate_max 
                                                    val_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]     = val
                                                    #println("Value is: ", val)

                                                    c_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = c 
                                                    H_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = H_prime
                                                    LTV_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = LTV
                                                    FC_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]  = FC
                                                    α_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = α
                                                    Move_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = Move

                                                    if Move == 1
                                                        S_and_B_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j] = move_budget_constraint(X, H, P, c, H_prime, LTV, FC, para)
                                                    end 

                                                    if Move == 0
                                                        S_and_B_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j] = no_move_budget_constraint(X, H, P, c, H_prime, LTV, FC, para)
                                                    end 

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
                end # IFC Loop
            end  # η loop
        end # H Loop
    end # X-Loop
end # T loop

######################################
# Optimization Functions 
######################################
###############################################################################
# 1. VALUE FUNCTION
###############################################################################
function worker_value(j::Int, ω_grid::Vector{Float64}, T_ω::Matrix{Float64}, nω::Int64, H::Float64, P::Float64, X::Float64,
                       η_index::Int, c::Float64, α::Float64, H_prime_index::Int64,
                       LTV::Float64, IFC::Int, FC::Int,
                       κ::Matrix{Any}, interp_functions::Vector{Any},
                       constraint::Function, para::Model_Parameters)

    @unpack_Model_Parameters para 
    H_prime = H_grid[H_prime_index]
    S_and_B = constraint(X, H, P, c, H_prime, LTV, FC, para)

    IFC_prime_index = max(IFC, FC) + 1          # 1 = no stock entry, 2 = entered
    S  = α       * S_and_B
    B  = (1 - α) * S_and_B

    D  = P * H_prime * LTV

    val = flow_utility_func(c, H_prime, para)

    # labour‐income next period (κ holds exogenous paths)

    # Compute highest/lowest continuation COH
    if j < TR - 1
        Y_prime_lb = κ[j + 1, 2] * exp(η_grid[1] + ω_grid[1])
    else 
        Y_prime_lb = κ[j + 1, 2] 
    end 

    R_prime_min = exp(ι_grid[1]  + μ)

    X_prime_lb  = R_prime_min * S + R_F * B - R_D * D + Y_prime_lb

    # Impose punishment continuation value for choices which 
    # imply outside-grid COH states. 
    if X_prime_lb < X_min
        return  pun
    end

    # continuation value
    for η_prime_index in 1:nη
        index_no_move = lin[1, IFC_prime_index, η_prime_index, H_prime_index]
        index_move    = lin[2, IFC_prime_index, η_prime_index, H_prime_index]  

        for ω_prime_index in 1:nω
            # labour‐income next period (κ holds exogenous paths)
            if j < TR - 1
                Y_prime = κ[j + 1, 2] * exp(η_grid[η_prime_index] + ω_grid[ω_prime_index])
            else 
                Y_prime = κ[j + 1, 2] 
            end 

            for ι_prime_index in 1:nι
                ι_prime  = ι_grid[ι_prime_index]
                R_prime  = exp(ι_prime + μ)

                X_prime  = R_prime * S + R_F * B - R_D * D + Y_prime

                v_no_move = interp_functions[index_no_move](map_X(X_prime,para))
                v_move    = interp_functions[index_move](map_X(X_prime,para))

                val += β * T_η[η_index, η_prime_index] * T_ι[1, ι_prime_index] * T_ω[1, ω_prime_index] *
                        ((1 - π_m) * v_no_move + π_m * v_move)
            end 
        end
    end

    return val
end

