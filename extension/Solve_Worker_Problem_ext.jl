# Solves the decision problem, outputs results back to the sols structure. 
function Solve_Worker_Problem(para::Model_Parameters, sols::Solutions)
    @unpack_Model_Parameters para 
    @unpack val_func, c_pol_func,M_pol_func, FC_pol_func, α_pol_func, Move_pol_func,  Own_pol_func, κ, σ_ω, s, F = sols
    #println("Solving the Worker's Problem")

    # Generate Transitory earnings grid based on σ_w of group
    ω_grid::Vector{Float64} = rouwenhorst(σ_ω,0.0,g)[1] 
    T_ω::Matrix{Float64} = rouwenhorst(σ_ω,0.0,g)[2]
    nω::Int64 = length(ω_grid)

    #println("Begin solving the model backwards")
    for j in TR-1:-1:1  # Backward induction

        #println("Age is ", 25 - 50/T + (50/T)*j)
        
       # Generate interpolation functions for cash-on hand given each possible combination of the other states tomorrow 
        interp_functions = Vector{Any}(undef, 2 * 2 * nη * 2 * nM ) 

        for Inv_Move_index in 1:2
            for IFC_index in 1:2
                for η_index in 1:nη
                    for Own_index = 1:2
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

                                                # If Inv_Move == 0 then the agent gets to optimize over moving versus not moving. 
                                                if Inv_Move == 0 && Own == Own_prime
                                                    c,val = optimize_worker_no_move(j,ω_grid, T_ω, nω, Own, P, X, η_index, α, M_index, M_prime_index,
                                                                                                 IFC,FC, κ, interp_functions, s, F, para)
                                                    Move = 0
                                                end 

                                                if Inv_Move == 0 && Own != Own_prime
                                                    c,val = optimize_worker_move(j,ω_grid, T_ω, nω, Own, Own_prime, P, X, η_index, α, M_index, M_prime_index,
                                                                                                 IFC,FC, κ, interp_functions, s, F, para)
                                                    Move = 1
                                                end 

                                                # If Inv_Mov == 1 then the agent must optimize conditional on moving only
                                                if Inv_Move == 1
                                                    c,val = optimize_worker_move(j,ω_grid, T_ω, nω, Own, Own_prime, P, X, η_index, α, M_index, M_prime_index,
                                                                                            IFC,FC, κ, interp_functions, s, F, para) 

                                                    Move = 1
                                                end 

                                                # Update value function
                                                if val > candidate_max 
                                                    val_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]     = val

                                                    c_pol_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]   = c 
                                                    M_pol_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]   = M_prime
                                                    FC_pol_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]  = FC
                                                    α_pol_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]   = α
                                                    Move_pol_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j]   = Move
                                                    Own_pol_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j] = Own_prime 

                                                    candidate_max = val 
                                                end 
                                            end 
                                        end 
                                    end
                                end

                                if candidate_max <= pun
                                    val = pun
                                    val_func[ Inv_Move_index, IFC_index, η_index, Own_index, M_index, X_index, j] = val
                                end 
                            end 
                        end 
                    end # IFC Loop
                end  # η loop
            end # Own Loop
        end # M Loop 
    end # X-Loop
end # T loop

######################################
# Optimization Functions 
######################################
###############################################################################
# 1. VALUE FUNCTION
###############################################################################
function worker_value(j::Int, ω_grid::Vector{Float64}, T_ω::Matrix{Float64}, nω::Int64, Own::Int64, P::Float64, X::Float64,
                       η_index::Int, c::Float64, α::Float64, Own_prime::Int64, M_index::Int64, M_prime_index::Int64,
                       IFC::Int, FC::Int,κ::Matrix{Any}, interp_functions::Vector{Any},
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
    if j < TR
        Y_prime_ub = κ[j + 1, 2] *exp(η_grid[3] + ω_grid[g])
        Y_prime_lb = κ[j + 1, 2] *exp(η_grid[1] + ω_grid[1])
    else 
        Y_prime_ub = κ[j + 1, 2] 
        Y_prime_lb = κ[j + 1, 2] 
    end 

    R_prime_max = exp(ι_grid[g] + μ)
    R_prime_min = exp(ι_grid[1]  + μ)

    X_prime_ub  = R_prime_max * S + R_F * B - R_D * M_prime + Y_prime_ub
    X_prime_lb  = R_prime_min * S + R_F * B - R_D * M_prime + Y_prime_lb

    if X_prime_lb < X_min || X_prime_ub > X_max || S_and_B < 0
        return  pun
    end

    # continuation value
    for η_prime_index in 1:nη
        index_no_move = lin[1, IFC_prime_index, η_prime_index, Own_prime_index, M_prime_index]
        index_move    = lin[2, IFC_prime_index, η_prime_index, Own_prime_index, M_prime_index]  

        for ω_prime_index in 1:nω
            # labour‐income next period (κ holds exogenous paths)
            if j < TR
                Y_prime = κ[j + 1, 2] * exp(η_grid[η_prime_index] + ω_grid[ω_prime_index])
            else 
                Y_prime = κ[j + 1, 2] 
            end 

            for ι_prime_index in 1:nι
                ι_prime  = ι_grid[ι_prime_index]
                R_prime  = exp(ι_prime + μ)

                X_prime  = R_prime * S + R_F * B - R_D * M_prime + Y_prime

                v_no_move = interp_functions[index_no_move](X_prime)
                v_move    = interp_functions[index_move](X_prime)

                val += β * T_η[η_index, η_prime_index] * T_ι[1, ι_prime_index] * T_ω[1, ω_prime_index] *
                        ((1 - π_m) * v_no_move + π_m * v_move)
            end 
        end
    end

    return val
end

###############################################################################
# 2. NO-MOVE OPTIMISATION
###############################################################################
function optimize_worker_no_move(j::Int, ω_grid::Vector{Float64}, T_ω::Matrix{Float64}, nω::Int64, Own::Int64, P::Float64, X::Float64,
                                  η_index::Int, α::Float64, M_index::Int64, M_prime_index::Int64,
                                  IFC::Int, FC::Int64, κ::Matrix{Any},
                                  interp_functions::Vector{Any}, s::Float64, F::Float64,
                                  para::Model_Parameters)

    @unpack_Model_Parameters para

    Own_prime = Own

    M = M_grid[M_index]
    M_prime = M_grid[M_prime_index]
    D = M_prime - M

    c_max = X - FC*F - δ * P * Own + D

    if c_max < 0 
        return  0.0, pun
    end

    result = optimize(
        c -> -worker_value(j,ω_grid, T_ω, nω, Own, P, X, η_index, c, α, Own_prime,
                            M_index, M_prime_index, IFC, FC, κ, interp_functions,
                            no_move_budget_constraint,s, F, para),
        0.0, c_max, Brent(); abs_tol = tol)

    c = Optim.minimizer(result)
    v = -Optim.minimum(result)

    return c, v
end

###############################################################################
# 3. MOVE: INNER OPTIMISATION OVER c FOR A GIVEN OWN′
###############################################################################
function optimize_worker_c(j::Int, ω_grid::Vector{Float64}, T_ω::Matrix{Float64}, nω::Int64, Own::Int64, P::Float64, X::Float64, η_index::Int,
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
        c -> -worker_value(j,ω_grid, T_ω, nω, Own, P, X, η_index, c, α, Own_prime,
                            M_index, M_prime_index, IFC, FC, κ, interp_functions,
                            move_budget_constraint,s, F, para),
        0.0, c_max, Brent(); abs_tol = tol)

    return Optim.minimizer(result)
end

###############################################################################
# 4. OBJECTIVE IN OWN′  (NESTED c SEARCH INSIDE)
###############################################################################
function objective_worker_own_prime(Own_prime::Int64, j::Int, ω_grid::Vector{Float64}, T_ω::Matrix{Float64}, nω::Int64, Own::Int64, P::Float64, X::Float64,
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

        c_star = optimize_worker_c(j,ω_grid, T_ω, nω, Own, P, X, η_index, α, Own_prime,
                                M_index, M_prime_index, IFC, FC, κ, interp_functions, s, F, para)
    end

    return worker_value(j, ω_grid, T_ω, nω, Own, P, X, η_index, c_star, α, Own_prime,M_index,M_prime_index, 
    IFC, FC, κ, interp_functions, move_budget_constraint, s, F, para)
end

###############################################################################
# 5. MOVE OPTIMISER (OUTER SEARCH OVER OWN′)
###############################################################################
function optimize_worker_move(j::Int, ω_grid::Vector{Float64}, T_ω::Matrix{Float64}, nω::Int64, Own::Int64, Own_prime::Int64, P::Float64, X::Float64,
                               η_index::Int, α::Float64, M_index::Int64, M_prime_index::Int64,
                               IFC::Int, FC::Int, κ::Matrix{Any},
                               interp_functions::Vector{Any}, s::Float64, F::Float64,
                               para::Model_Parameters)

    @unpack_Model_Parameters para

    v = objective_worker_own_prime(Own_prime, j, ω_grid, T_ω, nω, Own, P, X, η_index, α, M_index, M_prime_index, IFC, FC,
                                 κ, interp_functions, s, F, para)

   

    c_opt = optimize_worker_c(j, ω_grid, T_ω, nω, Own, P, X, η_index, α, Own_prime,
                                     M_index, M_prime_index, IFC, FC, κ, interp_functions, s, F, para)


    return c_opt, v
    
end