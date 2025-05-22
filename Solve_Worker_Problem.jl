# Solves the decision problem, outputs results back to the sols structure. 
function Solve_Worker_Problem(para::Model_Parameters, sols::Solutions)
    @unpack_Model_Parameters para 
    @unpack val_func, c_pol_func, H_pol_func, D_pol_func, FC_pol_func, α_pol_func, κ, σ_ω = sols
    println("Solving the Worker's Problem")

    # Generate Transitory earnings grid based on σ_w of group
    ω_grid::Vector{Float64} = tauchen_hussey_iid(g,sqrt(σ_ι), 0.0)[1] 
    T_ω::Matrix{Float64} = tauchen_hussey_iid(g,sqrt(σ_ι), 0.0)[2][1:g,1:1]'
    nω::Int64 = length(ω_grid)

    println("Begin solving the model backwards")
    for j in TR-1:-1:1  # Backward induction

        println("Age is ", 25 - 50/T + (50/T)*j)
        
       # Generate interpolation functions for cash-on hand given each possible combination of the other states tomorrow 
       interp_functions = Vector{Any}(undef, 2 * 2 * nη * (nH+1)) 

       for Inv_Move_index in 1:2
           for IFC_index in 1:2
               for η_index in 1:nη
                   for H_index in 1:(nH + 1)
                        # Compute linear index 
                        index = lin[Inv_Move_index, IFC_index, η_index, H_index]
                        # Access val_func with dimensions [Inv_Move, IFC, η, H, X, j]
                        interp_functions[index] = linear_interp(val_func[Inv_Move_index, IFC_index, η_index, H_index, :, j+1], X_grid)
                   end
               end
           end
       end

        # Loop over cash on hand states
        Threads.@threads for X_index in 1:nX
            X = X_grid[X_index]

            # Loop over housing states
            for H_index in 1:(nH + 1)
                H = H_state_grid[H_index]

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

                            # If an agent has already paid the stock market entry fee, they won't pay it again. 
                            if IFC == 1
                                FC_index = 1
                                FC = FC_grid[FC_index]

                                # Loop over Housing choices 
                                for H_prime_index in 1:nH 
                                    H_prime = H_choice_grid[H_prime_index]
                                    
                                    # Loop over Risky-share choices
                                    for α_index in 1:nα 
                                        α = α_grid[α_index]

                                        # Debt and bills are perfect substitutes
                                        if  α < 1.0
                                            D = 0.0 
                                            c = optimize_worker_c(j, H, P, X, η_index, Inv_Move, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, κ, ω_grid,T_ω, nω, para)
                                            val = compute_worker_value(j, H, P,  X, η_index, Inv_Move, c, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, κ, ω_grid,T_ω, nω, para)
                                        else
                                            D, c, val = optimize_worker_d(j, H, P, X, η_index, Inv_Move, α, H_prime, H_prime_index, IFC, FC, interp_functions, κ, ω_grid,T_ω, nω, para)
                                        end 

                                            # Update value function
                                            if val > candidate_max 
                                                val_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]     = val
                                                #println("Value is: ", val)

                                                c_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = c 
                                                H_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = H_prime
                                                D_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = D
                                                FC_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]  = FC
                                                α_pol_func[ Inv_Move_index, IFC_index, η_index, H_index, X_index, j]   = α

                                                candidate_max = val 
                                            end 
                                    end
                                end 
                            # They have not already paid the entry fee (IFC == 0 )
                            else 

                                # Loop over Housing choices 
                                for H_prime_index in 1:nH
                                    H_prime = H_choice_grid[H_prime_index]                                
                                    
                                    # Loop over enter/not enter choices 
                                    for FC_index in 1:2
                                        FC = FC_grid[FC_index]
                                    
                                        # If the person has not paid the entry cost and does not pay the entry cost today 
                                        # they must invest 0 in stocks. 
                                        if FC == 0 && IFC == 0 
                                            α_index = 1 
                                            α = α_grid[α_index]
                                    
                                            D, c, val = optimize_worker_d(j, H, P, X, η_index, Inv_Move, α, H_prime, H_prime_index, IFC, FC, interp_functions, κ, ω_grid,T_ω, nω, para)
                                                                                        
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
                                        # If IFC == 0 & FC == 1
                                        else 
                                        # Loop over Risky-share choices
                                            for α_index in 1:nα 
                                                α = α_grid[α_index]

                                                # Debt and bills are perfect substitutes. 
                                                # If α is less than 1 and FC == 1, then you must hold positive values of both stocks and bonds. 
                                                # Therefore, Debt must be zero, and we can skip the outer optimization. 
                                                if  α < 1.0
                                                    D  = 0.0 
                                                    c  = optimize_worker_c(j, H, P, X, η_index, Inv_Move, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, κ, ω_grid,T_ω, nω, para)
                                                    val = compute_worker_value(j, H, P,  X, η_index, Inv_Move, c, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, κ, ω_grid,T_ω, nω, para)
                                                else
                                                    D, c, val = optimize_worker_d(j, H, P, X, η_index, Inv_Move, α, H_prime, H_prime_index, IFC, FC, interp_functions, κ, ω_grid,T_ω, nω, para)
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
function compute_worker_value(j::Int64, H::Float64, P::Float64, X::Float64, η_index::Int64, Inv_Move::Int64, 
                                c::Float64, α::Float64, H_prime::Float64, H_prime_index::Int64, D::Float64, 
                                IFC::Int64, FC::Int64, interp_functions::Vector{Any}, κ::Matrix{Any}, ω_grid::Vector{Float64},
                                T_ω::Matrix{Float64}, nω::Int64, para::Model_Parameters )

    @unpack_Model_Parameters para
    
    S_and_B = budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)

    # Stock market entry state tomorrow is the maximum of whether you entered before or entered today (+1 to find the index)
    IFC_prime_index = max(IFC,FC) + 1

    # Compute Stock and Bond Positions 
    S = α * S_and_B 
    B = (1-α) * S_and_B 

    # Utility comes from the choice of consumption and housing today
    val = flow_utility_func(c, H_prime, para)

    # Find the continuation value 

    # Compute the upper bound of cash on hand tomorrow 
    R_prime_ub = exp(ι_grid[g] + μ)

    # If not about to retire, future labor income is risky
    if j < TR - 1
        Y_Prime_ub = κ[j+1, 2] * exp(η_grid[3] + ω_grid[g])
        Y_Prime_lb = κ[j+1, 2] * exp(η_grid[1] + ω_grid[1])       
    # If about to retire, future labor income is no longer risky. 
    else 
        Y_Prime_ub = κ[j+1, 2]
        Y_Prime_lb = κ[j+1, 2]
    end 

    X_prime_ub = R_prime_ub * S + R_F * B - R_D * D + Y_Prime_ub

    # Compute the lower bound of cash on hand tomorrow 
    R_prime_lb = exp(ι_grid[1] + μ)
    X_prime_lb = R_prime_lb * S + R_F * B - R_D * D + Y_Prime_lb

    # Impose that the agent must invest such that they never leave the grid. 
    if X_prime_lb < X_min || X_prime_ub > X_max || S_and_B < 0  
        val += pun 
    else 

        # Loop over random variables 
        for η_prime_index in 1:nη
            η_prime = η_grid[η_prime_index]

            index_no_move = lin[1, IFC_prime_index, η_prime_index, H_prime_index + 1]                                                  
            index_move = lin[2, IFC_prime_index, η_prime_index, H_prime_index + 1]   

            for ω_prime_index in 1:nω
                ω_prime = ω_grid[ω_prime_index]

                # If not about to retire, future labor income is risky
                if j < TR - 1
                    Y_Prime = κ[j+1, 2] * exp(η_prime + ω_prime)
                
                # If about to retire, future labor income is no longer risky. 
                else 
                    Y_Prime = κ[j+1, 2]
                end 

                for ι_prime_index in 1:nι
                    ι_prime = ι_grid[ι_prime_index]

                    R_prime = exp(ι_prime + μ)

                    # Compute next period's liquid wealth
                    X_prime = R_prime * S + R_F * B - R_D * D + Y_Prime

                    val += β * ( (1-π_m) * T_η[η_index, η_prime_index]  * T_ι[1, ι_prime_index] * T_ω[1, ω_prime_index] *
                            interp_functions[index_no_move](X_prime) +
                            π_m * T_η[η_index, η_prime_index]   * T_ι[1, ι_prime_index] * T_ω[1, ω_prime_index] *
                            interp_functions[index_move](X_prime)   
                            ) 
                end   
            end 
        end 
    end 
    return val 
end 

# Optimize value function over c given a choice of D
function optimize_worker_c(j::Int64, H::Float64, P::Float64, X::Float64, η_index::Int64, Inv_Move::Int64, α::Float64, H_prime::Float64, 
                            H_prime_index::Int64, D::Float64, IFC::Int64, FC::Int64, interp_functions::Vector{Any}, κ::Matrix{Any},
                            ω_grid::Vector{Float64}, T_ω::Matrix{Float64}, nω::Int64, para::Model_Parameters)
    @unpack_Model_Parameters para
    # Find maximum feasible consumption
    budget(c) =  budget_constraint(X, H, P, Inv_Move, c, H_prime, D, FC, para)

    c_max_case = find_zero(budget, X, Roots.Order1())

    if c_max_case < 0
        return 0.0

    else
        # Optimize using Brent's method
        result = optimize(c -> -compute_worker_value(j, H, P,  X, η_index, Inv_Move, c, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, κ, ω_grid,T_ω, nω, para), 0.0, c_max_case, Brent(); abs_tol = tol)
        
        return Optim.minimizer(result)
    end 
end

# Find the value of the problem given a choice of D taking the optimizing choice of c function as given. 
function objective_worker_D(D::Float64, j::Int64, H::Float64, P::Float64, X::Float64, η_index::Int64, Inv_Move::Int64, α::Float64, 
                            H_prime::Float64,H_prime_index::Int64,IFC::Int64, FC::Int64, interp_functions::Vector{Any}, κ::Matrix{Any},
                            ω_grid::Vector{Float64}, T_ω::Matrix{Float64}, nω::Int64, para::Model_Parameters)
    @unpack_Model_Parameters para

    # Get optimal c for this D
    c = optimize_worker_c(j, H, P, X, η_index, Inv_Move, α, H_prime, H_prime_index, D, IFC, FC, interp_functions, κ, ω_grid,T_ω, nω, para)
    
    # Compute value
    val = compute_worker_value(j, H, P,  X, η_index, Inv_Move, c, α, H_prime, H_prime_index, D,IFC, FC, interp_functions, κ, ω_grid,T_ω, nω, para)
    return val  # Negative for maximization
end

function optimize_worker_d(j::Int64, H::Float64, P::Float64, X::Float64, η_index::Int64, Inv_Move::Int64, α::Float64, 
                           H_prime::Float64, H_prime_index::Int64,IFC::Int64, FC::Int64,interp_functions::Vector{Any}, κ::Matrix{Any},
                           ω_grid::Vector{Float64}, T_ω::Matrix{Float64}, nω::Int64, para::Model_Parameters)

    @unpack_Model_Parameters para
    # Find maximum feasible consumption
    debt_limit(D) =  debt_constraint(D, H_prime, P, para)
    initial_guess = (1-d) * H_prime * P / 2 # Guess half of the maximum value. 
    D_max_case = find_zero(debt_limit, initial_guess, Roots.Order1())

    if D_max_case < 0
        return (D_opt = 0.0,c_opt = 0.0, val_opt = pun)
    else
    
    # Optimize using Brent's method
    result = optimize(D -> -objective_worker_D(D, j, H, P,  X, η_index, Inv_Move, α, H_prime, H_prime_index, IFC, FC, interp_functions, κ, ω_grid,T_ω, nω, para), 0.0, D_max_case, Brent(); abs_tol = tol)
    D_opt = Optim.minimizer(result)

    c_opt = optimize_worker_c(j, H, P, X, η_index, Inv_Move,α, H_prime, H_prime_index, D_opt, IFC, FC, interp_functions, κ, ω_grid,T_ω, nω, para)
    val_opt = -Optim.minimum(result)
        return (D_opt, c_opt, val_opt)
    end 
end