#  Matches three moments: the expectation - autocorrelation and variance 
#  you want to match and three parameters - N, σ and ρ. 

# Inputs[variance of process you want to discretize (not the variance of the shock), autocorrelation of that process, N = number of gridpoints; Weighting Matrix]
# Outputs[gridpoint vector, transition probability matrix]
function rouwenhorst(σ::Float64, ρ::Float64, N::Int64)


    # --- Step 3: Estimate optimal (p, q, ψ) using the method of moments
    function GMM(σ::Float64, ρ::Float64, N::Int64)
        p_hat = zeros(3)
        p_hat[1] = (1 + ρ)/2 
        p_hat[2] = (1 + ρ)/2 
        p_hat[3] = sqrt(σ * (N-1))
        return p_hat 
    end

    # --- Step 4: Construct transition matrix
    function compute_transition_matrix(p_hat::Vector{Float64}, N::Int64)
        p, q = p_hat[1], p_hat[2]
        T = [p 1 - p; 1 - q q]

        for i in 2:N-1
            zero_vec = zeros(i, 1)
            T = p   * [T zero_vec; zero_vec' 0] +
                (1 - p) * [zero_vec T; 0 zero_vec'] +
                (1 - q) * [zero_vec' 0; T zero_vec] +
                q   * [0 zero_vec'; zero_vec T]

            for n = 2:size(T, 2) - 1
                T[n, :] *= 0.5
            end
        end
        return T
    end

    # --- Step 5: Generate evenly spaced grid
    function generate_grid(p_hat::Vector{Float64}, N::Int64)
        ψ = p_hat[3]
        return collect(range(start = -ψ, stop = ψ, length = N))
    end

    # === Call steps ===
    p_hat = GMM(σ, ρ, N)
    P = compute_transition_matrix(p_hat, N)
    grid = generate_grid(p_hat, N)

    return grid, P
end
