function tauchen_hussey_iid(N, σ, μ)
    """
    N: Number of discrete states
    σ: Standard deviation of the normal shocks
    μ: Mean of the normal shocks
    Returns: (nodes, weights) for the discretized IID shocks
    """
    nodes, weights = gausshermite(N)         # Gauss-Hermite nodes and weights
    nodes = nodes .* sqrt(2) .* σ .+ μ       # Apply scalar operations element-wise using dot syntax
    weights = weights ./ sqrt(π)             # Normalize weights
    return nodes, weights
end


 #= ################################################################################################## 
    Tauchen-Hussey for Persistent 
=# ##################################################################################################

"""
    tauchen_hussey(ρ, σ, μ, N)

Discretise an AR(1) using the Tauchen–Hussey method.

# Arguments
- `ρ::Float64`   : AR(1) persistence (|ρ| < 1)
- `σ::Float64`   : innovation std. dev. (σ_ε)
- `μ::Float64`   : unconditional mean
- `N::Int`       : number of grid points

# Returns
- `z::Vector{Float64}` : length-`N` state grid (sorted low → high)
- `Π::Matrix{Float64}` : N×N Markov transition matrix
"""
function tauchen_hussey(ρ, σ, μ, N)
    # 1.  Gauss–Hermite nodes (ξ) and weights (ω) for N-point quadrature on ℝ
    ξ, ω = gausshermite(N)
    ω    = ω ./ sqrt(π)                     # normalise weights to integrate N(0,1)
    ξ    = ξ .* sqrt(2.0)                  # convert from Hermite to N(0,1) std dev

    # 2.  Match first two moments of stationary distribution
    σ_x  = σ / sqrt(1 - ρ^2)               # unconditional std dev of x_t
    z    = μ .+ σ_x .* ξ                   # grid = μ + σ_x * ξ  (sorted already)

    # 3.  Compute transition matrix Π_(i,j) = Pr[x_{t+1}=z_j | x_t=z_i]
    Π = zeros(N, N)
    for i in 1:N
        # conditional mean of ε given current node x_i under GH weighting
        #   m_i = E[ε | ε ≈ ξ?]  -->  Tauchen–Hussey adjust with conditional expectation
        #   BUT for AR(1) with Gaussian shocks, conditional ε mean is 0.
        # conditional standard deviation of x_{t+1}
        cond_mean  = μ + ρ * (z[i] - μ)
        cond_stdev = σ
        for j in 1:N
            # probability that x_{t+1} lands in neighbourhood of z_j
            Π[i, j] = ω[j] *
                      pdf(Normal(cond_mean, cond_stdev), z[j]) /
                      pdf(Normal(μ, σ_x), z[j])
        end
        Π[i, :] ./= sum(Π[i, :])           # normalise row to sum to one
    end
    return z, Π
end