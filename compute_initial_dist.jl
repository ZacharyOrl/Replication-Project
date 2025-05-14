function compute_initial_dist(μ_0::Float64, σ_0::Float64, grid::Vector{Float64})
    n = length(grid)
    dist = zeros(n)

    # Precompute normal
    N = Normal(μ_0, σ_0)

    # Compute midpoints between grid points
    midpoints = [(grid[i] + grid[i+1])/2 for i in 1:n-1]

    # First point: mass from -∞ to first midpoint
    dist[1] = cdf(N, midpoints[1])

    # Interior points
    for i in 2:n-1
        dist[i] = cdf(N, midpoints[i]) - cdf(N, midpoints[i-1])
    end

    # Last point: mass from last midpoint to +∞
    dist[n] = 1 - cdf(N, midpoints[n-1])

    return dist
end