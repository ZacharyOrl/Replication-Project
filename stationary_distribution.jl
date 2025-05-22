# Computes the stationary distribution of a Markov Chain
# Requires StatsBase
function stationary_distribution(η_grid::Vector{Float64},T_η::Matrix{Float64}; S = 10000, burn_in = 2000)
    nη = length(η_grid)

    # Find the stationary distribution of aggregate income process η
    perm_dists = [Categorical(T_η[i, :]) for i in 1:nη]

    # Need to choose some initial state to start the markov-chain 
    initial_dist = perm_dists[1]

    persistent = zeros(S)
    index_persistent = rand(initial_dist)

    # Persistent and Transitory components 
    persistent[1] = η_grid[index_persistent]

    for s = 2:S

        index_persistent = rand(perm_dists[index_persistent]) # Draw the new permanent component based upon the old one.         

        # Outputs 
        persistent[s] = η_grid[index_persistent]

    end 

    chain = persistent[(burn_in +1) : S]
    state_counts = countmap(chain)                 # Dict( state => frequency )
    T            = length(chain)
    Stationary_Distribution = [ state_counts[η_grid[s]] / T for s in 1:nη ]
    return Stationary_Distribution
end 
