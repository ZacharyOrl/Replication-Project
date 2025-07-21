@with_kw struct Model_Parameters
   # Number of gridpoints for random variables 
    g = 3

    # Parameters - Converting all annual variables to their five-year equivalents
    # Variance Parameters 

    # Variance of aggregate component of earnings, 
    ann_var::Float64 = 0.019^2

    # Variance of house prices - scaled to five years, using that the persistence comes from the aggregate state.  
    σ_p::Float64 =       0.062^2   
    σ_ι::Float64 =     5  * 0.1674^2  # Variance of stock market innovation

    # Correlations between processes 
    κ_ω::Float64 = 0.00             # Correlation between house prices and transitory component
    ρ_ϵ_ι::Float64 = 0.0            # Correlation between aggregae component of income and the stock market
    φ::Float64 = 0.748^5            # Persistence in the aggregate component - obtained by monte carlo simulation. 
    σ_η::Float64 =  ann_var 
    κ_η::Float64 = sqrt(σ_η/σ_p)    # Regression coefficient of cyclical fluctuations in house prices on aggregate component (correlation is 1)

    # One time stock market entry cost 
    F::Float64 = 5000.0 

    # Weight of the bequest period 
    weight::Float64 = 1.0 # Default is 1.0 

    # Returns / interest rates 
    R_D::Float64 =  compound(1 + 0.04, 5)                  # Mortgage interest rate 
    R_F::Float64 =  compound(1 + 0.02, 5)                  # Risk-free rate
    R_S::Float64 =  compound(1 + 0.1, 5)                   # Expected return on stocks 
    μ::Float64 = log(R_S) - σ_ι/2                          # Expected five-year log-return on stocks

    # Housing parameters
    d::Float64 = 0.15                               # Down-Payment proportion 
    π_m::Float64 = 1 - (1 - 0.032)^5                # Moving shock probability 
    δ::Float64 = 1 - 0.99^5                         # Housing Depreciation
    λ::Float64 = 0.08                               # House-sale cost 
    b::Float64 = 5 * 0.01                           # Real log house price growth over 5 years  - matching the way it is presented in the paper 

    # Utility function parameters
    θ::Float64 = 0.1              # Utility from housing services relative to consumption
    γ::Float64 = 5.0              # Risk-aversion parameter 
    β::Float64 = compound(0.96,5) # Discount Rate 

    # Lifecycle Parameters 
    T::Int64 = 10 # Each T represents five years of life from 25 to 75
    TR::Int64 = 9 # The final two time periods represent retirement 

    # Grids & Transition Matrices
    # Aggregate Earnings
    η_grid::Vector{Float64} =  tauchen_hussey(φ,sqrt(σ_η*(1 - φ^2)),0.0,3)[1] 
    T_η::Matrix{Float64} = tauchen_hussey(φ,sqrt(σ_η*(1 - φ^2)),0.0,3)[2]
    nη::Int64 = length(η_grid)

    # Compute the stationary distribution of aggregate earnings
    # This will be the distribution over the initial aggregate state 
    π_η::Vector{Float64} = stationary_distribution(η_grid, T_η)

    # Stock market grids
    ι_grid::Vector{Float64} = tauchen_hussey(0.0,sqrt(σ_ι),0.0,g)[1] 
    T_ι::Matrix{Float64} = tauchen_hussey(0.0,sqrt(σ_ι),0.0,g)[2] 
    nι::Int64 = g

    # Housing grids
    p_grid::Vector{Float64} =  (1 / κ_η) .* η_grid
    P_bar::Float64 = 1 # Initial Price 
    np::Int64 = nη

    # Punishment value 
    pun::Float64 = -10^5 # The value agents face if they default. 

    # State / Choice Grids    

    # Agents start life with no housing and are forced to purchase a home in the first period. 
    H_min::Float64 = 20000.0
    H_max::Float64 = 160000.0
    nH::Int64 = 9

    H_grid::Vector{Float64} = vcat(0.0, collect(range(H_min, length = nH - 1, stop = H_max)))

    # The minimum level of cash on hand one can sustain. 
    X_min::Float64 = -200000.0
    X_max::Float64 =  3000000.0
    nX::Int64 = 125
    X_grid::Vector{Float64} = collect(range(X_min, length = nX, stop = X_max))

    α_min::Float64 = 0.0
    α_max::Float64 = 1.0
    nα::Int64 = 6
    α_grid::Vector{Float64} = collect(range(α_min, length = nα, stop = α_max))

    # The investor's loan-to-home value ratio. 
    LTV_min::Float64 = 0.0
    LTV_max::Float64 = (1-d)
    nLTV::Int64 = 5
    LTV_grid::Vector{Float64} = collect(range(LTV_min, length = nLTV, stop = LTV_max))

    # The investor's consumption share of their total resources
    C_share_min::Float64 = 0.000001
    C_share_max::Float64 = 0.999999
    nC::Int64 = 40
    ν               = 1.0      # Adjust this curvature parameter: ν > 1 → more points near minc
    ξ_grid          = range(C_share_min, C_share_max, length = nC)
    ξ_shaped_grid   = [ξ^ν for ξ in ξ_grid]

    Inv_Move_grid::Vector{Int64} = [0, 1]

    FC_grid::Vector{Int64}       = [0, 1]

    Move_grid::Vector{Int64}     = [0, 1]

    IFC_grid::Vector{Int64}      = [0, 1]

    # A holder for the indices of each random state  
    lin::LinearIndices{4,Tuple{Base.OneTo{Int64},Base.OneTo{Int64},Base.OneTo{Int64},Base.OneTo{Int64}}} = LinearIndices((2, 2, nη, nH))
    
    # Weighting grid for simulations - taken from Cocco(2005) 
                            #    nhs  hs    clg
    wts::Matrix{Float64} = [
                                0.01  0.06  0.02;   # 25–29
                                0.02  0.08  0.04;   # 30–34
                                0.01  0.10  0.06;   # 35–39
                                0.0175  0.075  0.0675;   # 40–44 - Values slightly adjusted to account for Cocco rounding
                                0.01  0.05  0.03;   # 45–49
                                0.0175  0.035  0.0175;   # 50–54 - Values slightly adjusted so the row sum matches Cocco
                                0.02  0.04  0.02;   # 55–59
                                0.02  0.03  0.02;   # 60–64
                                0.02  0.04  0.01;   # 65–69
                                0.02  0.03  0.01    # 70–74
                                ]                      
end

#initialize value function and policy functions
mutable struct Solutions

    # 6 states, it turns out that the retired's value function still depends on η even after retirement,as it pins down housing.
    val_func::Array{Float64,6} 
    c_pol_func::Array{Float64,6}
    H_pol_func::Array{Float64,6}
    LTV_pol_func::Array{Float64,6}
    FC_pol_func::Array{Float64,6}
    α_pol_func::Array{Float64,6}
    Move_pol_func::Array{Float64,6}
    S_and_B_pol_func::Array{Float64,6}

    σ_ω::Float64
    κ::Matrix{Any}

end

# Initialize Simulations Structure
struct Sim_Results
    bonds               ::Vector{Float64}
    stocks              ::Vector{Float64}
    stock_share         ::Vector{Float64}
    stock_market_entry  ::Vector{Int}
    IFC_paid            ::Vector{Int}

    housing             ::Vector{Float64}
    moved               ::Vector{Float64}
    Inv_Move_shock      ::Vector{Float64}
    cash_on_hand        ::Vector{Float64}
    expected_earnings   ::Vector{Float64}
    

    debt                ::Vector{Float64}
    LTV                 ::Vector{Float64}
    consumption         ::Vector{Float64}
    wealth              ::Vector{Float64}
    bequest             ::Vector{Float64}

    income              ::Vector{Float64}
    persistent          ::Vector{Float64}
    transitory          ::Vector{Float64}
    stock_market_shock  ::Vector{Float64}

    age                 ::Vector{Float64}
    education           ::Vector{Float64}
end

function build_solutions(para, σ_ω::Float64, κ::Matrix{Any}) 

    # Value function has an extra level for age due to bequest.
    # domain of val/pol funcs is [Inv_Move,IFC_Paid,aggregate_state,Housing State, Cash State, Age] 
    val_func    = zeros(Float64,2,2,para.nη, para.nH, para.nX, para.T + 1 ) 
    c_pol_func  = zeros(Float64,2,2,para.nη, para.nH, para.nX, para.T ) 
    H_pol_func  = zeros(Float64,2,2,para.nη, para.nH, para.nX, para.T ) 
    LTV_pol_func  = zeros(Float64,2,2,para.nη, para.nH, para.nX, para.T ) 
    FC_pol_func = zeros(Float64,2,2,para.nη, para.nH, para.nX, para.T ) 
    α_pol_func  = zeros(Float64,2,2,para.nη, para.nH, para.nX, para.T ) 
    Move_pol_func  = zeros(Float64,2,2,para.nη, para.nH, para.nX, para.T ) 
    S_and_B_pol_func  = zeros(Float64,2,2,para.nη, para.nH, para.nX, para.T ) 

    sols = Solutions(val_func, c_pol_func, H_pol_func, LTV_pol_func, FC_pol_func, α_pol_func, Move_pol_func, S_and_B_pol_func, σ_ω , κ)

    return sols
end 

function Initialize_Model(σ_ω::Float64, κ::Matrix{Any}) 

    para = Model_Parameters()
    sols = build_solutions(para, σ_ω, κ)

    return para, sols 
end