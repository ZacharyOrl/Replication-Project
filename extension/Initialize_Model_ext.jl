@with_kw struct Model_Parameters
    # Number of gridpoints for random variables 
    g = 3

    # Parameters - Converting all annual variables to their five-year equivalents
    # Variance Parameters 

    # Variance of aggregate component of earnings, 
    # 16.64 is the variance of the rolling sum of an annual AR(1) with autocorrelation of 0.748 and var 0.019^2
    # Persistent income is aggregated using a sum  
    σ_η::Float64 =     0.019^2   # For now, assume the parameters report in Cocco are already five year adjusted. 

    # Variance of house prices - scaled to five years, using that the persistence comes from the aggregate state.  
    σ_p::Float64 =     0.062^2   
    σ_ι::Float64 =     5  * 0.1674^2  # Variance of stock market innovation

    # Correlations between processes 
    κ_ω::Float64 = 0.00             # Correlation between house prices and transitory component
    κ_η::Float64 = sqrt(σ_η/σ_p)    # Regression coefficient of cyclical fluctuations in house prices on aggregate component (correlation is 1)
    ρ_ϵ_ι::Float64 = 0.0            # Correlation between aggregae component of income and the stock market
    φ::Float64 = 0.748^5            # Persistence in the aggregate component - obtained by monte carlo simulation. 

    # Returns / interest rates 
    R_D::Float64 =  compound(1 + 0.04, 5) - 1              # Mortgage interest rate - as mortgage is a stock this needs to be the NET interest rate.
    R_D_R::Float64 =  compound(1 + 0.05, 5) - 1            # Retirees pay a slightly higher rate to encourage downsizing.  
    R_F::Float64 =  compound(1 + 0.02, 5)                  # Risk-free rate
    R_S::Float64 =  compound(1 + 0.1, 5)                   # Expected return on stocks 
    μ::Float64 = log(R_S) - σ_ι/2                          # Expected five-year log-return on stocks

    # Housing parameters
    d::Float64 = 0.2                               # Down-Payment proportion 
    π_m::Float64 = 1 - (1 - 0.05)^5                # Moving shock probability 
    δ::Float64 = 1 - 0.99^5                         # Housing Depreciation
    λ::Float64 = 0.05                               # House-trade cost
    b::Float64 = 5 * 0.01                           # Real log house price growth over 5 years  - matching the way it is presented in the paper 

    # Rental Parameters 
    r_p::Float64 = 0.216 # The discounted present value of five years of rent payments. 

    # Utility function parameters
    θ::Float64 = 0.2              # Utility from housing services relative to consumption
    γ::Float64 = 5.0              # Risk-aversion parameter 
    β::Float64 = compound(0.96,5) # Discount Rate 

    # Lifecycle Parameters 
    T::Int64 = 10 # Each T represents five years of life from 25 to 75
    TR::Int64 = 9 # The final two time periods represent retirement 

    # Grids & Transition Matrices
    # aggregae Earnings
    η_grid::Vector{Float64} =  rouwenhorst(σ_η,φ,3)[1] 
    T_η::Matrix{Float64} = rouwenhorst(σ_η,φ,3)[2]
    nη::Int64 = length(η_grid)

    # Compute the stationary distribution of aggregate earnings
    # This will be the distribution over the initial aggregate state 
    π_η::Vector{Float64} = stationary_distribution(η_grid, T_η)

    # Stock market grids
    ι_grid::Vector{Float64} = rouwenhorst(σ_ι,0.0,g)[1] 
    T_ι::Matrix{Float64} = rouwenhorst(σ_ι,0.0,g)[2] 
    nι::Int64 = length(ι_grid)

    # Housing grids
    p_grid::Vector{Float64} = η_grid ./ κ_η
    P_bar::Float64 = 80000 # Median price of a home in 1992
    P_max::Float64 = exp((T)*b + p_grid[3]) * P_bar # Maximum home price investor can borrow against (can't borrow in T+1)
    np::Int64 = nη

    # Punishment value 
    pun::Float64 = -10^5 # The value agents face if they default. 

    # State / Choice Grids 
    X_min::Float64 = 0.0
    X_max::Float64 =  2000000.0
    nX::Int64 = 100
    X_grid::Vector{Float64} = collect(range(X_min, length = nX, stop = X_max))

    α_min::Float64 = 0.0
    α_max::Float64 = 1.0
    nα::Int64 = 10
    α_grid::Vector{Float64} = collect(range(α_min, length = nα, stop = α_max))

    # The investor's mortgage debt:
    M_min::Float64 = 0.0
    M_max::Float64 = (1 - d) * P_max
    nM::Int64 = 8
    M_grid::Vector{Float64} = collect(range(M_min, length = nM, stop = M_max))

    Inv_Move_grid::Vector{Int64} = [0, 1]

    FC_grid::Vector{Int64}       = [0, 1]

    Move_grid::Vector{Int64}     = [0, 1]

    IFC_grid::Vector{Int64}      = [0, 1]

    Own_grid::Vector{Int64}      = [0, 1]

    # A holder for the indices of each random state  
    lin::LinearIndices{5,Tuple{Base.OneTo{Int64},Base.OneTo{Int64},Base.OneTo{Int64},Base.OneTo{Int64},Base.OneTo{Int64}}} = LinearIndices((2, 2, nη, 2, nM))
    
    tol::Float64 = 2000.0          # stop optimizing once the candidate bracket is ≤ $1000 wide                   
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

    # 7 states, it turns out that the retired's value function still depends on η even after retirement,as it pins down housing.
    val_func::Array{Float64,7} 
    c_pol_func::Array{Float64,7}
    M_pol_func::Array{Float64,7}
    FC_pol_func::Array{Float64,7}
    α_pol_func::Array{Float64,7}
    Move_pol_func::Array{Float64,7}
    Own_pol_func::Array{Float64,7}

    σ_ω::Float64
    κ::Matrix{Any}

    F::Float64
    s::Float64 

end

# Initialize Simulations Structure
struct Sim_Results
    bonds               ::Vector{Float64}
    stocks              ::Vector{Float64}
    stock_share         ::Vector{Float64}
    stock_market_entry  ::Vector{Int}
    IFC_paid            ::Vector{Int}

    own                 ::Vector{Int64}
    moved               ::Vector{Float64}
    Inv_Move_shock      ::Vector{Float64}
    cash_on_hand        ::Vector{Float64}
    expected_earnings   ::Vector{Float64}
    
    debt                ::Vector{Float64}
    consumption         ::Vector{Float64}
    wealth              ::Vector{Float64}
    bequest             ::Vector{Float64}

    income              ::Vector{Float64}
    persistent          ::Vector{Float64}
    transitory          ::Vector{Float64}
    stock_market_shock  ::Vector{Float64}

    age                 ::Vector{Float64}
end

function build_solutions(para, σ_ω::Float64, κ::Matrix{Any}, s::Float64, F::Float64) 

    # Value function has an extra level for age due to bequest.
    # domain of val/pol funcs is [Inv_Move,IFC_Paid,aggregate_state, Ownership State, Mortgage Debt, Cash State, Age] 
    val_func      = zeros(Float64,2,2,para.nη, 2, para.nM, para.nX, para.T + 1 ) 
    c_pol_func    = zeros(Float64,2,2,para.nη, 2, para.nM, para.nX, para.T ) 
    M_pol_func    = zeros(Float64,2,2,para.nη, 2, para.nM, para.nX, para.T ) 
    FC_pol_func   = zeros(Float64,2,2,para.nη, 2, para.nM, para.nX, para.T ) 
    α_pol_func    = zeros(Float64,2,2,para.nη, 2, para.nM, para.nX, para.T ) 
    Move_pol_func = zeros(Float64,2,2,para.nη, 2, para.nM, para.nX, para.T ) 
    Own_pol_func  = zeros(Float64,2,2,para.nη, 2, para.nM, para.nX, para.T ) 

    sols = Solutions(val_func, c_pol_func, M_pol_func, FC_pol_func, α_pol_func, Move_pol_func, Own_pol_func, σ_ω , κ, s , F)
    return sols
end 

function Initialize_Model(σ_ω::Float64, κ::Matrix{Any}, s::Float64, F::Float64) 

    para = Model_Parameters()
    sols = build_solutions(para, σ_ω, κ, s, F)

    return para, sols 
end