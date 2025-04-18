"""
    compound(rate::Float64, T::Int)

Returns the compounded factor over `T` periods given an annual rate `rate`.
For example, compound(0.05, 3) returns (1 + 0.05)^3 = 1.157625.
"""
function compound(rate::Float64, T::Int)
    return (1 + rate)^T
end