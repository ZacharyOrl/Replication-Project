###########################################
# Creates the final event study plots for 
# the buy vs. rent extension of 
# portfolio choice in the presence of housing
# event study of the impact of a small 
# or large house price shock 
# on behavior. 
###########################################

using Parameters, DelimitedFiles, CSV, Plots, Distributions,LaTeXStrings, Statistics, DataFrames

indir_parameters = "parameters"
cd(indir_parameters)

############################################
# Clean 
############################################
coefficients = DataFrame(CSV.File("event_study_coefs.csv"))

coefficients_own1 = coefficients[1:8,:]
coefficients_own2 = coefficients[9:16,:]
coefficients_entry_s1 = coefficients[17:24,:]
coefficients_entry_s2 = coefficients[25:32,:]
coefficients_wealth_s1 = coefficients[33:40,:]
coefficients_wealth_s2 = coefficients[41:48,:]

function add_treat_row(df::DataFrame; coef_name = "treat_time_m1")
    if any(df.coef .== coef_name)
        return df            # nothing to do
    end
    row = DataFrame(; coef = coef_name, beta = 0.0, se = 0.0)

    pos = findfirst(df.coef .== "treat_time_m2")
    pos === nothing && (pos = 1)       # fallback: stick it at the top

    vcat(df[1:pos, :], row, df[pos+1:end, :]; cols = :union)
end

coefficients_own1 = add_treat_row(coefficients_own1)
coefficients_own2 = add_treat_row(coefficients_own2)

coefficients_entry_s1 = add_treat_row(coefficients_entry_s1)
coefficients_entry_s2 = add_treat_row(coefficients_entry_s2)

coefficients_wealth_s1 = add_treat_row(coefficients_wealth_s1)
coefficients_wealth_s2 = add_treat_row(coefficients_wealth_s2)
#############################################
# Plot 
#############################################
event_time = 5 .* [-2,-1,0,1,2,3,4,5,6]
 
# Home Ownership 
plot(event_time,coefficients_own1.beta, 
label = "Small housing shock", 
xlabel = "Years from Shock", 
ylabel = "Homeownership Proportion")

plot!(event_time,coefficients_own2.beta, label = "Large housing shock")
savefig("Ownership Event Study.png")

# Stock Market Entry rates
plot(event_time,coefficients_entry_s1.beta,
label = "Small housing shock", 
xlabel = "Years from Shock", 
ylabel = "Stock Market Entry Proportion")
plot!(event_time,coefficients_entry_s2.beta,
 label = "Large housing shock")

savefig("Entry Event Study.png")

# Wealth Accumulation
plot(event_time,coefficients_wealth_s1.beta,
label = "Small housing shock", 
xlabel = "Years from Shock", 
ylabel = L"Wealth (1992 $)")
plot!(event_time,coefficients_wealth_s2.beta,
label = "Large Housing Shock")

savefig("Wealth Event Study.png")

