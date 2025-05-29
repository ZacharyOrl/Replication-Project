####################################################
# This file recreates tables 4 and 5 in Cocco (2005) 
####################################################
using Plots, CSV, DataFrames, Statistics

indir_parameters = "parameters"
cd(indir_parameters)

simulated_data = DataFrame(CSV.File("simulations_panel_eyeballed_from_paper.csv"))
simulated_data = simulated_data[simulated_data.cons .!=   0.0, :]

# Liquid Assets 
simulated_data.liquid = simulated_data.bonds .+ simulated_data.stocks

# Financial Assets 
simulated_data.financial = simulated_data.wealth + simulated_data.debt
simulated_data.real_estate = simulated_data.wealth + simulated_data.debt - simulated_data.liquid

# Total Assets 
simulated_data.total = simulated_data.liquid + simulated_data.real_estate + simulated_data.expected_earnings
##########################################################################
# Table 4 Portfolio shares by financial assets
##########################################################################
function compute_stats_Table4(df::DataFrame,group::String, greater_or_less::Int64)

    if greater_or_less == 0 
       assets_filtered_dataset =  df[simulated_data[!,group]  .< 100000, :]
    else 
        assets_filtered_dataset =  df[simulated_data[!,group]  .>= 100000, :]
    end 

    stock_shr = assets_filtered_dataset.stocks ./(assets_filtered_dataset[!,group] )
    bills_shr = assets_filtered_dataset.bonds ./(assets_filtered_dataset[!,group] )
    real_est_shr = assets_filtered_dataset.real_estate ./(assets_filtered_dataset[!,group] )
    human_cap_shr = assets_filtered_dataset.expected_earnings ./(assets_filtered_dataset[!,group] )
    stock_participation = assets_filtered_dataset.IFC_paid
    debt_shr = assets_filtered_dataset.debt ./ (assets_filtered_dataset[!,group] )

    st = mean(stock_shr)
    bi = mean(bills_shr)
    re = mean(real_est_shr)
    hc = mean(human_cap_shr)
    part = mean(stock_participation)
    dt = mean(debt_shr)

    return vcat(st, bi , re , hc, part, dt)
end

group_list = ["liquid", "financial", "total"]
asset_range_list = [0,1]

Table_4 = zeros(6,6)
lin_indx = LinearIndices((2,3))
for i in eachindex(group_list) 
    group = group_list[i]
    for j in eachindex(asset_range_list)
        asset_range = asset_range_list[j]
        index = lin_indx[j,i]
        Table_4[:,index] = compute_stats_Table4(simulated_data,group,asset_range)
    end 
end 
colnames = [:liquid_under, :liquid_over,
            :financial_under, :financial_over,
            :total_under, :total_over]
out_4 = DataFrame(Table_4,colnames)
CSV.write("Table_4_estimates.csv", 
         out_4)
##################################################
# Table 5. Portfolio composition over the lifecycle 
##################################################
function compute_stats_Table5(df::DataFrame,group::String,ages::Vector{Int64})
    max_age = maximum(ages)
    min_age = minimum(ages)
    temp = df[df.age  .<= max_age , :]
    age_filtered_dataset = temp[temp.age .>= min_age,:]

    stock_shr = age_filtered_dataset.stocks ./age_filtered_dataset[!,group]
    bills_shr = age_filtered_dataset.bonds ./age_filtered_dataset[!,group]
    stock_participation = age_filtered_dataset.IFC_paid 
    real_est_shr = age_filtered_dataset.real_estate ./age_filtered_dataset[!,group]
    human_cap_shr = age_filtered_dataset.expected_earnings ./age_filtered_dataset[!,group]
    debt_shr = age_filtered_dataset.debt ./ age_filtered_dataset[!,group]

    st = mean(stock_shr)
    bi = mean(bills_shr)
    re = mean(real_est_shr)
    hc = mean(human_cap_shr)
    part = mean(stock_participation)
    dt = mean(debt_shr)

    return vcat(st, bi , re , hc, part, dt)
end 

group_list = ["liquid", "financial", "total"]
age_list = [[1,2],[3,4,5],[6,7,8],[9,10]]

Table_5 = zeros(6,12)
lin_indx = LinearIndices((4,3))
for i in 1:length(group_list) 
    group = group_list[i]
    for j in 1:length(age_list)
        age_range = age_list[j]
        index = lin_indx[j,i]
        Table_5[:,index] = compute_stats_Table5(simulated_data,group,age_range)
    end 
end 
colnames = [:liquid_30orless, :liquid_35to45, :liquid_50to60, :liquid_65plus, 
                    :financial_30orless, :financial_35to45, :financial_50to60, :financial_65plus, 
                    :total_30orless, :total_35to45, :total_50to60, :total_65plus]

out_5 = DataFrame(Table_5,colnames)
CSV.write("Table_5_estimates.csv", out_5)