##############################################
# This file creates the deterministic income 
# lifecycle comparison plots and  
# house price comparison plots. 
##############################################

using Plots, CSV, DataFrames, LaTeXStrings

indir_parameters = "C:/Users/zacha/Documents/Research Ideas/Housing and Portfolio Choice/Replication/parameters"
outdir_images = "C:/Users/zacha/Documents/Research Ideas/Housing and Portfolio Choice/Replication/images"

##############################################

# Income 
##############################################
cd(indir_parameters)
# Import deterministic income by five-year age group 
# eyeballed from the paper. 

κ1 =  CSV.File("life_cycle_income_1_eyeballed_from_paper.csv").age_dummies ./ 1000
κ2 =  CSV.File("life_cycle_income_2_eyeballed_from_paper.csv").age_dummies ./ 1000
κ3 =  CSV.File("life_cycle_income_3_eyeballed_from_paper.csv").age_dummies ./ 1000

# Import the estimated equivalent from the PSID equivalent file:  

κ1_hat =  CSV.File("life_cycle_income_1.csv").age_dummies ./ 1000
κ2_hat =  CSV.File("life_cycle_income_2.csv").age_dummies ./ 1000
κ3_hat =  CSV.File("life_cycle_income_3.csv").age_dummies ./ 1000

# Import the age groups 
age_groups =  CSV.File("life_cycle_income_1_eyeballed_from_paper.csv").age_group

cd(outdir_images)

# Plot against each other 

plot(age_groups,κ1, color = "red", label = "No Highschool" )
plot!(age_groups, κ1_hat, color = "red", linestyle = :dash, label = "No Highschool est.")
plot!(age_groups,κ2, color = "blue", label = "Highschool")
plot!(age_groups, κ2_hat, color = "blue", linestyle = :dash, label = "Highschool est.")
plot!(age_groups,κ3, color = "green", label = "College")
plot!(age_groups, κ3_hat, color = "green", linestyle = :dash, label = "College est.", xlabel = "Age Groups", ylabel = L"Income ( Thousands of 1992 $)")

savefig("Determinstic_Income_Path.png") 

##############################################

# House Prices 
##############################################
cd(indir_parameters)

l_HP_psid = CSV.File("psid_log_house_price_index.csv").m_log_HOME_OWN_VAL_RD
HP_psid = exp.(l_HP_psid)
HP_psid_index = 100 * HP_psid ./HP_psid[12]
HP_BIS = CSV.File("C:/Users/zacha/Documents/Research Ideas/Housing and Portfolio Choice/Replication/BIS_HP_index.csv").QUSR628BIS_NBD19810101

years = CSV.File("psid_log_house_price_index.csv").YEAR

cd(outdir_images)
plot(years,HP_psid_index, ylabel = "Index (1981 = 100)", label = "House Price Index PSID")
plot!(years,HP_BIS, label = "House Price Index BIS")

savefig("House Price Comparison.png") 

