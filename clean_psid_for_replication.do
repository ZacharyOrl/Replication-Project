*********************************************
* This file cleans the PSID equivalent file 
* in order to generate lifecycle predictable labor income 
* and a residual component. 

*********************************************
global indir "C:\Users\zacha\Documents\Research Ideas\Housing and Portfolio Choice\Replication"

global outdir_parameters "$indir/parameters"
global outdir_images "$indir/images"

* Set earnings cutoff requirements 

local cutoff = 5000 // Close to the 1% level for the sample. 
local minimimum_times_satisified = 5

****************************************************************
* Variables Used: 
* Key: 
* x11101LL = Individual identifier 
* year = year of survey

* Income: 
* i11102 = Household Post-Government income
* i11104 = Household Income from Asset Flows
* i11109 = Total Household Taxes
* i11118 = Household Windfall income

* Try another method for computing total labor income 
* i11103 = Household labor income 
* i11106 = Household Private Transfers 
* i11107 = Household Public Transfers 
* i11108 = Household Social Security Pensions 
* Demographics (for Deterministic Component):
* d11102LL = Gender 
* d11101 = Age 
* d11104 = Marital Status (1 = Married Living with Partner, 2 = Single not living with partner, 3 = Widowed, 4 = Divorced, not living with Partner, 5 = Separated, not living with Partner, 6 = Not living with partner & over 18, 7= not living with partner & under 18 )
* d11106 = Number of Persons in Household 
* d11108 = Education with respect to high school (1 = Less than HS, 2 = HS, 3 = More than HS)

* Additional vars used for sample restrictions: 
* x11104LL = Oversample identifier (11 = Main Sample, 12 = SEO)
* d11105 = Relationship to Household Head (Head = 1,Partner = 2, Child = 3, Relative = 4, Non-Relative = 5)
* 
* w11103 = Longitudinal Weight
* x11102 = Household identifier 
*********************************************
cd "$indir" 

use pequiv_long.dta,clear 
* Rename variables
local oldnames x11101LL i11110 i11113 i11104 i11114 i11118 d11102LL d11101 d11104 d11106 x11104LL d11105 d11108 w11103 x11102 i11103 i11106 i11107 i11108
local newnames person_id indiv_labor_inc post_gov_income asset_income taxes windfall_income gender age married hh_size oversample_id relationship_head education wgt hh_id  hh_labor_income hh_pvt_transf hh_pub_transf hh_ss

local i = 1
foreach oldvar in `oldnames' {
    local newvar: word `i' of `newnames'
    rename `oldvar' `newvar'
    local ++i
}

keep person_id year indiv_labor_inc post_gov_income asset_income  taxes windfall gender age married hh_size oversample_id relationship_head education wgt hh_id hh_labor_income hh_pvt_transf hh_pub_transf hh_ss
sort hh_id year
* Generate labor income measure 
gen hh_income = hh_labor_income + hh_pvt_transf + hh_pub_transf + hh_ss
* Income is reported in nominal dollars, adjust for inflation using annual price index for all urban consumers in US. 
preserve 

	clear 
	
	import delimited "CPI.csv"
	
	gen year = yofd(date(observation_date, "YMD"))
	drop observation_date 
	
	rename cpiaucsl price_level
	tempfile infl
	
	save `infl'

restore 

merge m:1 year using `infl'
drop if _merge == 2
drop _merge 

replace hh_income = 100* hh_income/price_level

* Construct the cohort 
gen cohort = year - age

* Dummy if the household head is male 
gen temp = (gender == 1 & relationship_head == 1)
bysort hh_id year: egen male_head = max(temp)
drop temp 

* Impose sample restrictions 
keep if inrange(year,1970,1992) // year must be between 1970 and 1992
keep if oversample_id == 11 // Drop the SEO sample 
keep if male_head == 1
keep if inrange(age,25,74)
keep if relationship_head == 1

* Drop if demographics are missing
gen missing_dem = 0
foreach var in `newnames' {
	
	replace missing_dem = 1 if missing(`var')
}

drop if missing_dem == 1 

drop missing_dem 	

replace married = (married == 1)

* Generate five-year age_group 
egen age_group = cut(age), at(25(5)75)

* Drop when less than five years of income are observed income when all five years are observed
bysort age_group person_id(age): gen tot = _N

drop if tot < 5

* Generate five-year income, discounting each subsequent year's labor income by (1+r)
bysort age_group person_id(age): gen count = _n

local r = 0.05 // Discounting value used for combining five years of labor income into one amount. 

gen temp = (1 + `r')^(1 - count) * hh_income

bysort age_group person_id (year): egen five_year_hh_income = total(temp)

sort hh_id year

*collapse (last) five_year_hh_income married hh_size education wgt (first) year cohort hh_id, by(age_group person_id)
* Partial out deterministic income component 
*gen log_hh_inc = log(five_year_hh_income)

gen log_hh_inc = log(five_year_hh_income)
bysort hh_id age_group: gen n =_n
drop age
keep if n == 1

forvalues i = 1/3 {
	
	reghdfe log_hh_inc i.age i.married hh_size if education == `i' [pweight = wgt], a(hh_id)
	// Extract coefficient estimates and standard errors
	matrix b_est = e(b)
	
	// Create a new coefficient matrix with the reference period included
	matrix b_coeffs = J(1, 10, .)
	
	forvalues j = 1/10 {
		matrix b_coeffs[1,`j'] = exp(b_est[1, `j'] + b_est[1, 14])
	}
	
	// Combine the coefficients into 10 5 - year values, using a discount rate of 5% 
	matrix b_plot = J(1, 10, .)
	/*
	forvalues j = 1/8 {
		matrix b_plot[1,`j'] = b_coeffs[1, 5*`j' - 4] + 0.95 * b_coeffs[1, 5*`j' - 4 + 1] + 0.95^2 * b_coeffs[1, 5*`j' - 4 + 2] + 0.95^3 * b_coeffs[1, 5*`j' - 4 + 3] + 0.95^4 * b_coeffs[1, 5*`j' - 4 + 4]
	}
	*/
	
	matrix b_plot = b_coeffs 
	
	// Add retirement labor income equal to average labor income 
	
	reghdfe log_hh_inc i.married hh_size if education == `i' [pweight = wgt], a(hh_id)
	matrix b_est = e(b)
	
	forvalues j = 9/10{

			matrix b_plot[1,`j'] = exp(b_est[1, 4])
	}
	
		// Create the plot
		coefplot matrix(b_plot),  ///
		vertical ///
		mcolor(navy%70) msymbol(circle) ///
		xtitle("Age") ///
		connect(l) lcolor(black) lwidth(medthick) /// Added connecting line
		ytitle("Earnings" "(1992 $)") ///
		title("Five-year earnings for Education Group"  "`i'") ///
		note("Note:") ///
		legend(off)

		 graph export "$outdir_images/age_profile_psid_`i'.png", width(2000) replace
		 
		 * Export 
		 preserve 
			 clear 
			 matrix b_plot = b_plot'  
			 svmat b_plot ,names(col)
			 
			* Create age group labels
			gen age_group = ""
			local start = 25
			forvalues k = 1/10 {
				local lower = `start' + (`k' - 1) * 5
				local upper = `lower' + 5
				replace age_group = "`lower'-`upper'" if _n == `k'
			}

			rename r1 age_dummies
			 
			 export delimited "$outdir_parameters/life_cycle_income_`i'.csv", replace 
		 restore

}

drop n

* Estimate the labor market process parameters 
preserve 
	* Aggregate shock variance 
	reghdfe log_hh_inc i.age i.married hh_size [pweight = wgt], a(hh_id)
	predict fitted
	gen residual = log_hh_inc - fitted
	bysort year: egen log_y_tilda = mean(residual)
	bysort year: gen n = _n
	summ log_y_tilda if n == 1
	local sigma_eta_2 = (r(sd))
	display `sigma_eta_2'
	
	sort year 
	gen lag_log_y_tilda = log_y_tilda[_n-1]
	
	reg log_y_tilda lag_log_y_tilda if n ==1
	local phi_hat = _b[lag_log_y_tilda]
	
	* Idiosyncratic shock variance 

	* No high school 
	reghdfe log_hh_inc i.age i.married hh_size if education == 1 [pweight = wgt], a(hh_id)
	predict fitted_edu1
	gen residual_edu1 = log_hh_inc - fitted_edu1 if education == 1
	summ residual_edu1 if education == 1

	local sigma_w1_2 = sqrt((r(Var)) - (`sigma_eta_2')^2)
	display `sigma_w1_2'

	* High School 
	reghdfe log_hh_inc i.age i.married hh_size if education == 2 [pweight = wgt], a(hh_id)
	predict fitted_edu2
	gen residual_edu2 = log_hh_inc - fitted_edu2 if education == 2
	summ residual_edu2 if education == 2

	local sigma_w2_2 = sqrt((r(Var)) - (`sigma_eta_2')^2)
	display `sigma_w2_2'

	* College 
	reghdfe log_hh_inc i.age i.married hh_size if education == 3 [pweight = wgt], a(hh_id)
	predict fitted_edu3
	gen residual_edu3 = log_hh_inc - fitted_edu1 if education == 3
	summ residual_edu3 if education == 3

	local sigma_w3_2 = sqrt((r(Var)) - (`sigma_eta_2')^2)
	display `sigma_w3_2'
	
	scalar sigma_eta_2 = `sigma_eta_2'
	scalar sigma_w1_2  = `sigma_w1_2'
	scalar sigma_w2_2  = `sigma_w2_2'
	scalar sigma_w3_2  = `sigma_w3_2'
	scalar phi_hat     =  `phi_hat'
	
	clear
	
	matrix params = (`phi_hat', `sigma_eta_2', `sigma_w1_2', `sigma_w2_2', `sigma_w3_2' )
	matrix colnames params = phi_hat sigma_eta_2 sigma_w1_2 sigma_w2_2 sigma_w3_2

	svmat double params, names(col)

	export delimited "$outdir_parameters/shock_parameters.csv", replace 

restore