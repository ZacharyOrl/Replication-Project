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
local oldnames x11101LL i11110 i11113 i11104 i11109 i11118 d11102LL d11101 d11104 d11106 x11104LL d11105 d11108 w11103 x11102
local newnames person_id indiv_labor_inc post_gov_income asset_income windfall_income taxes  gender age married hh_size oversample_id relationship_head education wgt hh_id

local i = 1
foreach oldvar in `oldnames' {
    local newvar: word `i' of `newnames'
    rename `oldvar' `newvar'
    local ++i
}

keep person_id year indiv_labor_inc post_gov_income asset_income  taxes windfall_income  gender age married hh_size oversample_id relationship_head education wgt hh_id

* Generate labor income measure 
gen hh_income = post_gov_income + taxes - asset_income
* Income is reported in nominal dollars, adjust for inflation using annual price index for all urban consumers in US. 
preserve 

	clear all 
	
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

* Impose sample restrictions 
keep if inrange(year,1970,1992) // year must be between 1970 and 1992
keep if oversample_id == 11 // Drop the SEO sample 
keep if relationship_head == 1 // Consider household heads only 
keep if gender == 1 // Consider only male HH heads
keep if inrange(age, 25,74)	// Consider only observations used by Cocco

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

bysort age_group person_id(age): egen five_year_hh_income = total(temp)

* Collapse 

collapse (last) five_year_hh_income married hh_size education wgt (first) year cohort hh_id, by(age_group person_id)
* Partial out deterministic income component 
gen log_hh_inc = log(five_year_hh_income)

forvalues i = 1/3 {
	reghdfe log_hh_inc i.age i.married hh_size if education == `i' [pweight = wgt], a(hh_id)
	
	// Extract coefficient estimates and standard errors
	matrix b_est = e(b)
	
	// Create a new coefficient matrix with the reference period included
	matrix b_plot = J(1, 10, .)
	
	forvalues j = 1/10 {
		matrix b_plot[1,`j'] = exp(b_est[1, `j'] + b_est[1, 14])
	}
	
		// Create the plot
		coefplot matrix(b_plot),  ///
		vertical ///
		mcolor(navy%70) msymbol(circle) ///
		xtitle("Years after layoff") ///
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




