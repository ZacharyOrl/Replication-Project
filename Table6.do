**********************************************************************
* Code which runs the regressions specified in Cocco (2005) Table 6 
* using data simulated in portfolio_choice_with_housing.jl
* each cohort experiences the same aggregate shock in the same real-life year (not the same age)
**********************************************************************
clear all 
* Import 
global indir "Z:/Replication/parameters"

import delimited "$indir/simulations_panel_eyeballed_from_paper_fixed_shocks.csv",clear 

********************************************************************************
* Merge in income (as the simulation didn't provide this)
********************************************************************************
preserve 
    * Merge in each education group, and append 
    
    forvalues i = 1/3 {
        clear
        import delimited "$indir/life_cycle_income_`i'_eyeballed_from_paper.csv"
        gen edu = `i' 
        gen age = _n 
		rename age_dummies det_inc
        tempfile tmp`i'
        save `tmp`i''
    }
    
    use `tmp1', clear 
    append using `tmp2'
    append using `tmp3'
    
    save "$indir/det_inc.dta", replace 
restore
	
* Merge in deterministic income levels 
merge m:1 age edu using "$indir/det_inc.dta"
drop if _merge != 3
drop _merge 
********************************************************************************
* Generate variables 
********************************************************************************
gen INC = det_inc * exp(persistent + transitory) // persistent is always 0.0 anyway due to 1990 being a on trend year.
gen LA = stocks + bonds
gen RE = wealth + debt - LA
gen FA = LA + RE 
gen TA = FA + expected_earnings

* Generate net worth measures 
gen LNW = LA - debt 
gen FNW = FA - debt 
gen TNW = TA - debt 

* Generate ratio measures
gen REFNW = RE/FNW 
gen MORTFNW = debt/FNW

gen SLA = stocks/LA 
gen SFA = stocks/FA 
gen STA = stocks/TA 

* Put age which is in model periods into correct groups
gen AGE = age*5 + 20

********************************************************************************
* Run Tobit Regressions 
********************************************************************************

* Stocks relative to liquid assets (portfolio share)
tobit SLA INC FNW AGE REFNW MORTFNW, ll(0.0)

* Stocks relative to financial assets
tobit SFA INC FNW AGE REFNW MORTFNW, ll(0.0)

* Stocks relative to total assets
tobit STA INC FNW AGE REFNW MORTFNW, ll(0.0)

* In levels 
tobit stocks INC AGE RE debt, ll(0.0)


