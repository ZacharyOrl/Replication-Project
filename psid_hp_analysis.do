*********************************************
* This file cleans the PSID SHELF file 
* in order to generate a house price index 
* for the years 1970 - 1992 
* and find log house price growth and log house price variance.  

*********************************************

global indir "C:\Users\zacha\Documents\Research Ideas\Housing and Portfolio Choice\Replication"

global outdir_parameters "$indir/parameters"
global outdir_images "$indir/images"

/*
cd "$indir/PSIDSHELF_1968_2021_LONG_7.9_GB" 

* Clean raw dataset and re-save due to its massive size. 
use PSIDSHELF_1968_2021_LONG.dta,clear 

* Keep only relevant variables and years
keep if inrange(YEAR,1970,1992)
keep if SAMPLE == 1 // Only Main Sample PSID - drop SEO / refresher samples 
keep  ID LINEAGE FUID HHDID FW IW  YEAR RESPONDENT  REL  SAMPLE DEMO_SEX DEMO_BIRTH_YEAR DEMO_AGE_REP  EDU_YEAR EDU_YEAR_RP  EDU_YEAR_MAX_RP FAM_SIZE FAM_PARTNERED EARN_TOT_ND FINC_TOT_ND FINC_TOT_RD HOME_STAT HOME_OWN_VAL_ND  HOME_OWN_VAL_NDF HOME_OWN_VAL_RD HOME_OWN_VAL_RDF HOME_OWN_MOR_ANY_1M HOME_OWN_MOR_ANY_2M HOME_OWN_MOR_ANY_3M HOME_OWN_MOR_VAL_1M_ND HOME_OWN_MOR_VAL_2M_ND

* Keep only observations who have information on where they are living 
keep if !missing(HOME_STAT)

save "$indir/PSID_SHELF_cleaned.dta",replace 

*/ 

use "$indir/PSID_SHELF_cleaned.dta", clear

* Merge in CPI 
preserve 

	clear all 
	
	import delimited "$outdir_parameters/CPI.csv"
	
	gen YEAR = yofd(date(observation_date, "YMD"))
	drop observation_date 
	
	rename cpiaucsl price_level
	tempfile infl
	
	save `infl'

restore 

merge m:1 YEAR using `infl'
drop if _merge == 2
drop _merge 

gen HOME_OWN_VAL_RD_1992 = 100* HOME_OWN_VAL_ND/price_level

* Find the distribution of self-reported home values in 1992 
preserve 

* Consider only 1 ob per household, the household head
	keep if REL == 1 & YEAR == 1992
	
	*keep if YEAR == 1992
	*duplicates drop HHDID, force
	
	keep if HOME_STAT == 1 // Owns home 
	summ HOME_OWN_VAL_RD_1992, d // Values are close but not exactly the same as Cocco (4300, 15000, 26000) - no change if I use dwelling definition instead.
	
restore 
	
* Find log house price growth 


* Consider only 1 ob per household, the household head
	keep if REL == 1	
	keep if HOME_STAT == 1 // Owns home 
	gen log_HOME_OWN_VAL_RD = log(HOME_OWN_VAL_RD_1992)
	
	* Generate Cocco's price index
	bysort YEAR: egen m_log_HOME_OWN_VAL_RD = mean(log_HOME_OWN_VAL_RD)
	
	*  - probably better to use PSID weights when doing this
	bysort YEAR: egen wsum = total(log_HOME_OWN_VAL_RD * FW)
	bysort YEAR: egen wtot = total(FW)
	gen m_log_HOME_OWN_VAL_RD_w = wsum / wtot
	
	* Generate the rate of increase in log real house prices 
	bysort YEAR: gen n = _n 
	keep if n == 1 
	drop n
	sort YEAR 
	
	* House price index (1981 = 100) 
	gen temp = m_log_HOME_OWN_VAL_RD if YEAR == 1992
	egen temp2 = max(temp)
	
	gen HP_INDEX_COCCO = 100 * exp(m_log_HOME_OWN_VAL_RD)/exp(temp2)
	drop temp temp2 
	
	gen temp = m_log_HOME_OWN_VAL_RD_w if YEAR == 1981
	egen temp2 = max(temp)
	gen HP_INDEX_WEIGHTED = 100 * exp(m_log_HOME_OWN_VAL_RD_w)/exp(temp2)
	drop temp temp2 
	
	* HP_INDEX_WEIGHTED and HP_INDEX_COCCO look pretty similar to the true series from the BIS. 
	* but not similar at all to Cocco's series 
	
	* House prices grew by 1.57% under Cocco's measure rather than 1.59% reported 
	egen b_Cocco = mean(100 * (m_log_HOME_OWN_VAL_RD - m_log_HOME_OWN_VAL_RD[_n-1]))
	
	* House prices grew by 1.39% under the weighted measure rather than the 1.59% reported. 
	egen b_weighted = mean(100 * (m_log_HOME_OWN_VAL_RD_w - m_log_HOME_OWN_VAL_RD_w[_n-1]))
	
	* House price standard deviation 
	summ m_log_HOME_OWN_VAL_RD_w // SD of .111199 rather than 0.062 in Cocco 
	summ m_log_HOME_OWN_VAL_RD // SD of .1291317 rather than 0.062 in Cocco 
	
	* Depends on how you do it. In Cocco's model, he creates innovations as deviations from a time trend 
	* So do this regression and find the variance of the residual. 
	reg m_log_HOME_OWN_VAL_RD_w YEAR
	predict fitted_w 
	gen residual_w = m_log_HOME_OWN_VAL_RD_w - fitted_w
	
	summ residual_w // SD of .048 rather than 0.062 in Cocco 
	
	reg m_log_HOME_OWN_VAL_RD YEAR
	predict fitted
	gen residual = m_log_HOME_OWN_VAL_RD - fitted
	
	summ residual // SD of .0517424 rather than 0.062 in Cocco 	
	
	twoway (line fitted YEAR, lcolor(blue) lpattern(solid))     ///
	(line m_log_HOME_OWN_VAL_RD YEAR, lcolor(red) lpattern(dash)), ///
    xtitle("Year") ///
    ytitle("Log House Price Index") ///
    title("Estimated Trend vs. Realized Value") ///
    legend(order(1 "Estimated Time Trend" 2 "Realized Value") ///
           rows(1) ring(0) position(6))
	graph export "$outdir_images/House Price Index Fluctuations.png", width(2000) replace	 
	
	preserve 
		keep m_log_HOME_OWN_VAL_RD YEAR 
		export delimited "$outdir_parameters/psid_log_house_price_index.csv", replace
	restore 
	
	* Construct a five-year home value index	
	egen year_group = cut(YEAR), at(1970(5)1992)
	bysort year_group: egen m_log_HOME_OWN_VAL_RD_five = mean(log_HOME_OWN_VAL_RD)
	
	reg m_log_HOME_OWN_VAL_RD_five year_group
	predict fitted_five
	gen residual_five = m_log_HOME_OWN_VAL_RD_five - fitted_five
	bysort year: gen n = _n
	
	summ residual_five if n == 1
	drop n
	* Generate classification for which five-year group corresponds to which aggregate realization. 
	
	* Find the overall deviation from trend over the five year group. 
	bysort year_group: egen sum_resid_year_group = sum(residual) 
	
	bysort year_group: gen n =_n 
	keep if n == 1
	
	keep year_group sum_resid_year_group
	export delimited "$outdir_parameters/aggregate_realization.csv", replace 
	

	