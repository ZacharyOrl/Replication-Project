# Finds the value of housing services from owning which
# matches homeownership rates at age 45-59 in the model 
# to homeownership rates at age 45-49 in the PSID + over 1970 - 1992. 
# data moment = 86%.  

# Finds the Stock Market Participation Fee required to match the 
# stock market participation rate in the PSID over age 25-29 over 1970-1992
# data moment reported in Cocco was 26%. 
function calibrate_model(data_moments::Vector{Float64})

    error = 1.0 

    tol = 10^-5 
    s_guess = 1.2 
    F_guess = 2000.0
    n = 1

    para, sols_hs = Initialize_Model(σ_ω_hs, κ_hs, s_guess, F_guess)

    while error > tol 

        s = s_guess 
        F = F_guess

        sols_hs.s = s 
        sols_hs.F = F 


        # Solve the model

        # High school
        Solve_Retiree_Problem(para, sols_hs)
        Solve_Worker_Problem(para, sols_hs)

        # Simulate the model 
        S = 100000
        sim_hs = simulate_model(para, sols_hs, S, 2)

        cols = [:bonds,:stocks,:stock_share,:sm_entry,:IFC_paid,:own,
         :moved,:Inv_Move_shock,:cash,:expected_earnings, :debt, :cons,:wealth,:bequest, :income,
        :persistent,:transitory,:sm_shock]


        # Allowing η to vary between simulations 
        mat_hs   = sim_to_matrix(sim_hs)
        df = DataFrame(mat_hs, cols)

        age45 = df[df.age .== 5  , :]
        age25 = df[df.age .== 1  , :]
        # The moments I want to match are the homeownership rate at age 45. 
        # and stock participation at age 25. 
        model_moment = [mean(age45.own), mean(age25.IFC_paid) ]

        # The error is the sum of squared differences from the data moment to the model moment
        error = sum((data_moments .- model_moment).^2)

        # Update s_guess in the same direction as the miss
        if model_moment[1] == 0.0 
            s_guess = 2 * s_guess 
        else 
            s_guess = 0.5 * s_guess + 0.5 * s_guess *( data_moments[1]/(model_moment[1]) )
        
        end
        
        if model_moment[2] == 0.0 
            F_guess = 0.5 * F_guess 
        else 
            F_guess = 0.5 * F_guess +  0.5 * F_guess * ((model_moment[2])/data_moments[2] )
        end 

        n += 1
        println("Model Moment is: ", model_moment, " n is: ", n, " s guess is : ", s_guess, " F guess is : ", F_guess)
    end 

    println("Final s is: ",s, " Final F is: ", F)
end 


