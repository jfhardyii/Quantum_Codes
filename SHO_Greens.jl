using LinearAlgebra
using Plots
using SparseArrays

# Construct the discritized Hamiltonian Matrix using the potential
function Hamiltonian(N,t,V)
    H = Matrix{Float64}(I, N, N) * 2t          # Start with 2t on the diagonal
    for i in 1:N
        H[i, i] += V[i]                        # Adds the potential to kinetic part of the Hamiltonian 
    end
    for i in 1:N-1
        H[i, i+1] = -t                         # Off-diagonal elements above the diagonal
        H[i+1, i] = -t                         # Off-diagonal elements below the diagonal
    end
    eigenvalues, eigenvectors = eigen(H)       # Finds Eigenvalues and Eigenvectors of Hamiltonian
    return(H,eigenvalues,eigenvectors)
end

function Normalized_Waveform(State,eigenvectors,a)
    phi = eigenvectors[:, State]              # Extract wavefunctions for the given number of states
    phi_squared = conj(phi) .* phi            # This finds the square of Phi which sum should be 1
    norm_factor = 1/sqrt(sum(phi_squared*a))  # This finds the normalization constant, if it is normaliezed sqrt(sum(phi_squared)) = 1
    phi .= phi * norm_factor                  # This gives new normalized Phi
    return(phi)
end

function Waveformed_Sqared(phi,a)
    phi_squared = conj(phi) .* phi
    phi_squared = phi_squared*a
    total_prob = sum(phi_squared) 
    println("Total Probability of the System: ",total_prob)
    return(phi_squared)
end
function expectation_x(phi,x,a)
    phi² = conj(phi) .* phi
    total_probability = sum(phi²)
    expectation_x = sum(phi² .* x)*a
    println("Expectation value of x: ", expectation_x)
end

function expectation_x²(phi,x,a)
    phi² = conj(phi) .* phi
    x²= x.^2
    expectation_x² = sum(phi² .* x²)*a
    println("Expectation value of x²: ", expectation_x²)
end

function greens_function(H,E,η)
    N = size(H,1)
    G = inv((E+η)*I(N)-H) #This is the Greens function
    return G
end


function density_of_states(H,Energy,η)
    dos = []
    total = length(Energy)
    for (i,E) in enumerate(Energy)
        G = greens_function(H,E,η)
        dos_value = -imag(tr(G))/pi
        push!(dos, dos_value)
        percent = 100 * i / total
        print("\rProgress: $(round(percent, digits=1))%")
        flush(stdout)
    end 
    display(plot(Energy, dos, labels = false, xlabel="Energy (E)", xticks=0:1:10, ylabel="Density of States (DOS)", title="Density of States in Simple Harmonic Occilator"))
    println()
    println("Press Enter to close image: ")
    readline()
    return dos 
end

function ldos(H, x, energies, η)
    N = size(H, 1)  # Number of grid points
    ldos_matrix = []  # Initialize an empty list to store LDOS for each energy
    total = length(energies)
    for (i,E) in enumerate(energies)
        G = greens_function(H, E, η)
        ρ = [-imag(G[i, i]) / π for i in 1:N]  # LDOS at each position
        push!(ldos_matrix, ρ)
        percent = 100 * i / total
        print("\rProgress: $(round(percent, digits=1))%")
        flush(stdout)
    end
    return hcat(ldos_matrix...)  # Combine LDOS arrays into a 2D matrix
end

function Plot_Waveform(ψ,x,State)
    for i in 1:State
        plot(x, ψ, label="ψ_$i(x)", xlabel="Position (x)", ylabel="ψ(x)",xlim=[-L,L])
    end
    display(plot!(title="Wavefunctions in an Simple Harmonic Occilator", legend=:topright))
    println("Press Enter to close image: ")
    readline()
end

function Plot_Waveform²(ψ,x,State)
    for i in 1:State
        plot(x, ψ, label="|ψ_$i(x)|²", xlabel="Position (x)", ylabel="|ψ(x)|²",xlim=[-L,L])
    end
    display(plot!(title="PDF an Simple Harmonic Occilator", legend=:topright))
    println("Press Enter to close image: ")
    readline()
end

function self_energies(N, m, ħ, t0, V, V2, energy)
# To calculate ka, this was taken from Datta's book equation 8.1.5: E = Ec + 2t₀(1 - cos(ka)) => ka = acos(((Ec - E)/2t₀) + 1)
# t₀ is the coupling constant which is equivalent to the kinetic energy.
# E is the energy of the particle.
# Ec is conduction band energy (In the case of the quantum system, it is the Potential Energy (V))
# k is the wavevector
# a is the lattice constant (The gridspacing of the quantum system)

    # ka1 is calcuated for the first Self-energy (Σ1) of the Left Lead by using the first value of V.
    x = ((V - energy) / (2*t0)) + 1
    if abs(x) > 1
        ka1 = im * acosh(abs(x)) # This equation is used when ka is not in the bound of [-1,1] which suggests it is in an Evervecent waveform when the particle is in a forbidden region
    else
        ka1 = acos(x) # This is used when ka is in the bounds of [1,1] when the particle acts as a propagating wave
    end

    # ka2 is calcuated for the second Self-energy (Σ2) of the Right Lead by using the last value of V.
     y = ((V2 - energy) / (2*t0)) + 1
    if abs(y) > 1
        ka2 = im * acosh(abs(y)) # Refer to line 137
    else
        ka2 = acos(y) # Refer to line 139
    end

    # Self-energy matrices
    Σ1 = spzeros(ComplexF64, N, N)
    Σ2 = spzeros(ComplexF64, N, N)
    
    # Self-energies given by Datta's eq (8.1.7a) refer to page 222
    Σ1[1, 1] = -t0 * exp(1*im * ka1)  # Lead 1 Self-energy
    Σ2[N, N] = -t0 * exp(1*im * ka2)  # Lead 2 Self-energy
    
    return Σ1, Σ2
end

function compute_transmission(E, H, Σ1, Σ2)
    # This gives the Broadening matricies defined on pg 217 of Datta's book
    Γ1 = 1*im * (Σ1 - Σ1')
    Γ2 = 1*im * (Σ2 - Σ2')
    N = size(H,1)
    G = inv((E+η)*I(N) - H - Σ1 - Σ2) #This is the retarded Greens function (9.1.5)
    T = real(tr(Γ1 * G * Γ2 * G')) #Equation for transmission (9.1.10) 
    return T
end

function transmission(N, m, ħ, t₀, Energy, V, H)
    transmissions = []
    temp_trans = []
    # Compute transmission for each energy
    total = length(Energy)
    for (i,E) in enumerate(Energy)
            Σ1, Σ2 = self_energies(N,m,ħ, t₀, V[1],V[total], E)
            T = compute_transmission(E, H, Σ1, Σ2)
            push!(transmissions, T)
            percent = 100 * i / total
            print("\rProgress: $(round(percent, digits=1))%")
            flush(stdout)
    end
    display(plot(Energy, transmissions,label=false, xlabel="Energy (E)", ylabel="Transmission (T(E))", title="Transmission vs Energy"))
    println()
    println("Press Enter to close image: ")
    readline()
    return(transmissions)
end

############################################################################################################################################
######################################################## Parameters ########################################################################
############################################################################################################################################

L = 10.0                                   # Width of the well
println("Enter Number of points in the grid (2000 is ideal for no numerical errors, but more datapoint equals more time): ")
N = parse(Int,readline())                  # Number of points in the grid (discretization)
ħ = 1                                      # Plank's with value of 1 to remain dimensionless
m = 1                                      # Mass with value of 1 to remain dimensionless
w = 1                                      # Angular Frequency with value of 1 to remain dimensionless 
x = range(-L , length=N, stop=L)           # x is the values of position
a = x[2] - x[1]                            # Update grid spacing
t = ħ^2/(2*m*a^2)                          # Constant for kinetic energy term
V = 0.5 * m * w^2 .* x.^2                  # This is the potential energy
η = (1E-10)*im                             # This is the small imaginary constant for the Green's Function
energies=range(45, length=N, stop = 65)     # This is the energy range for the Greens Function
H, eigenvalues, eigenvectors = Hamiltonian(N,t, V)
#############################################################################################################################################
println("Enter 1 to select/view a Waveform, 2 for DOS, 3 for LDOS, and 4 for Transmission Curve: ")
User_Choice = parse(Int,readline()) #
if User_Choice == 1
    println("Enter a Waveform to Examine: ")
    num_states = parse(Int,readline())         # This chooses the state to use
    println("Eigen Energy Value of the Waveform: ",eigenvalues[num_states])
    phi_new = Normalized_Waveform(num_states,eigenvectors,a)
    phi_squared = Waveformed_Sqared(phi_new,a)
    expectation_x(phi_new,x,a)
    expectation_x²(phi_new,x,a)
    Plot_Waveform(phi_new,x,num_states)
    Plot_Waveform²(phi_squared,x,num_states)
    display(plot(x, V,label=false, xlabel="Position (x)", ylabel="Potential (V)", title="Potential vs Postiion"))
    println("Press Enter to close image: ")
    readline()
elseif User_Choice == 2
    dos_values = density_of_states(H, energies, (7E-2)*im)
elseif User_Choice == 3
    ldos_values = ldos(H, x, energies, (7E-2)*im)
    display(plot(
        energies,x, ldos_values,
        st=:surface, ylabel="Position (x)", xlabel="Energy (E)", zlabel="LDOS",
        title="Local Density of States"
    ))
    println()
    println("Press Enter to close image: ")
    readline()
elseif User_Choice == 4
    transmissions=transmission(N, m, ħ, t, energies, V, H)
else
    println("Error: User Input Not Recognized")
end




