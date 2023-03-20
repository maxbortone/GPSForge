using ArgParse
using ITensors


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--N"
            help = "Number of sites in the chain"
            arg_type = Int
            default = 32
        "--t"
            help = "Hopping strength"
            arg_type = Float64
            default = 1.0
        "--U"
            help = "Interaction strength"
            arg_type = Float64
            default = 1.0
        "--pbc"
            help = "Periodic boundary conditions"
            arg_type = Bool
            default = true
    end

    return parse_args(s)
end

#
# DMRG calculation of the ground state wavefunction, and spin densities
# for the 1D Hubbard model at half-filling
#

function main()
    parsed_args = parse_commandline()
    N = parsed_args["N"]
    Npart = N
    t = parsed_args["t"]
    U = parsed_args["U"]
    pbc = parsed_args["pbc"]
    nsweeps = 10
    
    println("Running DMRG for N=$N, t=$t, U=$U and pbc=$pbc")

    sites = siteinds("Electron", N; conserve_qns=true)
    
    ampo = OpSum()
    for b in 1:(N-1)
        ampo += -t, "Cdagup", b, "Cup", b + 1
        ampo += -t, "Cdagup", b + 1, "Cup", b
        ampo += -t, "Cdagdn", b, "Cdn", b + 1
        ampo += -t, "Cdagdn", b + 1, "Cdn", b
    end
    if pbc && N % 4 == 0
        t *= -1
    end
    if pbc
        ampo += -t, "Cdagup", N, "Cup", 1
        ampo += -t, "Cdagup", 1, "Cup", N
        ampo += -t, "Cdagdn", N, "Cdn", 1
        ampo += -t, "Cdagdn", 1, "Cdn", N
    end
    for i in 1:N
        ampo += U, "Nupdn", i
    end
    H = MPO(ampo, sites)
    
    maxdim = [10, 20, 80, 200, 300, 400, 400, 800, 800, 1600]
    cutoff = [1E-5, 1E-6, 1E-7, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8]
    noise = [1E-5, 1E-5, 1E-8, 1E-9, 1E-10, 1E-10, 1E-10, 1E-10, 1E-10, 1E-10]
    
    state = ["Emp" for n in 1:N]
    p = Npart
    for i in N:-1:1
        if p > i
            println("Doubly occupying site $i")
            state[i] = "UpDn"
            p -= 2
        elseif p > 0
            println("Singly occupying site $i")
            state[i] = (isodd(i) ? "Up" : "Dn")
            p -= 1
        end
    end
    # Initialize wavefunction to be bond 
    # dimension 10 random MPS with number
    # of particles the same as `state`
    psi0 = randomMPS(sites, state, 10)
    
    # Check total number of particles:
    @show flux(psi0)
    
    # Start DMRG calculation:
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise)
    
    upd = fill(0.0, N)
    dnd = fill(0.0, N)
    for j in 1:N
        orthogonalize!(psi, j)
        psidag_j = dag(prime(psi[j], "Site"))
        upd[j] = scalar(psidag_j * op(sites, "Nup", j) * psi[j])
        dnd[j] = scalar(psidag_j * op(sites, "Ndn", j) * psi[j])
    end
    
    println("Up Density:")
    for j in 1:N
        println("$j $(upd[j])")
    end
    println()
    
    println("Dn Density:")
    for j in 1:N
        println("$j $(dnd[j])")
    end
    println()
    
    println("Total Density:")
    for j in 1:N
        println("$j $(upd[j]+dnd[j])")
    end
    println()
    
    println("\nGround State Energy = $energy")
end

main()