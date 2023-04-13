using ArgParse
using ITensors


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--N"
            help = "Number of sites in the chain"
            arg_type = Int
            default = 32
        "--J"
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
# for the 1D Heisenberg model
#

function main()
    parsed_args = parse_commandline()
    N = parsed_args["N"]
    J = parsed_args["J"]
    pbc = parsed_args["pbc"]
    nsweeps = 10
    
    println("Running DMRG for N=$N, J=$J and pbc=$pbc")

    sites = siteinds("S=1/2", N; conserve_qns=true)
    
    ampo = OpSum()
    for j in 1:(N-1)
        ampo .+= J*0.5, "S+", j, "S-", j + 1
        ampo .+= J*0.5, "S-", j, "S+", j + 1
        ampo .+= J, "Sz", j, "Sz", j + 1
    end
    if pbc
        ampo .+= J*0.5, "S+", N, "S-", 1
        ampo .+= J*0.5, "S-", N, "S+", 1
        ampo .+= J, "Sz", N, "Sz", 1
    end
    H = MPO(ampo, sites)
    
    maxdim = [10, 20, 80, 200, 300, 400, 400, 800, 800, 1600]
    cutoff = [1E-5, 1E-6, 1E-7, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8]
    noise = [1E-5, 1E-5, 1E-8, 1E-9, 1E-10, 1E-10, 1E-10, 1E-10, 1E-10, 1E-10]
    
    # Initialize wavefunction to be bond 
    # dimension 10 random MPS with number
    # of particles the same as `state`
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi0 = randomMPS(sites, state, 10)
    
    # Check total number of particles:
    @show flux(psi0)
    
    # Start DMRG calculation:
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise)    
    println("\nGround State Energy = $energy")
end

main()