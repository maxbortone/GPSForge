using ITensors
using PyCall



# DMRG + system parameters
M_bond = 128
pbc = true
n_sweeps = 10
max_dim = [M_bond] # [1, 2, 4, 8, 16, 32, 64, 64, 64, 64]
cut_off = [1e-10] # [1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12]
L = 4

# Define call to netket with PyCall
py"""
import numpy as np
from GPSKet.operator.hamiltonian import get_J1_J2_Hamiltonian
def get_edges_and_states(L, pbc):
    ha = get_J1_J2_Hamiltonian(L, L, on_the_fly_en=True, pbc=pbc)

    hi = ha.hilbert
    g = ha.graph


    all_states = hi.all_states()
    edges = g.edges()
    return edges, all_states
"""

edges, all_states = py"get_edges_and_states"(L, pbc)


# Set up DMRG calculation
sites = siteinds("S=1/2", L^2; conserve_qns=true)


os = OpSum()

for edge in edges
    os .+= 0.5, "S+", edge[1]+1, "S-", edge[2]+1
    os .+= 0.5, "S-", edge[1]+1, "S+", edge[2]+1
    os .+= "Sz", edge[1]+1, "Sz", edge[2]+1
end


for edge in edges
    println(edge)
end

H = MPO(os, sites)

init = [isodd(n) ? "Up" : "Dn" for n in 1:(L^2)]

psi0 = randomMPS(sites, init, M_bond)


energy, psi = dmrg(H, psi0; nsweeps=n_sweeps, maxdim=max_dim, cutoff=cut_off)



# Convert netket to Itensor representation
all_states_itensor = Int.((all_states .+ 1)/2 .+ 1)

# Now evaluate the exact amplitudes
dense_psi = dense(psi)
amplitudes = Vector{Float64}()

for config in eachrow(all_states_itensor)
    amp = ITensor(1.)
    for j = 1:(L^2)
        amp *= (dense_psi[j]*dense(state(sites[j], config[j])))
    end
    push!(amplitudes, (scalar(amp)))
end

# Save all MPS amplitudes
py"np.save"("heisenberg2d_L4_mps_M128_states.npy", all_states)
py"np.save"("heisenberg2d_L4_mps_M128_amplitudes.npy", amplitudes)