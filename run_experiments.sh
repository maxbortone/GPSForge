#!/bin/sh

# General parameters
samples=4000
discard=100
iterations=1000
lr=0.01
ds=0.01

# MSR on
for N in 1 2 3 4 5 6 7 8
do
    python heisenberg1d.py -L 8 -N $N --ansatz qgps --dtype real --samples $samples --discard $discard --iterations $iterations --lr $lr --ds $ds --msr --save results/msr_on
done
for N in 1 2 3 4 5 6 7 8
do
    python heisenberg1d.py -L 8 -N $N --ansatz ar-qgps --dtype real --samples $samples --iterations $iterations --lr $lr --ds $ds --msr --save results/msr_on
done
for N in 1 2 3 4 5 6 7 8
do
    python heisenberg1d.py -L 8 -N $N --ansatz qgps --dtype complex --samples $samples --discard $discard --iterations $iterations --lr $lr --ds $ds --msr --save results/msr_on
done
for N in 1 2 3 4 5 6 7 8
do
    python heisenberg1d.py -L 8 -N $N --ansatz ar-qgps --dtype complex --samples $samples --iterations $iterations --lr $lr --ds $ds --msr --save results/msr_on
done

# MSR off
for N in 1 2 3 4 5 6 7 8
do
    python heisenberg1d.py -L 8 -N $N --ansatz qgps --dtype real --samples $samples --discard $discard --iterations $iterations --lr $lr --ds $ds --no-msr --save results/msr_off
done
for N in 1 2 3 4 5 6 7 8
do
    python heisenberg1d.py -L 8 -N $N --ansatz ar-qgps --dtype real --samples $samples --iterations $iterations --lr $lr --ds $ds --no-msr --save results/msr_off
done
for N in 1 2 3 4 5 6 7 8
do
    python heisenberg1d.py -L 8 -N $N --ansatz qgps --dtype complex --samples $samples --discard $discard --iterations $iterations --lr $lr --ds $ds --no-msr --save results/msr_off
done
for N in 1 2 3 4 5 6 7 8
do
    python heisenberg1d.py -L 8 -N $N --ansatz ar-qgps --dtype complex --samples $samples --iterations $iterations --lr $lr --ds $ds --no-msr --save results/msr_off
done
