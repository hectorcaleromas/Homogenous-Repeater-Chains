tol='1e-7'

## For a single case ##
n=5
p=0.8
p_s=1
cutoff=5
echo "policyiter n=$n, p=$p, p_s=$p_s, cutoff=$cutoff, tol=$tol"
python policyiter_solve.py --n $n --p $p --p_s $p_s \
                          --cutoff $cutoff --tol $tol

## For several cases (in parallel, using screens) ##
#for n in 3 4
#do
#    for cutoff in 2 5 10
#    do
#        for p_s in $(seq 0.5 0.5 1)
#        do
#            for p in $(seq 0.1 0.1 1)
#            do
#                echo "policyiter n=$n, p=$p, p_s=$p_s, cutoff=$cutoff, tol=$tol"
#                screen -S policyiter_n${n}_p${p}_ps${p_s}_tcut${cutoff} -d -m \
#                    python policyiter_solve.py --n $n --p $p --p_s $p_s \
#                                             --cutoff $cutoff --tol $tol
#            done
#        done
#    done
#done