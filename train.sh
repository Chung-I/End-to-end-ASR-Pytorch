mkdir -p log/$2
cp $1 log/$2
export CUDA_VISIBLE_DEVICES=$3
python3 main.py --config $1 --name $2 --logdir log --ckpdir checkpoint --outdir save --njobs 10 --seed 0 $4
