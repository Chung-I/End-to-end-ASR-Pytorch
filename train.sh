mkdir -p log/$1
cp config/libri_asr_example.yaml log/$1
PYTORCH_JIT=0 python3 main.py --config log/$1/libri_asr_example.yaml --name $1 --logdir log --ckpdir checkpoint --outdir save --njobs 10 --seed 0 $2
