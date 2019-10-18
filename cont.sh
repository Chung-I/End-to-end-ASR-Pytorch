PYTORCH_JIT=0 python3 main.py --config log/$1/libri_asr_example.yaml --name $1 --logdir log --ckpdir checkpoint --outdir save --njobs 6 --seed 0 --load checkpoint/$1/cur_ctc.pth
