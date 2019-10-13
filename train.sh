mkdir -p log/$1
cp config/libri_asr_example.yaml log/$1
export CUDA_VISIBLE_DEVICES=$2
python3 main.py --config config/libri_asr_example.yaml --name $1 --logdir log --ckpdir checkpoint --outdir save --njobs 10 --seed 0 $3
