CONFIG="libri_tts_example.yaml"
mkdir -p log/$1
cp config/$CONFIG log/$1
sed -i s/layer_num\:.*$/layer_num\:\ $3/g log/$1/$CONFIG
export CUDA_VISIBLE_DEVICES=$4
python3 main.py --config log/$1/$CONFIG --tts --name $1 --logdir log --ckpdir checkpoint --outdir save --njobs 5 --seed 0 --load $2
