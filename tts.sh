CONFIG="libri_tts_example.yaml"
ASR_CONFIG="libri_asr_example.yaml"
mkdir -p log/$1
cp config/$CONFIG log/$1
sed -i "s/layer_num\:.*$/layer_num\:\ $3/g" log/$1/$CONFIG
sed -i "s/src:.*/src:\n\ \ config: \'log\/$2\/$ASR_CONFIG\'\n\ \ ckpt:\ \'checkpoint\/$2\/best_ctc.pth\'/g" "log/$1/$CONFIG"
python3 main.py --config log/$1/$CONFIG --tts --name $1 --logdir log --ckpdir checkpoint --outdir save --njobs 10 --seed 0 # --load checkpoint/$2/best_ctc.pth
