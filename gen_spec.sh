CONFIG="libri_tts_test.yaml"
SRC_CONFIG="libri_tts_example.yaml"
mkdir -p save/$1
cp config/$CONFIG save/$1
sed -i "s/src:.*/src:\n\ \ config: \'log\/$1\/$SRC_CONFIG\'\n\ \ ckpt:\ \'checkpoint\/$1\/tts.pth\'/g" "save/$1/$CONFIG"
CUDA_VISIBLE_DEVICES=0 python3 main.py --config save/$1/$CONFIG --tts --name $1 --logdir log --ckpdir checkpoint --outdir save --njobs 5 --seed 0 --test --no-pin
