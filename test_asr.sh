CONFIG="libri_asr_test.yaml"
SRC_CONFIG="libri_asr_example.yaml"
export CUDA_VISIBLE_DEVICES=0
mkdir -p save/$1
cp config/$CONFIG save/$1
sed -i "s/src:.*/src:\n\ \ config: \'log\/$1\/$SRC_CONFIG\'\n\ \ ckpt:\ \'checkpoint\/$1\/best_ctc.pth\'/g" "save/$1/$CONFIG"
PYTORCH_JIT=0 python3 main.py --config save/$1/$CONFIG --name $1 --logdir save --ckpdir checkpoint --outdir save --njobs 10 --seed 0 --test
