mkdir -p log/$1
cp log/$2/libri_id_example.yaml log/$1/
cp log/$2/libri_id_example.yaml checkpoint/$2/
python3 main.py --config log/$2/libri_id_example.yaml --id --test --name $1 --logdir log --outdir save --njobs 10 --seed 0 --load checkpoint/$2
