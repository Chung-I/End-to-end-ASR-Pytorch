for i in 0 4 9 10 12 14;
do
  name="musan-layer-$i-highway-tts"
  #bash tts.sh $name libri-asr-musan $i
  hrun -c 4 -m 20 -s -G bash gen_spec.sh $name LibriSpeech &
  for j in 0 10 20;
  do
    hrun -c 4 -m 20 -s -G bash gen_spec.sh $name aug_${j}dB &
  done
done
