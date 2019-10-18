#for i in 1 3 4;
#do
#  name="pure-lstm-$i-highway-tts"
#  bash tts.sh $name pure-lstm-ds-8 $i
#  bash gen_spec.sh $name LibriSpeech
#done

#for i in 0 1 2 3 4;
#do
#  name="pure-lstm-$i-highway-tts"
#  bash gen_spec.sh $name LibriSpeech
#  for j in 0 10 20;
#  do
#    bash gen_spec.sh $name aug_${j}dB
#  done
#done
#
#for i in 9;
#do
#  name="musan-layer-$i-highway-tts"
#  bash tts.sh $name libri-asr-musan $i
#done
#for i in 0 4 9 10 12 14;
#do
#  name="musan-layer-$i-highway-tts"
#  #bash tts.sh $name libri-asr-musan $i
#  bash gen_spec.sh $name LibriSpeech
#  for j in 0 10 20;
#  do
#    bash gen_spec.sh $name aug_${j}dB
#  done
#done
#
#for i in 0 4 9 10 12 14;
#do
#  name="layer-$i-highway-tts"
#  #bash tts.sh $name libri-asr-musan $i
#  bash tts.sh $name cont-vgg-delta $i
#  bash gen_spec.sh $name LibriSpeech
#  for j in 0 10 20;
#  do
#    bash gen_spec.sh $name aug_${j}dB
#  done
#done
#bash tts.sh musan-layer-0-highway-tts checkpoint/libri-asr-musan/best_ctc.pth 0
#bash tts.sh musan-layer-1-highway-tts checkpoint/libri-asr-musan/best_ctc.pth 1
#bash tts.sh musan-layer-2-highway-tts checkpoint/libri-asr-musan/best_ctc.pth 2
#bash tts.sh musan-layer-4-highway-tts checkpoint/libri-asr-musan/best_ctc.pth 4
#bash tts.sh musan-layer-6-highway-tts checkpoint/libri-asr-musan/best_ctc.pth 6
#bash tts.sh musan-layer-8-highway-tts checkpoint/libri-asr-musan/best_ctc.pth 8
#bash tts.sh musan-layer-9-highway-tts checkpoint/libri-asr-musan/best_ctc.pth 9
#bash tts.sh musan-layer-0-highway-tts-on-clean checkpoint/cont-vgg-delta/best_ctc.pth 0
#bash tts.sh musan-layer-4-highway-tts-on-clean checkpoint/cont-vgg-delta/best_ctc.pth 4
#bash tts.sh musan-layer-9-highway-tts-on-clean checkpoint/cont-vgg-delta/best_ctc.pth 9
#bash tts.sh musan-layer-12-highway-tts-on-clean checkpoint/cont-vgg-delta/best_ctc.pth 12
#bash tts.sh musan-layer-14-highway-tts-on-clean checkpoint/cont-vgg-delta/best_ctc.pth 14
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
