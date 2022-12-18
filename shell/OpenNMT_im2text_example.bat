onmt_translate -data_type img \
               -model demo-model_acc_x_ppl_x_e13.pt \
               -src_dir data/im2text/images \
               -src data/im2text/src-test.txt \
               -output pred.txt \
               -max_length 150 \
               -beam_size 5 \
               -gpu 0 \
               -verbose