onmt_translate -data_type img -model ../checkpoints/recognition/Recognition_All.pt -src_dir ~/Documents/DataSets/TableBank/Recognition/images -src ~/Documents/DataSets/TableBank/Recognition/annotations/src-all_test.txt -output ../output/pred.txt -max_length 150 -beam_size 5 -gpu 0 -verbose -batch_size 8