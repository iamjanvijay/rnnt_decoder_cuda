# RNN-Transducer Prefix Beam Search

This repository provides an optimised implementation of prefix beam search for RNN-Tranducer loss function (as described in "[Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/pdf/1211.3711.pdf)" paper). This implementation takes ~100 milliseconds for a speech segment of ~5 seconds and beam size of 10 (beam size of 10 is adequate for production level error rates).

## Sample Run

To execute a sample run of prefix beam search on your machine, execute the following commands:

1. Clone this repository.

```
git clone https://github.com/iamjanvijay/rnnt_decoder_cuda.git;
```

2. Clean the output folder.

```
rm rnnt_decoder_cuda/data/outputs/*;
```

3. Make the deocder object file.

```
cd rnnt_decoder_cuda/decoder;
make clean;
make;
```
4. Execute the decoder - decoded beams will be saved to data/output folder.

```
CUDA_VISIBLE_DEVICES=0 ./decoder ../data/inputs/metadata.txt 0 9 10 5001;
CUDA_VISIBLE_DEVICES=$GPU_ID$ ./decoder ../data/inputs/metadata.txt $index_of_first_file_to_read_from_metadata$ $index_of_last_file_to read_from_metadata$ $beam_size$ $vocabulary_size_excluding_blank$;
```

## Contributing

Contributions are welcomed and greatly appreciated.
