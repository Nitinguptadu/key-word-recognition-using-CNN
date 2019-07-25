# ConvNets for Speech Commands Recognition

## Installation
* Install [PyTorch](https://github.com/pytorch/pytorch#installation) 
* Install [LibRosa](https://github.com/librosa/librosa)

### Custom Dataset
You can also use the data loader and training scripts for your own custom dataset.
In order to do so the dataset should be arrange in the following way:
```
root/up/kazabobo.wav
root/up/asdkojv.wav
root/up/lasdsa.wav
root/right/blabla.wav
root/right/nsdf3.wav
root/right/asd932.wav
```

### Training
Use `python run.py --help` for more parameters and options.

```
python run.py --train_path <train_data_path> --valid_path <valid_data_path> --test_path <test_data_path>
```

### Inference


```
python classify.py --wav_path <wav_path>
```
