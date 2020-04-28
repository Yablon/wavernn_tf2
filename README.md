# WIP: Wavernn on tensorflow 2

I will try to complete state of art Vocoders here.

## Run

python version: 3.7

install packages:

pip install -r requirements.txt

## Data

I will use English and Chinese data both.
LbriSpeech and BZNSYP would be considered.

## TODOs

- [x] prepare data pipeline
- [ ] train with fatchord's wavernn
- [ ] add lpcnet pipeline
- [ ] train with lpcnet
- [ ] add multiple upsample networks
- [ ] add CI for tests

References:

- [keithito Tacotron](https://github.com/keithito/tacotron)

- [fatchord WaveRNN](https://github.com/fatchord/WaveRNN)