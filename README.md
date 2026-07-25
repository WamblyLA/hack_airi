# SPARC-ERA5

## Архитектура

Входной погодный кадр разделяется на четыре физические группы — температура, ветер, давление и влага с осадками, после чего небольшие специализированные stems извлекают признаки каждой группы и объединяют их в общем многомасштабном encoder

Encoder создаёт два latent-потока — **base latent** хранит крупномасштабную атмосферную структуру, а **detail latent** добавляет фронты, осадки и мелкие пространственные детали

Квантованные latent-символы записываются в настоящий ANS bitstream, поэтому compression ratio считается по полному размеру `.bin` файла вместе с hyperprior и заголовком

Decoder восстанавливает 28 погодных каналов из base-потока или из base и detail вместе, дополнительно используя известные static-поля — land-sea mask, орографию и координатные признаки широты

## Артефакты

Артефакты получены путем обучения на сэмпле из 2 временных кадров в течение 300 шагов

## Кодирование

```bash
python -m src.encode --checkpoint checkpoints/checkpoint.pt --input examples/sample_input.npz --output bitstreams/check.bin --mode full
```

Режим `full` использует base и detail для лучшего качества, для максимального сжатия достаточно заменить `full` на `base`

## Декодирование

```bash
python -m src.decode --checkpoint checkpoints/checkpoint.pt --input bitstreams/check.bin --static examples/sample_input.npz --output examples/reconstruction.npy
```

## Повторное обучение

```bash
python -m src.train --cache /data/era5.zarr --stats /data/stats.npz --out .
```
