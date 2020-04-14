# Power_Control_CMAC

## 1. Train Usage

If you want to train a new model, you can follow the command following.

```python
# Data Generating
python AWGN_datasets_generator.py
# Train
python train.py --save-path ./SNR_0_quick --power-dB 0 --gpu '0'
```

## 2. Result Show

Result analysis is showed in **./draw_fig.ipynb**.

## 3. Reference

[Deep Learning for Distributed Optimization: Applications to Wireless Resource Management](https://arxiv.org/abs/1905.13378)