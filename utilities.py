import tensorflow as tf
import numpy as np

def get_dataset_partitions_tf(
    ds, 
    ds_size, 
    train_split=0.8, 
    val_split=0.1, 
    test_split=0.1, 
    shuffle=True, 
    shuffle_size=10000
):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds