### CSV to HDF5
- Download the data files (train.csv, test.csv) from Kaggle and edit file paths 
  in `csv_to_hdf5.py`

Run:
```
$ python csv_to_hdf5.py
```

### Training
- Set the data_dir in all `*_data.pbtxt` so that it points to the directory where
  the data was downloaded. 
- Set the checkpoint directory in net.pbtxt. This is where the model, error
  stats, logs etc will be written. Make sure this diretory has been created.

Run:
```
$ train_convnet -b <board-id> -m netconv.pbtxt -t train_data.pbtxt -v val_data.pbtxt
```

Toronto users-
Make sure the board is locked before running this.

### Predictions from the model
- Set output file path in `feature_config.pbtxt`
- <model-file> name would be shown in the training output ( example: checkpoint_dir/mnist_net_20140627130044.pbtxt )

Run:
```
$ extract_representation -b <board-id> -m <model-file> -f feature_config.pbtxt
```

### HDF5 to CSV for submission
Run:
```
$ python submit_hdf5_to_csv.py
```

