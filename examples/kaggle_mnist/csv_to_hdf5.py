import numpy as np
import h5py as h5

def csv_to_hdf5(path_to_csv, path_to_hdf5, shuffle=False, test=False):
	alldata = np.loadtxt(open(path_to_csv,"rb"),delimiter=",",skiprows=1)
	if shuffle:
		np.random.shuffle(alldata)
	if test:
		data_ = alldata/255
		f = h5.File(path_to_hdf5, 'w')
		f.create_dataset('test', data=data_)
		f.close()
	else:
		labels = alldata[:,0]
		labels = labels.astype(int)
		data_ = alldata[:,1:]/255
		f = h5.File(path_to_hdf5, 'w')
		f.create_dataset('train', data=data_)
		f.create_dataset('train_labels', data=labels)
		f.close()

train_path = '/ais/gobi3/u/shikhar/mnist/train.csv'
train_save_path = '/ais/gobi3/u/shikhar/mnist/mnist_kaggle_train.h5'
csv_to_hdf5(train_path,train_save_path,shuffle=True)

test_path = '/ais/gobi3/u/shikhar/mnist/test.csv'
test_save_path = '/ais/gobi3/u/shikhar/mnist/mnist_kaggle_test.h5'
csv_to_hdf5(test_path,test_save_path,test=True)
