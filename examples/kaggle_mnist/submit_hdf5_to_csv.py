import numpy as np
import h5py as h5
import csv

f = h5.File('/ais/gobi3/u/shikhar/mnist/output.h5')
a = np.array(f['output'])
b = np.argmax(a,axis=1)

s = [['ImageId','Label']]
for i in xrange(len(b)):
	s.append([str(i+1),str(b[i])])
#np.savetxt('/home/shikhar/Downloads/mnist_kaggle/submit.csv', s, delimiter=",")

with open('/ais/gobi3/u/shikhar/mnist/submit.csv', 'w') as fp:
	f = csv.writer(fp, delimiter=',')
	f.writerows(s)

