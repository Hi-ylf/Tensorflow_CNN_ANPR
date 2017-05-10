#!/usr/bin/python
#coding:utf-8

##################################
#Author:yanglinfeng
#Date:2017-5-6
#desc:predict from ANPR
##################################



import sys
import numpy
import glob
import cv2

import common
import model
from util import unzip, read_data
import tensorflow as tf


def getImage(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        yield im    
def vec_to_plate(v):
    return "".join(common.CHARS[i] for i in v)

def predict(weights):
    '''
    step one: get pridict data
    '''
    data = getImage("predict/*.png")
    init_img = list(data)[:50]
    
    predict_xs = numpy.array(init_img)

    '''
    get_model:
    '''
    x, y, params = model.get_training_model()
    result = tf.argmax(tf.reshape(y[:, 1:], [-1, 7, len(common.CHARS)]), 2)    #???2

    if weights is not None:
        assert len(params) == len(weights)
        assign_ops = [w.assign(v) for w, v in zip(params, weights)]

    init = tf.initialize_all_variables()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        if weights is not None:
            sess.run(assign_ops)
	    r = sess.run([result,
                      tf.greater(y[:, 0], 0)],
                      feed_dict={x: predict_xs})

	    r_short = (r[0][:190], r[1][:190])
	    for b, pb in zip(*r_short):
        	print "{} {}".format(vec_to_plate(b), float(pb))

 
if __name__ == "__main__":

    if len(sys.argv) > 1:
        f = numpy.load(sys.argv[1])
        weights = [f[n] for n in sorted(f.files,
                                        key=lambda s: int(s[4:]))]

	ret = predict(weights)

    else:
	print('loss paragam, please train again')
	sys.exit()
