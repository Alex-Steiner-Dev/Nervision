import tensorflow as tf

def pc_distmat(x):
    batchsize = tf.shape(x)[0]
    m = tf.shape(x)[1]
    xx = tf.tile(tf.expand_dims(tf.reduce_sum(tf.square(x), axis=2), axis=2), [1, m, m])
    yy = tf.transpose(xx, perm=[0, 2, 1])
    inner = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
    distance = xx + yy - 2*inner
    return distance

def stitchloss(point, indices):
    batchsize = tf.shape(point)[0]
    dismat = pc_distmat(point)
    keypointvariance = 0
    for i in range(batchsize):
        tmpindexes = tf.unique(tf.reshape(indices[i], [512])).y
        tmp = tf.gather(dismat[i], tmpindexes, axis=0)
        tmp = -tmp
        min20distances,_ = tf.math.top_k(tmp, k=40, sorted=False)
        var11 = tf.math.reduce_variance(-min20distances, axis=1)
        keypointvariance += tf.math.reduce_mean(var11)
    return keypointvariance