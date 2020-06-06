from multiprocessing import Pool
import numpy as np
import tensorflow as tf

from multiprocessing import Pool
from functools import partial

import knn.lib.python.nearest_neighbors as nearest_neighbors

# from scipy.spatial import cKDTree, KDTree
from sklearn.neighbors import KDTree
from time import time

from numba import jit

class KDTreeLayer(tf.keras.layers.Layer):
    '''
    Input:\n
        pointCount\n
        points: (b, n, 3)\n
        newPoints: (b, m, 3)\n
    Output:\n
        idx: (b, m, pointCount)\n
    '''
    def __init__(self, pointCount, **kwargs):
        super(KDTreeLayer, self).__init__(**kwargs)
        self.pointCount = pointCount        
    
    # @tf.function(experimental_compile=True)
    @tf.function()
    def call(self, xyz, new_xyz):
        b = tf.shape(new_xyz)[0]
        m = new_xyz.get_shape()[1]

        out = tf.zeros((b, m, self.pointCount, 1), tf.int64)
        # out = tf.zeros((b, m, self.pointCount, 2), tf.int32)

        # idx = tf.py_function(knn_kdtree, [self.pointCount, xyz, new_xyz, False], tf.int32)
        idx = tf.py_function(cython_knn_kdtree, [self.pointCount, xyz, new_xyz], tf.int64)
        idx = tf.expand_dims(tf.cast(tf.convert_to_tensor(idx), tf.int64), 3)

        for i in tf.range(b):      
            temp = tf.gather_nd(idx, [[i]])
            out = tf.tensor_scatter_nd_update(out, [[i]], temp)
                
        out = tf.reshape(out, (-1, m, self.pointCount, 1))
        return out
    
    def get_config(self):
        config = super(KDTreeLayer, self).get_config()
        config.update({'pointCount': self.pointCount})
        return config

class KDTreeSampleLayer(tf.keras.layers.Layer):
    '''
    Input:\n
        pointCount\n
        nqueries\n
        points: (b, n, 3)\n        
    Output:\n
        idx: (b, nqueries, pointCount, 2)\n
        pts: (b, nqueries, pointCount)\n
    '''
    def __init__(self, pointCount, nqueries, **kwargs):
        super(KDTreeSampleLayer, self).__init__(**kwargs)
        self.pointCount = pointCount       
        self.nqueries = nqueries
    
    # @tf.function(experimental_compile=True)
    @tf.function()
    def call(self, xyz):
        b = tf.shape(xyz)[0]

        out_indices = tf.zeros((b, self.nqueries, self.pointCount, 1), tf.int64)
        # out_indices = tf.zeros((b, self.nqueries, self.pointCount, 2), tf.int32)
        out_pts = tf.zeros((b, self.nqueries, 3), tf.float32)

        # idx, pts = tf.py_function(knn_kdtree, [self.pointCount, xyz, self.nqueries, True], [tf.int32, tf.float32])
        idx, pts = tf.py_function(cython_knn_kdtree_sampler, [self.pointCount, xyz, self.nqueries], [tf.int64, tf.float32])
        idx = tf.expand_dims(tf.cast(tf.convert_to_tensor(idx), tf.int64), 3)
        pts = tf.convert_to_tensor(pts)

        for i in tf.range(b):      
            temp = tf.gather_nd(idx, [[i]])
            out_indices = tf.tensor_scatter_nd_update(out_indices, [[i]], temp)
            out_pts = tf.tensor_scatter_nd_update(out_pts, [[i]], tf.gather_nd(pts, [[i]]))

        out_indices = tf.reshape(out_indices, (-1, self.nqueries, self.pointCount, 1))        
        out_pts = tf.reshape(out_pts, (-1, self.nqueries, 3))

        return out_indices, out_pts
    
    def get_config(self):
        config = super(KDTreeSampleLayer, self).get_config()
        config.update({'pointCount': self.pointCount, 'nqueries': self.nqueries})
        return config

def knn_kdtree(nsample, xyz, new_xyz, resample = False):
    # if isinstance(xyz, tf.Tensor):
    xyz = xyz.numpy()
        
    if resample:
        rplc = False
        if(xyz.shape[0]<new_xyz):
            rplc = True

        new_xyz = [xyz[i][np.random.choice(xyz.shape[1], new_xyz, replace = rplc)] for i in range(xyz.shape[0])]
        new_xyz = np.asarray(new_xyz)
    else:
        # if isinstance(new_xyz, tf.Tensor):
        new_xyz = new_xyz.numpy()

    batch_size = xyz.shape[0]
    n_points = new_xyz.shape[1]

    # indices = np.zeros((batch_size, n_points, nsample, 2), dtype=np.int32)
    indices = np.zeros((batch_size, n_points, nsample, 1), dtype=np.int32)

    for batch_idx in range(batch_size):
        X = xyz[batch_idx, ...]
        q_X = new_xyz[batch_idx, ...]
        kdt = KDTree(X, leaf_size=10) #CoinvPoint suggests 10
        _, batch_indices = kdt.query(q_X, k = nsample)

        ##### fill batch indices like in nearest neighbors layer
        # indicesForBatch = np.full((batch_indices.shape[0], batch_indices.shape[1]), batch_idx)
        # batch_indices = np.concatenate((np.expand_dims(indicesForBatch, axis=2), np.expand_dims(batch_indices, axis=2)), axis=2)
        batch_indices = np.expand_dims(batch_indices, axis=2)

        indices[batch_idx] = batch_indices
    
    if resample:
        return indices, new_xyz
    else:
        return indices

def cython_knn_kdtree(nsample, xyz, new_xyz):
    xyz = xyz.numpy()
    indices = nearest_neighbors.knn_batch(xyz, new_xyz, nsample, omp=True)
    return indices

def cython_knn_kdtree_sampler(K, xyz, npts):
    xyz = xyz.numpy()
    indices, queries = nearest_neighbors.knn_batch_distance_pick(xyz, npts, K, omp=True)
    return indices, queries

def multiproces_kdtree(nsample, xyz, new_xyz, resample = False):
    # if isinstance(xyz, tf.Tensor):
    # xyz = xyz.numpy()
            
    if resample: ######## move this to threads
        rplc = False
        if(xyz.shape[0]<new_xyz):
            rplc = True

        new_xyz = [xyz[i][np.random.choice(xyz.shape[1], new_xyz, replace = rplc)] for i in range(xyz.shape[0])]
        new_xyz = np.asarray(new_xyz)
    # else:
    #     # if isinstance(new_xyz, tf.Tensor):
    #     new_xyz = new_xyz.numpy()

    batch_size = xyz.shape[0]
    n_points = new_xyz.shape[1]

    indices = np.zeros((batch_size, n_points, nsample, 2), dtype=np.int32)

    data = [(xyz[batch_idx, ...], new_xyz[batch_idx, ...]) for batch_idx in range(batch_size)]
    pool = Pool(processes = 8)
    GenerateFunc=partial(ProcessBatch, nsample=nsample)
    result = pool.starmap(GenerateFunc, data)

    for batch_idx in range(batch_size):
        batch_indices = result[batch_idx]

        ##### fill batch indices like in nearest neighbors layer
        indicesForBatch = np.full((batch_indices.shape[0], batch_indices.shape[1]), batch_idx)
        batch_indices = np.concatenate((np.expand_dims(indicesForBatch, axis=2), np.expand_dims(batch_indices, axis=2)), axis=2)

        indices[batch_idx] = batch_indices
    
    if resample:
        return indices, new_xyz
    else:
        return indices

def ProcessBatch(X, q_X, nsample):
    kdt = KDTree(X, leaf_size=30) #CoinvPoint suggests 10
    _, batch_indices = kdt.query(q_X, k = nsample)

    return batch_indices

if __name__ == "__main__":
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D
    from tensorflow.python import debug as tf_debug
    from tensorflow.python.eager import profiler as profile
    from NearestNeighbors import NearestNeighborsLayer, SampleNearestNeighborsLayer

    profile.start_profiler_server(6009)
    tf.compat.v1.keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.compat.v1.Session(), "DESKTOP-TKAPBCU:6007"))
    log_dir="logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3)

    batchSize = 1000
    pointCount = 5
    aCount = 2000
    radius = 1
    
    a = [[[9,9,9], [8,8,8], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 
        [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
        [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2],
        [3, 3, 3], [3, 3, 3]]]
    a = np.array(a, dtype=np.float32)

    b = [[[9,9,9], [8,8,8], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    b = np.array(b, dtype=np.float32)

    a = np.concatenate([np.zeros((1, aCount, 3), dtype=np.float32), a], axis=1)

    a = np.tile(a, (batchSize, 1, 1))
    b = np.tile(b, (batchSize, 1, 1))
    # a = np.tile(a, (3, 1, 1))
    # b = np.tile(b, (3, 1, 1))

    t=time()
    # RandomSample(a, 1000)
    # tf.print("Random sample done in {:.5f}".format((time() - t)/60))
    idx = knn_kdtree(5, a, b)
    print("knn_kdtree done:",time() - t)
    t=time()
    idx = multiproces_kdtree(5, a, b)
    print("multiproces_kdtree done:",time() - t)
    # print(idx)
    
    input()

    ptsA = Input(shape=(a.shape[1], 3), dtype=tf.float32)
    ptsB = Input(shape=(b.shape[1], 3), dtype=tf.int32)

    ps = tf.expand_dims(ptsA, axis = 1)    
    ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    # ps = Conv2D(3, [1, 1], [1, 1], 'valid', activation='relu')(ps)
    ps = tf.squeeze(ps, axis=1)

    # out = KDTreeLayer(5)(ps, ptsB)
    # out, outpts = KDTreeSampleLayer(5, 1000)(ps)
    out, outpts = SampleNearestNeighborsLayer(5, 1000)(ps)
    out = tf.cast(out, tf.float32)

    # out = tf.expand_dims(out, axis = 1)
    out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)
    # out = Conv2D(5, [1, 1], [1, 1], 'valid', activation='relu')(out)    
    # out = tf.squeeze(out, axis=1)

    model = Model([ptsA, ptsB], [out], name ="model")
    model.compile(tf.keras.optimizers.Adam(), loss='mse', metrics=['accuracy'])

    print(model.summary())

    y = np.ones((a.shape[0], 1000, 5, 5), dtype=np.float32)
    # y = np.ones((batchSize, 1000, 5), dtype=np.float32)
    model.fit([a, b], [y], batch_size = 100, epochs = 3, callbacks=[tensorboard_callback])