from models import Generator
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def saveFromVoxels(voxels, path):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig(path)

def test():
    generator = Generator().build_generator()
    generator.load_weights('Models\generator_epoch_501.h5')
    sample_noise = np.random.normal(0, 0.33, size=[1, 1, 1, 1, 200]).astype(np.float32)
    generated_volumes = generator.predict(sample_noise, verbose=1)
    generated_volumes = generated_volumes.reshape(64, 64,64)
    saveFromVoxels(generated_volumes, "test.png")

test()