'''Compare the classification speed of MobileNet and VGG.

This uses randomly initialized weights, so we won't get any real classification
results, but it should give us a general idea of how fast each one is.

On a GeForce 960m GPU:
MobileNet, alpha=1: 89fps
MobileNet, alpha=0.5: 100fps
VGG16: 21fps
'''
import time
import keras
import numpy as np
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras_mobilenet import MobileNet

def run_test(model_name, alpha=None):
    # Get the model and compile it.
    img_input = keras.layers.Input(shape=(128, 128, 3))
    
    if model_name == 'mobilenet':
        model = MobileNet(input_tensor=img_input, alpha=alpha, classes=1000)
    elif model_name == 'vgg16':
        model = VGG16(input_tensor=img_input, weights=None)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Classify a bunch of "images".
    print("Classifying.")
    start = time.time()
    for _ in range(100):
        image = np.random.random((128, 128, 3))
        model.predict(np.expand_dims(image, axis=0))
    end = time.time()
    print("%s: Classified 100 images in %.2f seconds (%.2ffps)" %
          (model_name, (end - start), (100 / (end - start))))

def main():
    run_test('vgg16')
    run_test('mobilenet', 1)
    run_test('mobilenet', 0.5)

if __name__ == '__main__':
    main()
