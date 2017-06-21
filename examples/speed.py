'''Compare the classification speed of MobileNet and VGG.

This uses randomly initialized weights, so we won't get any real classification
results, but it should give us a general idea of how fast each one is.

On a GeForce 960m GPU:
MobileNet, alpha=1: 53fps
MobileNet, alpha=3: 32fps
VGG16: 27fps
'''
import keras
import numpy as np
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16
from keras_mobilenet import MobileNet

def run_test(model_name):
    # Get the model and compile it.
    img_input = keras.layers.Input(shape=(112, 112, 3))
    
    if model_name == 'mobilenet':
        model = MobileNet(input_tensor=img_input, alpha=3, classes=1000)
    elif model_name == 'vgg16':
        model = VGG16(input_tensor=img_input, weights=None)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())

    # Classify a bunch of "images".
    print("Classifying.")
    import time
    start = time.time()
    for _ in range(100):
        image = np.random.random((112, 112, 3))
        model.predict(np.expand_dims(image, axis=0))
    end = time.time()
    print("%s: Classified 100 images in %.2f seconds (%.2ffps)" %
          (model_name, (end - start), (100 / (end - start))))

def main():
    run_test('vgg16')

if __name__ == '__main__':
    main()
