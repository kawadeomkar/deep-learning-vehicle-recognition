import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from resnet_152 import resnet152_model

image_width, image_height = 224, 224
classes, train_samples, valid_samples = 196, 6549, 1595
batch_size = 16
epochs = 50000

data_train_dir = 'data/train'
data_valid_dir = 'data/valid'

if __name__ == '__main__':
    
    model = resnet152_model(image_height, image_width, 3, classes)

    trainData, validData  = ImageDataGenerator(rotation_range=20.,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.2,
                                        horizontal_flip=True),
                            ImageDataGenerator()

    trainGen = trainData.flow_from_directory(data_train_dir, (image_width, image_height), 
        batch_size=batch_size, class_mode='categorical')
    validGen = valid_data_gen.flow_from_directory(data_valid_dir, (img_width, img_height), 
        batch_size=batch_size, class_mode='categorical')

   
    tb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    lr = ReduceLROnPlateau('val_acc', patience=12, factor=0.1, verbose=1)
    es = EarlyStopping('val_acc', patience=50)
    modelName = 'models/model' +'-{epoch:02d}-{val_loss:.2f}.hdf5'
    modelCheckpoint = ModelCheckpoint(modelName, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks = [tb, modelCheckpoint, es, lr]

    
    # fine tune
    model.fit_generator(
        trainGen,
        steps_per_epoch=train_samples / batch_size,
        validation_data=validGen,
        validation_steps=valid_samples / batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose)
