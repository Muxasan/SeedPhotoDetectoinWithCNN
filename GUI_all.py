from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from kerastuner.tuners import RandomSearch, Hyperband
from PySide2.QtWidgets import (QApplication, QWidget, QToolTip, QLineEdit, QPushButton, QDoubleSpinBox, QMessageBox,
                               QDesktopWidget, QPushButton, QSpinBox, QLabel, QGridLayout)
from PySide2.QtGui import QIcon, QFont
import sys
import time
import numpy as np
import imutils
import cv2
import logging
import os

class Example(QWidget):

    def __init__(self):
        super().__init__()
        
        self.initUI()

    def initUI(self):

        QToolTip.setFont(QFont('SansSerif', 12))
        
        Title = QLabel('Выберите параметры обучения нейросети')
        Title.setToolTip('Для корректной работы функций обучения и оптимизации \n'
                         'в каталоге с программой должны находится подкаталоги \n'
                         'train, val, test. В каждом из этих каталогов должно \n'
                         'быть столько подкаталогов с изображениями, сколько \n'
                         'необходимо распознавать классов')
        Title1 = QLabel('Выберите параметры исследования изображения')
        Title1.setToolTip('Для корректной работы функции классификации и локализации \n'
                          'изображения для исследования и модель нейросети должны \n'
                          'находиться в одном каталоге с программой, формат изображения \n'
                          '.jpg, формат модели нейросети .h5, указывать формат в полях не нужно')
        TitleEp = QLabel('Эпохи:')
        TitleEp.setToolTip('Эпоха — один «проход» данных через нейросеть. \n'
                           'Изменение этого параметра позволяет избежать \n'
                           'недообученности или переобученности нейросети')
        TitleBa = QLabel('Размер мини-выборки:')
        TitleBa.setToolTip('Количество изображений загружаемых в нейросеть '
                           'за раз')
        TitleTr = QLabel('Количество изображений для обучения:')
        TitleTr.setToolTip('Количество изображений во всех подкаталогах '
                           'каталога train')
        TitleVa = QLabel('Количество изображений для проверки:')
        TitleVa.setToolTip('Количество изображений во всех подкаталогах '
                           'каталога val')
        TitleTe = QLabel('Количество изображений для тестирования:')
        TitleTe.setToolTip('Количество изображений во всех подкаталогах '
                           'каталога test')
        TitleSc = QLabel('Масштабирование пирамиды изображений:')
        TitleSc.setToolTip('Коэффициент масштабирования изображения')
        TitleSs = QLabel('Шаг скользящего окна:')
        TitleSs.setToolTip('Расстояние в пикселях, на которое смещается \n'
                           'окно классификатора по изображению за одну итерацию')
        TitleNi = QLabel('Имя изображения для исследования:')
        TitleNi.setToolTip('Наименование изображения без расширения. \n'
                           'Изображение должно иметь формат jpg и \n'
                           'находится в одном каталоге с программой, иначе \n'
                           'необходимо указывать полный путь до изображения')
        TitleNm = QLabel('Имя модели нейросети:')
        TitleNm.setToolTip('Наименование модели нейросети без расширения. \n'
                           'Модель должна иметь формат h5 и находится \n'
                           'в одном каталоге с программой, иначе необходимо \n'
                           'указывать полный путь до модели нейросети')
        TitleCl = QLabel('Количество классов:')
        
        self.InputEpochs = QSpinBox(self)
        self.InputEpochs.setRange(5,50)
        self.InputEpochs.setValue(10)

        self.InputBatch = QSpinBox(self)
        self.InputBatch.setRange(5,100)
        self.InputBatch.setValue(15)

        self.InputTrain = QSpinBox(self)
        self.InputTrain.setRange(100,100000)
        self.InputTrain.setValue(1050)

        self.InputValidation = QSpinBox(self)
        self.InputValidation.setRange(100,100000)
        self.InputValidation.setValue(225)

        self.InputTest = QSpinBox(self)
        self.InputTest.setRange(100,100000)
        self.InputTest.setValue(225)

        self.InputScale = QDoubleSpinBox(self)
        self.InputScale.setRange(1.1,5)
        self.InputScale.setValue(1.5)

        self.InputStep = QSpinBox(self)
        self.InputStep.setRange(1,150)
        self.InputStep.setValue(30)

        self.InputClass = QSpinBox(self)
        self.InputClass.setRange(3,20)
        self.InputClass.setValue(3)
        
        self.ImageName = QLineEdit("ImageName", self)
        self.ImageName.setMaxLength(20)

        self.ModelName = QLineEdit("TestModel", self)
        self.ModelName.setMaxLength(20)
        
        btn = QPushButton('Обучить нейросеть', self)
        btn.setToolTip('В результате обучения модель нейросети '
                       'будет сохранена в каталоге с программой')
        btn.resize(btn.sizeHint())
        btn.clicked.connect(self.buttonClicked)
        
        btn1 = QPushButton('Оптимизировать нейросеть', self)
        btn1.setToolTip('В результате оптимизации 3 лучшие модели '
                        'будут сохранены в каталоге с программой')
        btn1.resize(btn1.sizeHint())
        btn1.clicked.connect(self.buttonClicked1)
        
        btn2 = QPushButton('Поиск и распознавание', self)
        btn2.resize(btn2.sizeHint())
        btn2.clicked.connect(self.buttonClicked2)

        grid = QGridLayout(self)

        grid.setSpacing(10)

        grid.addWidget(Title, 1, 0, 1, 4)
        grid.addWidget(TitleEp, 2, 0)
        grid.addWidget(TitleBa, 3, 0)
        grid.addWidget(TitleCl, 4, 0)
        grid.addWidget(self.InputEpochs, 2, 1)
        grid.addWidget(self.InputBatch, 3, 1)
        grid.addWidget(self.InputClass, 4, 1)
        grid.addWidget(TitleTr, 2, 2)
        grid.addWidget(TitleVa, 3, 2)
        grid.addWidget(TitleTe, 4, 2)
        grid.addWidget(self.InputTrain, 2, 3)
        grid.addWidget(self.InputValidation, 3, 3)
        grid.addWidget(self.InputTest, 4, 3)
        grid.addWidget(btn, 5, 0, 1, 2)
        grid.addWidget(btn1, 5, 2, 1, 2)
        
        grid.addWidget(Title1, 6, 0, 1, 4)
        grid.addWidget(TitleSc, 7, 0)
        grid.addWidget(self.InputScale, 7, 1)
        grid.addWidget(TitleSs, 7, 2)
        grid.addWidget(self.InputStep, 7, 3)
        grid.addWidget(TitleNi, 8, 0)
        grid.addWidget(self.ImageName, 8, 1)
        grid.addWidget(TitleNm, 8, 2)
        grid.addWidget(self.ModelName, 8, 3)
        grid.addWidget(btn2, 9, 1, 1, 2)
        
        self.setLayout(grid)
        
        self.setWindowTitle('Система поиска и распознавания сорных растений')
        self.setWindowIcon(QIcon('icon.png'))
        self.show()
    
    def buttonClicked(self):

        train_dir = 'train' # Каталог с данными для обучения
        val_dir = 'val' # Каталог с данными для проверки
        test_dir = 'test' # Каталог с данными для тестирования
        img_width, img_height = 150, 150 # Размеры изображения
        input_shape = (img_width, img_height, 3) # Размерность тензора на основе изображения для входных данных в нейронную сеть
        
        epochs = self.InputEpochs.value()
        Nclasses = self.InputClass.value()
        batch_size = self.InputBatch.value()
        nb_train_samples = self.InputTrain.value()
        nb_validation_samples = self.InputValidation.value()
        nb_test_samples = self.InputTest.value()
        
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(768))
        model.add(Activation('selu'))
        model.add(Dropout(0.5))
        model.add(Dense(Nclasses))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        val_generator = datagen.flow_from_directory(
            val_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        test_generator = datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=nb_validation_samples // batch_size)

        scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)

        print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

        model.save('TestModel.h5')

    def buttonClicked1(self):

        train_dir = 'train'
        val_dir = 'val'
        test_dir = 'test'
        img_width, img_height = 150, 150
        input_shape = (img_width, img_height, 3)

        epochs = self.InputEpochs.value()
        Nclasses = self.InputClass.value()
        batch_size = self.InputBatch.value()
        nb_train_samples = self.InputTrain.value()
        nb_validation_samples = self.InputValidation.value()
        nb_test_samples = self.InputTest.value()

        l=0

        def build_model(hp):  
            model = Sequential()

            num_hidden_layers = hp.Int('num_hidden_layers', 1, 3, default=1)
            num_conv_layers = hp.Int('num_conv_layers', 2, 6, default=2)

            model.add(Conv2D(32, (3, 3), input_shape=input_shape))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for i in range(num_conv_layers):
                filters = hp.Int('filters'+str(i), 32, 64, step=16)
                model.add(Conv2D(filters,(3, 3)))
                model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(128, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
    
            for j in range(num_hidden_layers):
                model.add(Dense(units=hp.Int('units_hiddenNeurons_'+str(j),
                                             min_value=128,
                                             max_value=1024,
                                             step=64),
                                activation=hp.Choice('activation'+str(j),values=['relu','tanh','elu','selu'])))

            model.add(Dropout(0.5))
            model.add(Dense(Nclasses))
            model.add(Activation('softmax'))
            model.compile(
                loss='categorical_crossentropy',
                optimizer=hp.Choice('optimizer', values=['adam','rmsprop','SGD'],default='adam'),
                metrics=['accuracy'])
            return model

        tuner = RandomSearch(
            build_model,
            objective='val_accuracy',
            max_trials=15,
            directory='test_directory')

        tuner.search_space_summary()

        datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        val_generator = datagen.flow_from_directory(
            val_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        test_generator = datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        tuner.search(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=nb_validation_samples // batch_size)

        tuner.results_summary()

        models = tuner.get_best_models(num_models=3)

        for model in models:
            model.summary()
            l=l+1
            scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
            model.save('bestmodel_'+str(l)+'.h5')
            print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))


    def buttonClicked2(self):

        def pyramid(image, scale, minSize=(150, 150)):
            yield image
            while True:
                w = int(image.shape[1] / scale)
                image = imutils.resize(image, width=w)
                if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                    break
                yield image

        def sliding_window(image, stepSize, windowSize):
            # сдвиг окна по изображению
            for y in range(0, image.shape[0], stepSize):
                for x in range(0, image.shape[1], stepSize):
                    yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


        # расстояние между точками
        def dist(x,y):
            return np.sqrt(np.sum(np.square(np.array(center(x))-np.array(center(y)))))

        # вычисляем точку-центр рамки
        def center(frame):
            (x, y, x2, y2, _ , _ , _ ) = frame
            return [ x+((x2-x)//2), y2+((y-y2)//2) ]

        # очистка списка найденных рамок
        def clean(X, i=0, max_dist=150):
            # выбираем основной элемент X[i]
            j = i+1 # и, следующий за ним, элемент X[j]
            if i <= len(X) and i > 0:
                max_dist *= X[i][6]
            while(j<len(X)): # для всех элементов j (!= i)
                d = dist( X[i], X[j] ) # считаем расстояние между точками i,j
                if ( d < max_dist ): # i,j в одном кластере
                    if ( X[i][4] >= X[j][4] ): # сравниваем рейтинг
                        del X[j]  # удаляем элемент j
                    else:
                        del X[i] # удаляем основной элемент i
                    X = clean(X, i=i, max_dist=max_dist) # рекурсивно повторяем уже без элемента i
                    break
                else:
                    j+=1
            if ( i<(len(X)-1) ): # если есть ещё точки в других, относительно X[i], кластерах
                X = clean(X, i=i+1, max_dist=max_dist) # то выполняем проверку
            return X

        if __name__ == "__main__":

            k = 1
            path = os.getcwd()
            classes = os.listdir(path + "/train")
            boxes = []
            scale = self.InputScale.value()
            stepSize = self.InputStep.value()
            imagename = self.ImageName.text()
            modelname = self.ModelName.text()
            font = cv2.FONT_HERSHEY_SIMPLEX
	
            model = load_model(modelname+'.h5')
            img = cv2.imread(imagename+'.jpg')
            width1, height1, channels1 = img.shape
            if (width1 > 1000) or (height1 > 1000):
                img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
            (winW, winH) = (150, 150)
            clone = img.copy()

            for resized in pyramid(img, scale):
                for (x, y, window) in sliding_window(resized, stepSize, windowSize=(winW, winH)):
                    # если окно не соответствует нашему желаемому размеру окна, игнорируем его
                    if window.shape[0] != winH or window.shape[1] != winW:
                        continue
        		
                    img_tensor = image.img_to_array(window)                 # (height, width, channels)
                    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), добавим измерение, потому что модель ожидает эту форму: (batch_size, height, width, channels)
                    img_tensor /= 255.                                      # imshow ожидает значений в диапазоне [0, 1]
		
                    pred = model.predict(img_tensor)
                    nclass = np.argmax(pred)
                    pred = np.amax(pred)
                    if pred >= 0.9:
                        boxes.append([x*k, y*k, (x + winW)*k, (y + winH)*k, pred, classes[nclass], k])

                k *= scale
	

            boxes = clean(boxes)
	
            for (x, y, x2, y2, score, classname, scales) in boxes:
                    cv2.rectangle(clone, (int(x),int(y)),(int(x2), int(y2)),(0, 255, 0), 2)
                    cv2.putText(clone, classname + " %.3f" % score, (int(x),int(y2)), font, 0.7, (0, 255, 0), thickness=2)
            cv2.imshow("Result", clone)
            cv2.waitKey(1)

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Выход',
            "Вы уверены что хотите закрыть программу?", QMessageBox.Yes |
            QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':

    logging.getLogger('tensorflow').disabled = True
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
