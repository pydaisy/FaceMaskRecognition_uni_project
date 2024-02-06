import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Ustawienie zmiennej środowiskowej na 0

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import random



class FaceMaskDetector:

    def __init__(self, dataset_path="training_dataset_clean", categories=["mask", "no_mask"], init_lr=1e-4, epochs=20, batch_size=32):
        self.dataset_path = dataset_path
        self.categories = categories
        self.init_lr = init_lr
        self.epochs = epochs
        self.batch_size = batch_size

    def load_data(self):
        data = []
        labels = []

        for category in self.categories:
            path = os.path.join(self.dataset_path, category)
            category_data = []
            category_labels = []
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                image = load_img(img_path, target_size=(244, 244, 3))
                image = img_to_array(image)
                image = preprocess_input(image)

                category_data.append(image)
                category_labels.append(category)

            data.extend(category_data)
            labels.extend(category_labels)

        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        labels = to_categorical(labels)

        data = np.array(data, dtype="float32")
        labels = np.array(labels)

        train_data, validation_data, train_labels, validation_labels = train_test_split(
            data,
            labels,
            test_size=0.20,
            stratify=labels,
            random_state=random.randint(1, 1000)
        )

        return train_data, train_labels, validation_data, validation_labels, lb

    def build_model(self):
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(244, 244, 3)))

        model_head = base_model.output
        model_head = AveragePooling2D(pool_size=(7, 7))(model_head)
        model_head = Flatten(name="flatten")(model_head)
        model_head = Dense(128, activation="relu")(model_head)
        model_head = Dropout(0.5)(model_head)
        model_head = Dense(2, activation="softmax")(model_head)

        model = Model(inputs=base_model.input, outputs=model_head)

        for layer in base_model.layers:
            layer.trainable = False

        return model

    def compile_model(self, model):
        opt = Adam(learning_rate=self.init_lr)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    def train_model(self, model, train_data, train_labels, validation_data, validation_labels):
        aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        steps_per_epoch = len(train_data) // self.batch_size
        if len(train_data) % self.batch_size != 0:
            steps_per_epoch += 1

        validation_steps = len(validation_data) // self.batch_size
        if len(validation_data) % self.batch_size != 0:
            validation_steps += 1

        history = model.fit(
            aug.flow(train_data, train_labels, batch_size=self.batch_size),
            steps_per_epoch=steps_per_epoch,
            validation_data=(validation_data, validation_labels),
            validation_steps=validation_steps,
            epochs=self.epochs
        )

        return history

    def evaluate_model(self, model, test_data, test_labels, label_binarizer):
        pred_probs = model.predict(test_data, batch_size=self.batch_size)
        predIdxs = np.argmax(pred_probs, axis=1)

        print(classification_report(test_labels.argmax(axis=1), predIdxs, target_names=label_binarizer.classes_))

    def save_model(self, model, filename="model_mask_detector.h5"):
        model.save(filename, include_optimizer=True, save_format="h5")

    def plot_metrics(self, history: object, filename: object = "plot.png") -> object:
        N = self.epochs
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), history.history["accuracy"], label="train_accuracy")
        plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_accuracy")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(filename)

if __name__ == "__main__":

    face_mask_detector = FaceMaskDetector()
    train_data, train_labels, validation_data, validation_labels, lb = face_mask_detector.load_data()

    model = face_mask_detector.build_model()
    face_mask_detector.compile_model(model)

    history = face_mask_detector.train_model(model, train_data, train_labels, validation_data, validation_labels)
    face_mask_detector.evaluate_model(model, validation_data, validation_labels, lb)
    face_mask_detector.save_model(model)
    face_mask_detector.plot_metrics(history)