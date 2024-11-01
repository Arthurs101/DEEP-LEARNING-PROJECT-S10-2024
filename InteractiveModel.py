from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from keras.models import model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

class Simpsins_CNN:
    TARGET_NAMES = ['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson', 'brandine_spuckler','carl_carlson',
                  'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler','comic_book_guy', 'disco_stu', 'dolph_starbeam','duff_man','edna_krabappel',
                  'fat_tony', 'gary_chalmers','gil', 'groundskeeper_willie', 'homer_simpson','jimbo_jones', 'kearney_zzyzwicz','kent_brockman', 'krusty_the_clown', 'lenny_leonard',
                  'lionel_hutz', 'lisa_simpson', 'lunchlady_doris','maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby','milhouse_van_houten',
                  'miss_hoover', 'moe_szyslak', 'ned_flanders','nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner',
                  'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum','selma_bouvier', 'sideshow_bob', 'sideshow_mel',
                  'snake_jailbird','troy_mcclure', 'waylon_smithers']
    CLASSES = len(TARGET_NAMES)
    image_size_scaling = (128,128)
    
    def __init__(self,weights_file = None) -> None:
            self.model = Sequential()
            # Gonna work wit a 128x128 image on rgb
            # First Convolutional Block
            self.model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            # Second Convolutional Block
            self.model.add(Conv2D(64, (3,3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            # Third Convolutional Block
            self.model.add(Conv2D(128, (3,3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            # Fourth Convolutional Block
            self.model.add(Conv2D(256, (3,3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            # Flatten and Fully Connected Layers
            self.model.add(Flatten())
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            
             # Output Layer
            self.model.add(Dense(Simpsins_CNN.CLASSES, activation='softmax'))  # output of the classes
       
            if weights_file:
                self.model.load_weights(weights_file)
                
        
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def describe_self(self):
        print(self.model.summary())
    
    def input_image(self,image_path) -> str:
        # Load the image
        img = image.load_img(image_path, target_size=(128,128))
        
        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        
        # Rescale the image
        img_array /= 255.0
        
        processed_image = np.expand_dims(img_array, axis=0)  # Shape will be (1, 128, 128, 3)
    
        # Get the model prediction
        predictions = self.model.predict(processed_image)

        # The output will be a probability distribution over the classes
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = Simpsins_CNN.TARGET_NAMES[predicted_class]

        return predicted_label     
    def get_saliency_map(self, image_path):
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=self.image_size_scaling)
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Shape (1, 128, 128, 3)

        # Make a prediction to determine the class of the image
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Set up GradientTape to watch the input image
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = self.model(img_tensor)
            loss = predictions[:, predicted_class]  # Focus on the predicted class

        # Calculate the gradients of the class score w.r.t. the input image
        grads = tape.gradient(loss, img_tensor)
        
        # Take the absolute value of the gradients and reduce along color channels to create a grayscale map
        grads = tf.abs(grads)
        saliency_map = tf.reduce_max(grads, axis=-1).numpy()[0]
        
        # Normalize the saliency map to range between 0 and 1
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

        # Overlay the saliency map on the original image
        plt.figure(figsize=(8, 8))
        plt.imshow(img, alpha=0.7)  # Display the original image
        plt.imshow(saliency_map, cmap='viridis', alpha=0.8)  # Overlay the saliency map
        plt.title("Saliency Map Overlay")
        plt.axis('off')
        plt.show()