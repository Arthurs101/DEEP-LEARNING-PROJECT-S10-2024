from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from keras.models import model_from_json
from tensorflow.keras.preprocessing import image
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