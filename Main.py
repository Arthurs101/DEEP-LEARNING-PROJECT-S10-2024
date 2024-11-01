from InteractiveModel import Simpsins_CNN
import tkinter as tk
model = Simpsins_CNN("model_weightsk.h5")
prediction = model.input_image("data/simpsons_testdataset/simpsons_testdataset/agnes_skinner/001361.gif")
model.get_saliency_map("data/simpsons_testdataset/simpsons_testdataset/agnes_skinner/001361.gif")
print(f"your character is {prediction}")
