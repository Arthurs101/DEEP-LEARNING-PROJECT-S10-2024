from InteractiveModel import Simpsins_CNN

model = Simpsins_CNN("model_weightsk.h5")

# model.describe_self()

prediction = model.input_image("data/simpsons_dataset/simpsons_dataset/abraham_grampa_simpson/pic_0028.jpg")
print(f"your character is {prediction}")