import streamlit as st
from PIL import Image
from InteractiveModel import Simpsins_CNN
import numpy as np
import matplotlib.pyplot as plt
import io

# Inicializa el modelo
model = Simpsins_CNN("model_weightsk.h5")

def main():
    st.title("Simpsons Character Recognition")
    st.write("Sube una imagen de un personaje de Los Simpson y el modelo intentará reconocerlo.")

    # Cargar imagen
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "gif"])
    
    if uploaded_file:
        # Muestra la imagen cargada
        img = Image.open(uploaded_file)
        st.image(img, caption="Imagen cargada", use_column_width=True)
        
        # Predicción
        if st.button("Predecir personaje"):
            prediction = model.input_image(uploaded_file)
            st.write(f"Personaje Predicho: {prediction}")

        # Mapa de saliencia
        if st.button("Mostrar mapa de saliencia"):
            # Obtener el mapa de saliencia como imagen
            saliency_img = get_saliency_map_image(uploaded_file)
            st.image(saliency_img, caption="Mapa de Saliencia", use_column_width=True)

def get_saliency_map_image(image_file):
    """Genera el mapa de saliencia y devuelve la imagen."""
    img = Image.open(image_file)
    img_array = np.array(img.resize((128, 128))) / 255.0  # Normaliza la imagen
    
    # Calcula el mapa de saliencia usando el método `get_saliency_map` de `Simpsins_CNN`
    fig, ax = plt.subplots()
    model.get_saliency_map(image_file)
    
    # Guarda el gráfico como imagen en un buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

if __name__ == "__main__":
    main()
