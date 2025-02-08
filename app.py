import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.sidebar.title("Crop & Fertilizer Recommendation System for Sustainable Agriculture")
app_mode = st.sidebar.radio("Select Page",["HOME","CROP RECOMMENDATION","FERTILIZER RECOMMENDATION"])

from PIL import Image
img = Image.open("Diseases.png")

st.image(img)

if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>Crop & Fertilizer Recommendation System for Sustainable Agriculture", unsafe_allow_html=True)

elif(app_mode=="CROP RECOMMENDATION"):
    st.header("Crop Recommendation System")
    dtc = joblib.load("crop_prediction_model.pkl")
    scaler = joblib.load("crop_scaler.pkl")
    crop_dict = {1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans', 5: 'pigeonpeas',
                 6: 'mothbeans', 7: 'mungbean', 8: 'blackgram', 9: 'lentil', 10: 'pomegranate',
                 11: 'banana', 12: 'mango', 13: 'grapes', 14: 'watermelon', 15: 'muskmelon',
                 16: 'apple', 17: 'orange', 18: 'papaya', 19: 'coconut', 20: 'cotton',
                 21: 'jute', 22: 'coffee'}
    N = st.number_input("Nitrogen (N)", min_value=0.0,value=0.0)
    P = st.number_input("Phosphorus (P)",min_value=0.0, value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0, value=0.0)
    temp = st.number_input("Temperature (°C)", value=0.0)
    hum = st.number_input("Humidity (%)",min_value=0.0, max_value=100.0,value=0.0)
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0)
    rain = st.number_input("Rainfall (mm)",min_value=0.0, max_value=1200.00, value=0.0)
    def crop_rec(N, P, K, temp, hum, ph, rain):
        features = np.array([[N, P, K, temp, hum, ph, rain]])
        transformed_features = scaler.transform(features)
        prediction = dtc.predict(transformed_features)
        crop = crop_dict.get(prediction[0], "Unknown Crop")
        return f"{crop} is the recommended crop for the given conditions."
    if st.button("Predict"):
        col1, col2 = st.columns(2)
        with col1:
            labels = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
            sizes = [N, P, K]
            colors = ['#ff9999', '#66b3ff', '#99ff99']

            def custom_autopct(pct, allsizes):
                total = sum(allsizes)
                absolute = round(pct / 100. * total, 1)
                closest_value = min(allsizes, key=lambda x: abs(x - absolute))
                return f' {closest_value} units ({pct:.1f}%)'

            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

            ax.pie(sizes, labels=labels, colors=colors,
                   autopct=lambda pct: custom_autopct(pct, sizes),
                   startangle=90, wedgeprops=dict(width=0.3))
            plt.title('NPK Composition', size=20, color='blue', fontweight='bold')
            plt.axis('equal')
            st.pyplot(fig)

            rfall = (rain / 1200) * 100
            sizes = [rfall, 100 - rfall]  # Example values
            colors = ['#66b3ff', '#D3D3D3']
            fig, ax = plt.subplots()
            ax.pie(sizes, colors=colors,
                   startangle=90)  # Edge color for visibility
            plt.title(f'Rain Fall : {rain} mm')
            plt.axis('equal')
            st.pyplot(fig)

        with col2:
            sizes = [hum, 100 - hum]  # Example values
            colors = ['#4169E1', '#D3D3D3']
            fig, ax = plt.subplots()
            ax.pie(sizes, colors=colors,
                   startangle=90)  # Edge color for visibility
            plt.title(f'Humidity : {hum} %')
            plt.axis('equal')
            st.pyplot(fig)


            sizes = [temp, 100-temp]  # Example values
            colors = ['#FFA500', '#D3D3D3']
            fig, ax = plt.subplots()
            ax.pie(sizes, colors=colors,
                   startangle=90)  # Edge color for visibility
            plt.title(f'Temperature : {temp}°C')
            plt.axis('equal')
            st.pyplot(fig)

        ph_values = np.linspace(0, 14, 100)

        # Define a colormap for pH scale
        colors = [
            (1, 0, 0),  # Red (Strong Acid, pH 0)
            (1, 0.5, 0),  # Orange
            (1, 1, 0),  # Yellow
            (0, 1, 0),  # Green (Neutral, pH 7)
            (0, 0, 1),  # Blue (Weak Base)
            (0.5, 0, 1)  # Purple (Strong Base, pH 14)
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("pH Scale", colors, N=100)

        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 1))

        # Create a gradient color bar
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        ax.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 14, 0, 1])

        # Set x-axis labels for pH values
        ax.set_xticks(np.arange(0, 15, 1))
        ax.set_xticklabels(np.arange(0, 15, 1))
        ax.set_yticks([])  # Hide y-axis

        # Mark a specific pH point
        ph_point = ph  # Change this to mark another pH value
        ax.scatter(ph_point, 0.5, color="black", s=100, label=f'pH {ph_point}')
        ax.vlines(ph_point, 0, 1, color="black", linestyle="dashed", linewidth=1)

        # Add title and legend
        ax.set_title("pH Scale Visualization", fontsize=12)
        ax.legend(loc="upper right")

        st.pyplot(fig)


        result = crop_rec(N, P, K, temp, hum, ph, rain)
        st.success(result)

elif(app_mode=="FERTILIZER RECOMMENDATION"):
    st.header("Fertilizer Recommendation System")
    # Load the model and scaler
    model = joblib.load('fertilizer_prediction_model.pkl')
    sc = joblib.load('fertilizer_scaler.pkl')

    # Fertilizer dictionary
    fert_dict = {1: 'Urea', 2: 'DAP', 3: '14-35-14', 4: '28-28', 5: '17-17-17', 6: '20-20', 7: '10-26-26'}

    # Soil type dictionary
    soil_type_dict = {0: 'Black', 1: 'Clayey', 2: 'Loamy', 3: 'Red', 4:'Sandy'}

    # Crop type dictionary
    crop_type_dict = {
    0: "Barley",
    1: "Cotton",
    2: "Ground Nuts",
    3: "Maize",
    4: "Millets",
    5: "Oil seeds",
    6: "Paddy",
    7: "Pulses",
    8: "Sugarcane",
    9: "Tobacco",
    10: "Wheat"
}



    # Function to recommend fertilizer
    def recommend_fertilizer(Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous):
        features = np.array([[Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous]])
        transformed_features = sc.transform(features)
        prediction = model.predict(transformed_features).reshape(1, -1)
        fertilizer = [fert_dict[i] for i in prediction[0]]
        return f"{fertilizer[0]} is the best fertilizer for the given conditions"


    # Streamlit app
    Temparature = st.number_input('Temperature', min_value=0.0, max_value=100.0, value=0.0)
    Humidity = st.number_input('Humidity', min_value=0.0, max_value=100.0, value=0.0)
    Moisture = st.number_input('Moisture', min_value=0.0, max_value=100.0, value=0.0)
    Soil_Type = st.selectbox('Soil Type', options=list(soil_type_dict.values()))
    Crop_Type = st.selectbox('Crop Type', options=list(crop_type_dict.values()))
    Nitrogen = st.number_input('Nitrogen', min_value=0.0, max_value=100.0, value=0.0)
    Potassium = st.number_input('Potassium', min_value=0.0, max_value=100.0, value=0.0)
    Phosphorous = st.number_input('Phosphorous', min_value=0.0, max_value=100.0, value=0.0)

    if st.button('Recommend Fertilizer'):
        col1, col2 = st.columns(2)

        with col1:
            labels = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
            sizes = [Nitrogen, Potassium, Phosphorous]
            colors = ['#ff9999', '#66b3ff', '#99ff99']

            def custom_autopct(pct, allsizes):
                total = sum(allsizes)
                absolute = round(pct / 100. * total, 1)
                closest_value = min(allsizes, key=lambda x: abs(x - absolute))
                return f' {closest_value} units ({pct:.1f}%)'

            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

            ax.pie(sizes, labels=labels, colors=colors,
                   autopct=lambda pct: custom_autopct(pct, sizes),
                   startangle=90, wedgeprops=dict(width=0.3))
            plt.title('NPK Composition', size=20, color='blue', fontweight='bold')
            plt.axis('equal')
            st.pyplot(fig)

            sizes = [Humidity, 100 - Humidity]  # Example values
            colors = ['#4169E1', '#D3D3D3']
            fig, ax = plt.subplots()
            ax.pie(sizes, colors=colors,
                   startangle=90)  # Edge color for visibility
            plt.title(f'Humidity : {Humidity} %')
            plt.axis('equal')
            st.pyplot(fig)

        with col2:
            sizes = [Moisture, 100 - Moisture]  # Example values
            colors = ['#4169E1', '#D3D3D3']
            fig, ax = plt.subplots()
            ax.pie(sizes, colors=colors,
                   startangle=90)  # Edge color for visibility
            plt.title(f'Moisture : {Moisture} %')
            plt.axis('equal')
            st.pyplot(fig)

            sizes = [Temparature, 100 - Temparature]  # Example values
            colors = ['#FFA500', '#D3D3D3']
            fig, ax = plt.subplots()
            ax.pie(sizes, colors=colors,
                   startangle=90)  # Edge color for visibility
            plt.title(f'Temperature : {Temparature}°C')
            plt.axis('equal')
            st.pyplot(fig)


        # Convert soil and crop types to their numerical values
        Soil_Type_num = list(soil_type_dict.keys())[list(soil_type_dict.values()).index(Soil_Type)]
        Crop_Type_num = list(crop_type_dict.keys())[list(crop_type_dict.values()).index(Crop_Type)]

        result = recommend_fertilizer(Temparature, Humidity, Moisture, Soil_Type_num, Crop_Type_num, Nitrogen,
                                      Potassium, Phosphorous)
        st.success(result)
