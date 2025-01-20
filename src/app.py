import pickle
import streamlit as st

# Load the model
parkinson_model = pickle.load(open(r'C:/Users/av725/Downloads/Parkinson-s-Disease-Detection/model/parkinson_model.sav', 'rb'))

# CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://images.unsplash.com/photo-1516574187841-cb9cc2ca948b?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8dHJlYXRtZW50fGVufDB8fDB8fHww');
        background-size: cover;
        background-position: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput > div > input {
        background-color: #f0f0f0;
        color: #000;
        border-radius: 5px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        text-shadow: 2px 2px 4px #000000;
        font-weight: bold;
    }
    p, label {
        color: #ffffff;
        font-weight: bold;
        text-shadow: 1px 1px 2px #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Parkinson's Disease Detection")
st.sidebar.subheader("Navigation")

selection = st.sidebar.radio("Go to", ['Home', "Parkinson Prediction"])

if selection == 'Home':
    st.title("Welcome to Parkinson's Disease Detection")
    st.markdown("""
    This application predicts the likelihood of Parkinson's disease based on various input features.
    Navigate to 'Parkinson Prediction' to get started.
    """)

elif selection == "Parkinson Prediction":
    st.title('Parkinson Prediction using Machine Learning')

    # Default values for the input fields
    default_values = {
        "MDVP: Fundamental Frequency (Fo) in Hz": 119.992,
        "MDVP: Maximum Fundamental Frequency (Fhi) in Hz": 157.302,
        "MDVP: Minimum Fundamental Frequency (Flo) in Hz": 74.997,
        "MDVP: Jitter (%)": 0.00784,
        "MDVP: Jitter (Abs)": 0.00007,
        "MDVP: Relative Amplitude Perturbation (RAP)": 0.00370,
        "MDVP: Five-Point Period Perturbation Quotient (PPQ)": 0.00554,
        "Jitter: Detrended Period Perturbation Quotient (DDP)": 0.01109,
        "MDVP: Shimmer": 0.04374,
        "MDVP: Shimmer in dB": 0.426,
        "Shimmer: Three-Point Amplitude Perturbation Quotient (APQ3)": 0.02182,
        "Shimmer: Five-Point Amplitude Perturbation Quotient (APQ5)": 0.03130,
        "MDVP: Amplitude Perturbation Quotient (APQ)": 0.02971,
        "Shimmer: Detrended Amplitude Perturbation Quotient (DDA)": 0.06545,
        "Noise-to-Harmonics Ratio (NHR)": 0.02211,
        "Harmonics-to-Noise Ratio (HNR)": 21.033,
        "Recurrence Period Density Entropy (RPDE)": 0.414783,
        "Detrended Fluctuation Analysis (DFA)": 0.815285,
        "Spread1": -4.813031,
        "Spread2": 0.266482,
        "D2": 2.301442,
        "Pitch Period Entropy (PPE)": 0.284654
    }

    # Input fields for Parkinson Prediction
    cols1, cols2, cols3 = st.columns(3)

    with cols1:
        MDVP_Fo_Hz = st.number_input('MDVP: Fundamental Frequency (Fo) in Hz', min_value=0.0, max_value=300.0, step=0.1, format='%.1f', value=default_values["MDVP: Fundamental Frequency (Fo) in Hz"])
        MDVP_Fhi_Hz = st.number_input('MDVP: Maximum Fundamental Frequency (Fhi) in Hz', min_value=0.0, max_value=600.0, step=0.1, format='%.1f', value=default_values["MDVP: Maximum Fundamental Frequency (Fhi) in Hz"])
        MDVP_Flo_Hz = st.number_input('MDVP: Minimum Fundamental Frequency (Flo) in Hz', min_value=0.0, max_value=600.0, step=0.1, format='%.1f', value=default_values["MDVP: Minimum Fundamental Frequency (Flo) in Hz"])

    with cols2:
        MDVP_Jitter_percent = st.number_input('MDVP: Jitter (%)', min_value=0.0, max_value=1.0, step=0.01, format='%.2f', value=default_values["MDVP: Jitter (%)"])
        MDVP_Jitter_Abs = st.number_input('MDVP: Jitter (Abs)', min_value=0.0, max_value=0.1, step=0.001, format='%.3f', value=default_values["MDVP: Jitter (Abs)"])
        MDVP_RAP = st.number_input('MDVP: Relative Amplitude Perturbation (RAP)', min_value=0.0, max_value=0.1, step=0.001, format='%.3f', value=default_values["MDVP: Relative Amplitude Perturbation (RAP)"])

    with cols3:
        MDVP_PPQ = st.number_input('MDVP: Five-Point Period Perturbation Quotient (PPQ)', min_value=0.0, max_value=0.1, step=0.001, format='%.3f', value=default_values["MDVP: Five-Point Period Perturbation Quotient (PPQ)"])
        Jitter_DDP = st.number_input('Jitter: Detrended Period Perturbation Quotient (DDP)', min_value=0.0, max_value=0.1, step=0.001, format='%.3f', value=default_values["Jitter: Detrended Period Perturbation Quotient (DDP)"])
        MDVP_Shimmer = st.number_input('MDVP: Shimmer', min_value=0.0, max_value=1.0, step=0.01, format='%.2f', value=default_values["MDVP: Shimmer"])

    cols4, cols5 = st.columns(2)

    with cols4:
        Shimmer_APQ3 = st.number_input('Shimmer: Three-Point Amplitude Perturbation Quotient (APQ3)', min_value=0.0, max_value=1.0, step=0.01, format='%.2f', value=default_values["Shimmer: Three-Point Amplitude Perturbation Quotient (APQ3)"])
        Shimmer_APQ5 = st.number_input('Shimmer: Five-Point Amplitude Perturbation Quotient (APQ5)', min_value=0.0, max_value=1.0, step=0.01, format='%.2f', value=default_values["Shimmer: Five-Point Amplitude Perturbation Quotient (APQ5)"])
        MDVP_APQ = st.number_input('MDVP: Amplitude Perturbation Quotient (APQ)', min_value=0.0, max_value=1.0, step=0.01, format='%.2f', value=default_values["MDVP: Amplitude Perturbation Quotient (APQ)"])

    with cols5:
        Shimmer_DDA = st.number_input('Shimmer: Detrended Amplitude Perturbation Quotient (DDA)', min_value=0.0, max_value=1.0, step=0.01, format='%.2f', value=default_values["Shimmer: Detrended Amplitude Perturbation Quotient (DDA)"])
        NHR = st.number_input('Noise-to-Harmonics Ratio (NHR)', min_value=0.0, max_value=1.0, step=0.01, format='%.2f', value=default_values["Noise-to-Harmonics Ratio (NHR)"])
        HNR = st.number_input('Harmonics-to-Noise Ratio (HNR)', min_value=0.0, max_value=100.0, step=0.1, format='%.1f', value=default_values["Harmonics-to-Noise Ratio (HNR)"])

    cols6, cols7 = st.columns(2)

    with cols6:
        RPDE = st.number_input('Recurrence Period Density Entropy (RPDE)', min_value=0.0, max_value=1.0, step=0.01, format='%.2f', value=default_values["Recurrence Period Density Entropy (RPDE)"])
        DFA = st.number_input('Detrended Fluctuation Analysis (DFA)', min_value=0.0, max_value=2.0, step=0.01, format='%.2f', value=default_values["Detrended Fluctuation Analysis (DFA)"])
        MDVP_Shimmer_dB = st.number_input('MDVP: Shimmer in dB', min_value=0.0, max_value=10.0, step=0.1, format='%.1f', value=default_values["MDVP: Shimmer in dB"])

    with cols7:
        spread1 = st.number_input('Spread1', min_value=-10.0, max_value=10.0, step=0.1, format='%.2f', value=default_values["Spread1"])
        spread2 = st.number_input('Spread2', min_value=-10.0, max_value=10.0, step=0.1, format='%.2f', value=default_values["Spread2"])
        D2 = st.number_input('D2', min_value=0.0, max_value=5.0, step=0.01, format='%.2f', value=default_values["D2"])
        PPE = st.number_input('Pitch Period Entropy (PPE)', min_value=0.0, max_value=1.0, step=0.01, format='%.2f', value=default_values["Pitch Period Entropy (PPE)"])

    # Code for Prediction
    parkinson_diagnosis = ''

    # Creating a Button for Prediction
    if st.button('Parkinson Test Result'):
        parkinson_prediction = parkinson_model.predict([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])

        if parkinson_prediction[0] == 1:
            parkinson_diagnosis = '<p style="color:red; font-size: 20px;">The person has Parkinson\'s disease</p>'
        else:
            parkinson_diagnosis = '<p style="color:green; font-size: 20px;">The person does not have Parkinson\'s disease</p>'

    st.markdown(parkinson_diagnosis, unsafe_allow_html=True)
