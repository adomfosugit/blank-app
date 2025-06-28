import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from scipy.optimize import fsolve
import pandasai
from pandasai import SmartDataframe
from pandasai.llm import BambooLLM

st.set_page_config(page_title="BHP Estimator", layout="wide")
st.title('Bottom Hole Pressure (BHP) Estimator')
st.subheader("Upload your production data or enter manual inputs")

# Required columns for file validation
required_columns = ['PRODUCTION DATE', 'Qo', 'THT', 'GOR ', 'Pwh(psi)', 'THT', 'Depth', 'WCT']

# Initialize session state variables
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Sidebar for model selection
with st.sidebar:
    st.header("Model Configuration")
    
    # Model selection
    model_option = st.selectbox(
        "Select Well Model",
        ["J57 & J05", "J19 & J56", "J37 & J51"],
        index=0,
        help="Select the appropriate model for your well"
    )
    
    # Model file mapping
    model_files = {
        "J57 & J05": 'modelBIGDATA5US1NNMAXP57.pkl(1)',
        "J19 & J56": 'modelBIGDATA5US1NNMAXP576168allL1NN.pkl(1)',
        "J37 & J51": 'modelBIGDATA5US1NNMAXP5137.pkl(1)'
    }

# Model loading function with caching
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            saved_data = pickle.load(file)
            model = saved_data['model']
            scaler = saved_data['scaler']
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

# Load the selected model
model, scaler = load_model(model_files[model_option])
if model is None or scaler is None:
    st.stop()

# Define tabs only once
tab1, tab2, tab3, tab4 = st.tabs(["📁 File Prediction", "✍️ Manual Prediction", "🔍 Solve for Parameter", "🧠 AI Data Analysis "])

with tab1:
    # File uploader section
    st.header("File-based Prediction")
    st.info(f"Using model: {model_option}")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'], key="file_uploader")

    # File processing and validation
    if uploaded_file is not None:
        try:
            # Read file based on type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            # Convert to numeric and drop NA
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
            
            # Check required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                # Convert date and add features
                df['PRODUCTION DATE'] = pd.to_datetime(df['PRODUCTION DATE'])
                df['Fluid gradient'] = (df['WCT']/100)*0.433 + (1-(df['WCT']/100))*0.273
                df['Ph'] = df['Fluid gradient'] * df['Depth']
                
                st.session_state.original_df = df
                st.success("File successfully processed!")
                
                # Show preview
                st.write("Data Preview:")
                st.dataframe(df.head(), use_container_width=True)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Prediction function for file data
    def make_predictions():
        if st.session_state.original_df is not None:
            try:
                # Prepare data
                scaled_features = ['Qo', 'GOR ', 'THT', 'Pwh(psi)', 'Ph', 'Depth']
                X_test_scaled = scaler.transform(st.session_state.original_df[scaled_features])
                
                # Make predictions
                predictions = model.predict(X_test_scaled)
                st.session_state.predictions = predictions.flatten()
                
                # Add to dataframe for display
                result_df = st.session_state.original_df.copy()
                result_df['Predicted_BHP'] = st.session_state.predictions
                result_df['Model_Used'] = model_option
                
                return result_df
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                return None
        else:
            st.error("Please upload and process a file first")
            return None

    # Prediction button
    if st.button("Make Predictions from File", key="file_predict"):
        result_df = make_predictions()
        if result_df is not None:
            st.write("Prediction Results:")
            st.dataframe(result_df[['PRODUCTION DATE', 'Predicted_BHP', 'Model_Used']], use_container_width=True)

    # Visualization
    if st.session_state.predictions is not None and st.session_state.original_df is not None:
        st.subheader("Visualization")
        
        # Create dataframe for plotting
        plot_df = st.session_state.original_df.copy()
        plot_df['Predicted_BHP'] = st.session_state.predictions
        
        # Line chart comparison
        st.line_chart(
            plot_df.set_index('PRODUCTION DATE')[['Predicted_BHP']],
            use_container_width=True
        )

with tab2:
    st.header("Manual Single Prediction")
    st.info(f"Using model: {model_option}")
   
    with st.form("manual_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            qo = st.number_input("Oil Rate (Qo, STB/d)", min_value=0.0)
            gor = st.number_input("Gas-Oil Ratio (GOR, scf/STB)", min_value=0.0)
            tht = st.number_input("Tubing Head Temperature (THT, °C)", min_value=0.0)
        with col2:
            pwh = st.number_input("Wellhead Pressure (Pwh, psi)", min_value=0.0)
            wct = st.number_input("Water Cut (WCT, %)", min_value=0.0, max_value=100.0)
            depth = st.number_input("Depth m", min_value=0.0, max_value=10000.0)
        
        submitted = st.form_submit_button("Predict BHP")
        
        if submitted:
            try:
                # Calculate derived features
                fluid_gradient = (wct/100)*0.433 + (1-(wct/100))*0.273
                ph = fluid_gradient * depth
                
                # Prepare input array
                input_data = np.array([[qo, gor, tht, pwh, ph, depth]])
                
                # Scale and predict
                scaled_input = scaler.transform(input_data)
                prediction = model.predict(scaled_input)[0][0]
                
                # Display results
                st.success(f"Predicted Bottom Hole Pressure: **{prediction:.2f} psi**")
                
                # Show input summary with well information
                st.subheader("Input Summary")
                input_summary = {
                    "Parameter": [ "Oil Rate", "GOR", "THT", "Wellhead Pressure", 
                                "Depth", "Water Cut", "Fluid Gradient",  "Model Used"],
                    "Value": [
                        f"{qo} STB/d", 
                        f"{gor} scf/STB", 
                        f"{tht} °C", 
                        f"{pwh} psi", 
                        f"{depth} m", 
                        f"{wct}%", 
                        f"{fluid_gradient:.4f} psi/ft", 
                        model_option
                    ],
                    "Units": [ "STB/d", "scf/STB", "°C", "psi", "m", "%", "psi/ft", ""]
                }
                st.table(pd.DataFrame(input_summary))
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

with tab3:
    st.header("Iterative Parameter Solver")
    st.info("Solve for Qo, GOR, or WCT to match a target BHP")
    
    # Select which parameter to solve for
    target_param = st.selectbox(
        "Parameter to Solve For",
        ["Qo", "GOR", "WCT"],
        index=0
    )
    
    # Input known values with defaults
    col1, col2 = st.columns(2)
    with col1:
        qo = st.number_input("Oil Rate (Qo, STB/d)", min_value=0.0, value=1000.0, key="solve_qo")
        gor = st.number_input("Gas-Oil Ratio (GOR, scf/STB)", min_value=0.0, value=500.0, key="solve_gor")
        tht = st.number_input("Tubing Head Temp (THT, °C)", min_value=0.0, value=10.0, key="solve_tht")
    with col2:
        pwh = st.number_input("Wellhead Pressure (Pwh, psi)", min_value=0.0, value=300.0, key="solve_pwh")
        depth = st.number_input("Depth (m)", min_value=0.0, value=5000.0, key="solve_depth")
        wct = st.number_input("Water Cut (WCT, %)", min_value=0.0, max_value=100.0, value=30.0, key="solve_wct")
    
    # Target BHP value to match
    target_bhp = st.number_input("Target BHP (psi)", min_value=0.0, value=2000.0, key="target_bhp")
    
    if st.button("Solve Iteratively"):
        try:
            # Define the equation to solve with WCT constraints
            def equation_to_solve(x):
    # Enforce constraints depending on the parameter to solve
                if target_param == "WCT":
                    x[0] = np.clip(x[0], 0.0, 100.0)
                elif target_param == "Qo":
                    x[0] = max(x[0], 0.0)  # Prevent negative oil rate
                elif target_param == "GOR":
                    x[0] = max(x[0], 0.0)  # Prevent negative GOR

    # Prepare inputs
                inputs = {
                    'Qo': x[0] if target_param == "Qo" else qo,
                    'GOR ': x[0] if target_param == "GOR" else gor,
                    'THT': tht,
                    'Pwh(psi)': pwh,
                    'Depth': depth,
                    'WCT': x[0] if target_param == "WCT" else wct
                }

    # Calculate fluid gradient and Ph
                current_wct = inputs['WCT']
                fluid_gradient = (current_wct/100)*0.433 + (1-(current_wct/100))*0.273
                ph = fluid_gradient * inputs['Depth']

    # Scale inputs and predict BHP
                scaled_input = scaler.transform(np.array([
                    [inputs['Qo'], inputs['GOR '], inputs['THT'], 
                     inputs['Pwh(psi)'], ph, inputs['Depth']]
                ]))
                current_bhp = model.predict(scaled_input)[0][0]

                return current_bhp - target_bhp

            # Initial guesses (parameter-specific)
            initial_guess = {
                "Qo": max(qo, 100),  # Avoid 0 for Qo
                "GOR": max(gor, 300),  # Avoid 0 for GOR
                "WCT": np.clip(wct, 1.0, 99.0)  # Avoid boundaries for stability
            }[target_param]
            
            # Solve with numerical safeguards
            solution = fsolve(
                equation_to_solve, 
                [initial_guess],
                xtol=1e-6  # Tight tolerance for precision
            )
            
            # Clip WCT to [0, 100] if solved
            if target_param == "WCT":
                solution[0] = np.clip(solution[0], 0.0, 100.0)
            
            # Display results
            st.success(f"**Solved {target_param} = {solution[0]:.2f}** (for BHP = {target_bhp} psi)")
            
            # Show all parameters
            st.subheader("Solution Summary")
            results = pd.DataFrame({
                "Parameter": ["Qo", "GOR", "WCT", "THT", "Pwh", "Depth", "Target BHP"],
                "Value": [
                    f"{solution[0] if target_param == 'Qo' else qo:.2f}",
                    f"{solution[0] if target_param == 'GOR' else gor:.2f}",
                    f"{solution[0] if target_param == 'WCT' else wct:.2f}%",
                    f"{tht:.2f}",
                    f"{pwh:.2f}",
                    f"{depth:.2f}",
                    f"{target_bhp:.2f}"
                ],
                "Units": ["STB/d", "scf/STB", "%", "°C", "psi", "m", "psi"]
            })
            st.dataframe(results, hide_index=True)
            
        except Exception as e:
            st.error(f"Solving failed: {str(e)}")
            st.warning("Check if the target BHP is physically achievable with given inputs.")
with tab4:
    st.header("AI-Powered Data Analysis")
    st.caption("Ask questions about your uploaded data using natural language.")

    # Set PandasAI API key
    pandasai.api_key.set("PAI-09de3b8d-edb9-4d72-998f-60508abfb286")

    # Upload data file if not done already
    uploaded_file_ai = uploaded_file if uploaded_file is not None else st.file_uploader("Upload a CSV or Excel file for analysis", type=["csv", "xlsx"], key="ai_file")

    if uploaded_file_ai is not None:
        try:
            if uploaded_file_ai.name.endswith(".csv"):
                df_ai = pd.read_csv(uploaded_file_ai)
            else:
                df_ai = pd.read_excel(uploaded_file_ai)

            # Display data preview
            st.dataframe(df_ai.head(), use_container_width=True)

            # Wrap with SmartDataFrame
            sdf = SmartDataframe(df_ai, config={"llm": BambooLLM(api_key="PAI-09de3b8d-edb9-4d72-998f-60508abfb286")})

            # Ask the user for a natural language query
            user_query = st.text_area("Ask a question about your data", placeholder="")

            if st.button("Run AI Query", key="run_pandasai"):
                with st.spinner("Thinking..."):
                    try:
                        result = sdf.chat(user_query)
                        st.subheader("AI Response:")
                        st.write(result)
                    except Exception as e:
                        st.error(f"Error processing query: {e}")

        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")
    else:
        st.info("Upload a file to begin AI analysis.")
# Additional data exploration (only for file data)
if st.session_state.original_df is not None:
    with st.expander("Advanced Data Exploration"):
        selected_columns = st.multiselect(
            "Select parameters to visualize",
            st.session_state.original_df.columns.drop('PRODUCTION DATE'),
            default=['Qo']
        )
        
        if selected_columns:
            st.line_chart(
                st.session_state.original_df.set_index('PRODUCTION DATE')[selected_columns],
                use_container_width=True
            )