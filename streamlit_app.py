import streamlit as st
import pandas as pd
import numpy as np
from bisect import bisect_right
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Well Test Analysis", layout="wide")

# References
st.sidebar.markdown("""
### References
- [IHS Energy Documentation](https://www.ihsenergy.ca/support/documentation_ca/WellTest/)
""")


# Define prepare data function
def prepare_data(raw_DD_data, raw_BU_data, params, required_test="BU"):
    # add parameters to dictionary
    params["tp"] = (raw_DD_data.loc[len(raw_DD_data) - 1, "DateTime"] - raw_DD_data.loc[0, "DateTime"]).total_seconds() / 3600
    params["BU_duration"] = (raw_BU_data.loc[len(raw_BU_data) - 1, "DateTime"] - raw_BU_data.loc[0, "DateTime"]).total_seconds() / 3600
    params["test_type"] = required_test
    # prepare Buildup data
    if params["test_type"] == "BU":
        raw_data = raw_BU_data.copy()
        raw_data["p"] = raw_data["Press"]
        raw_data["t"] = raw_data["DateTime"].apply(lambda x:  (x - raw_data.loc[0, "DateTime"]).total_seconds() / 3600)
        raw_data["dp"] = raw_data["p"] - raw_data.loc[0, "p"]
        raw_data["te"] = raw_data["t"] * params["tp"] / (raw_data["t"] + params["tp"])
        params["pwf"] = raw_data.loc[0, "p"]
    # prepare Draw-down data
    else:
        raw_data = raw_DD_data.copy()
        raw_data["p"] = raw_data["Press"]
        raw_data["t"] = raw_data["DateTime"].apply(lambda x:  (x - raw_data.loc[0, "DateTime"]).total_seconds() / 3600)
        raw_data["dp"] = params["Pi"] - raw_data["p"]
        raw_data["te"] = raw_data["t"]
        params["pwf"] = np.mean(raw_data["p"])
    return raw_data, params


# define Bourdet derivative function
def calc_der(raw_data, params, req_L=0.1):
    params["L"] = req_L

    # define binary function for pws search
    def BinarySearch(a, x, b=pd.Series([0])):
        i = bisect_right(a, x)
        if len(b) == 1:
            if i:
                return a[i - 1]
            else:
                return np.nan
        else:
            if i:
                return b[i - 1]
            else:
                return np.nan
    
    # prepare Bourdet derivative
    raw_data["X_C"] = np.log(raw_data["te"])
    raw_data["X_L"] = raw_data["X_C"].apply(lambda x: BinarySearch(raw_data["X_C"], x - params["L"]))
    raw_data["X_R"] = raw_data["X_C"].apply(lambda x: BinarySearch(raw_data["X_C"], x + params["L"]))

    raw_data["P_C"] = raw_data["dp"]
    raw_data["P_L"] = raw_data["X_L"].apply(lambda x: BinarySearch(raw_data["X_C"], x, b=raw_data["P_C"]))
    raw_data["P_R"] = raw_data["X_R"].apply(lambda x: BinarySearch(raw_data["X_C"], x, b=raw_data["P_C"]))

    raw_data["t(ddelP/dt)_L"] = (raw_data["P_C"] - raw_data["P_L"]) / (raw_data["X_C"] - raw_data["X_L"])
    raw_data["t(ddelP/dt)_R"] = (raw_data["P_R"] - raw_data["P_C"]) / (raw_data["X_R"] - raw_data["X_C"])
    raw_data["t(ddelP/dt)_R"] = raw_data.apply(lambda x: x["t(ddelP/dt)_L"] if np.isnan(x["t(ddelP/dt)_R"]) else x["t(ddelP/dt)_R"], axis=1)
    raw_data["t(ddelP/dt)_C"] = (((raw_data["X_R"] - raw_data["X_C"]) * raw_data["t(ddelP/dt)_L"]) + ((raw_data["X_C"] - raw_data["X_L"]) * raw_data["t(ddelP/dt)_R"])) / (raw_data["X_R"] - raw_data["X_L"])
    max_te = max(raw_data["t"])
    raw_data["t(ddelP/dt)_C"] = raw_data.apply(lambda x: np.nan if x["t"] > max_te - params["L"] else x["t(ddelP/dt)_C"], axis=1)
    raw_data["derv"] = raw_data["t(ddelP/dt)_C"]

    return raw_data.loc[:, ["t", "p", "dp", "derv"]], params


def calculate_parameters(derv_data, params, m_value, der_value):
    """Calculate permeability, skin, and wellbore storage"""
    if params["test_type"] == "BU":
        k = 70.6 * params["qo"] * params["bo"] * params["muo"] / (params["h"] * m_value)
        s = 1.151 * ((params["Pi"] - params["pwf"]) / (2.303 * m_value) - np.log10(k / (params["PHIE"] * params["muo"] * params["ct"] * params["rw"] ** 2)) + 3.23 - np.log10(params["tp"]))
    else:
        k = 70.6 * params["qo"] * params["bo"] * params["muo"] / (params["h"] * m_value)
        t_fit = derv_data.loc[(derv_data["derv"] >= m_value*0.9) & (derv_data["derv"] <= m_value*1.1), "t"].median()
        pwf_fit = derv_data.loc[(derv_data["derv"] >= m_value * 0.9) & (derv_data["derv"] <= m_value * 1.1), "p"].median()
        s = 1.151 * ((params["Pi"] - pwf_fit) / (2.303 * m_value) - np.log10(k * t_fit / (params["PHIE"] * params["muo"] * params["ct"] * params["rw"] ** 2)) + 3.23)
    
    c = params["qo"] * params["bo"] * 0.0001 / (24 * der_value)
    
    return k, s, c


# Main App
st.title(" Well Test Analysis")
st.markdown("---")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'limits_set' not in st.session_state:
    st.session_state.limits_set = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

# Sidebar - Input Parameters
st.sidebar.header("Input Parameters")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    p_data = pd.read_csv(uploaded_file)
    p_data["DateTime"] = pd.to_datetime(p_data["Date"] + " " + p_data["Time"])
    st.session_state.p_data = p_data
    st.session_state.data_loaded = True
    st.sidebar.success("Data loaded successfully!")
else:
    st.info(" <----Please upload your p_data.csv file from the sidebar")
    st.stop()

# Parameters input
st.sidebar.subheader("Well Parameters")
bo = st.sidebar.number_input("Formation Volume Factor (bo)", value=1.5, format="%.2f")
muo = st.sidebar.number_input("Oil Viscosity (muo, cp)", value=0.35, format="%.3f")
qo = st.sidebar.number_input("Flow Rate (qo, STB/D)", value=800, format="%.0f")
h = st.sidebar.number_input("Net Pay Thickness (h, ft)", value=40, format="%.0f")
PHIE = st.sidebar.number_input("Effective Porosity (PHIE)", value=0.12, format="%.3f")
Pi = st.sidebar.number_input("Initial Pressure (Pi, psi)", value=5410, format="%.0f")
ct = st.sidebar.number_input("Total Compressibility (ct, 1/psi)", value=1e-5, format="%.2e")
rw = st.sidebar.number_input("Wellbore Radius (rw, ft)", value=0.3, format="%.2f")
test_type = st.sidebar.selectbox("Test Type", ["BU", "DD"])
L_value = st.sidebar.slider("Bourdet L Parameter", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

params_dict = {
    "bo": bo, 
    "muo": muo, 
    "qo": qo, 
    "h": h, 
    "PHIE": PHIE, 
    "Pi": Pi, 
    "ct": ct, 
    "rw": rw
}

# Step 1: Select Limits
st.header("Step 1: Select Time Limits")

col1, col2 = st.columns(2)

with col1:
    DD_start = st.number_input("DD Start (index)", min_value=0, max_value=len(p_data)-1, 
                                value=int(np.percentile(p_data.index, 30)))
    DD_end = st.number_input("DD End (index)", min_value=0, max_value=len(p_data)-1, 
                              value=int(np.percentile(p_data.index, 60)))
    BU_end = st.number_input("BU End (index)", min_value=0, max_value=len(p_data)-1, 
                              value=int(np.percentile(p_data.index, 80)))

with col2:
    # Plot pressure data with limits
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_data.index, y=p_data["Press"], 
                             mode='markers', name='Pressure',
                             marker=dict(size=4)))
    
    fig.add_vline(x=DD_start, line_dash="dash", line_color="green", 
                  annotation_text="DD Start")
    fig.add_vline(x=DD_end, line_dash="dash", line_color="red", 
                  annotation_text="DD End")
    fig.add_vline(x=BU_end, line_dash="dash", line_color="black", 
                  annotation_text="BU End")
    
    fig.update_layout(title="Pressure Plot", 
                      xaxis_title="Index", 
                      yaxis_title="Pressure (psi)",
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

if st.button("Confirm Limits and Proceed"):
    # Filter Draw-down and Buildup data
    DD_data = p_data.loc[(p_data.index >= DD_start) & (p_data.index <= DD_end), ["DateTime", "Press"]].copy()
    DD_data.reset_index(inplace=True, drop=True)
    BU_data = p_data.loc[(p_data.index >= DD_end) & (p_data.index <= BU_end), ["DateTime", "Press"]].copy()
    BU_data.reset_index(inplace=True, drop=True)
    
    st.session_state.DD_data = DD_data
    st.session_state.BU_data = BU_data
    st.session_state.limits_set = True
    st.success(" Limits confirmed! Proceed to Step 2 below.")

# Step 2: Calculate Derivative
if st.session_state.limits_set:
    st.markdown("---")
    st.header("Step 2: Calculate Derivative")
    
    if st.button(" Calculate Derivative"):
        with st.spinner("Calculating Bourdet derivative..."):
            data, params_dict = prepare_data(st.session_state.DD_data, 
                                            st.session_state.BU_data, 
                                            params_dict, test_type)
            final_data, params_dict = calc_der(data, params_dict, L_value)
            
            st.session_state.final_data = final_data
            st.session_state.params_dict = params_dict
            st.session_state.analysis_done = True
            st.success(" Derivative calculated!")

# Step 3: Analysis
if st.session_state.analysis_done:
    st.markdown("---")
    st.header("Step 3: Derivative Analysis")
    
    final_data = st.session_state.final_data
    params_dict = st.session_state.params_dict
    
    # Interactive sliders for parameter estimation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Permeability & Skin Estimation")
        m_value = st.slider("Derivative Level (m)", 
                           min_value=float(final_data["derv"].min()), 
                           max_value=float(final_data["derv"].max()), 
                           value=float(final_data["derv"].mean()))
        
        st.subheader("Wellbore Storage Estimation")
        der_value = st.slider("Unit Slope Derivative", 
                             min_value=0.001, 
                             max_value=100.0, 
                             value=0.01,
                             format="%.4f")
    
    with col2:
        # Plot derivative analysis
        fig2 = go.Figure()
        
        # Delta Pressure
        fig2.add_trace(go.Scatter(x=final_data["t"], y=final_data["dp"], 
                                 mode='markers', name='Delta Pressure',
                                 marker=dict(size=4, color='blue')))
        
        # Derivative
        fig2.add_trace(go.Scatter(x=final_data["t"], y=final_data["derv"], 
                                 mode='markers', name='Pressure Derivative',
                                 marker=dict(size=4, color='red')))
        
        # Horizontal line for permeability
        fig2.add_hline(y=m_value, line_dash="dash", line_color="black",
                      annotation_text="Permeability Line")
        
        # Unit slope line
        t_range = np.logspace(np.log10(final_data["t"].min()), 
                             np.log10(final_data["t"].max()), 100)
        unit_slope = der_value * t_range / t_range[0]
        fig2.add_trace(go.Scatter(x=t_range, y=unit_slope, 
                                 mode='lines', name='Unit Slope',
                                 line=dict(dash='dash', color='green')))
        
        fig2.update_xaxes(type="log", title="Time (hr)")
        fig2.update_yaxes(type="log", title="Pressure (psi)")
        fig2.update_layout(title="Pressure Analysis Plot", height=500)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Calculate parameters
    k, s, c = calculate_parameters(final_data, params_dict, m_value, der_value)
    
    # Display Results
    st.markdown("---")
    st.header(" Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Permeability", f"{k:.5f} md")
        st.metric("Draw-down Duration", f"{params_dict['tp']:.2f} hr")
    
    with col2:
        st.metric("Skin Factor", f"{s:.5f}")
        st.metric("Draw-down Rate", f"{params_dict['qo']:.0f} STB/D")
    
    with col3:
        st.metric("Wellbore Storage", f"{c:.5f} bbl/psi")
        st.metric("Buildup Duration", f"{params_dict['BU_duration']:.2f} hr")
    
   

   