# Well Test Analysis - Pressure Transient Analysis

A powerful web-based application for pressure transient analysis (PTA) in oil and gas wells. This tool helps petroleum engineers analyze drawdown and buildup test data to estimate reservoir properties like permeability, skin factor, and wellbore storage.

[![Open in Streamlit]

## 📋 Overview

This application implements the **Bourdet derivative method** for well test analysis, providing an interactive interface to:
- Load and visualize pressure test data
- Select drawdown and buildup periods interactively
- Calculate pressure derivatives automatically
- Estimate key reservoir parameters through graphical analysis


## Features

###  Interactive Data Selection
- **Visual limit selection** with real-time plot updates
- **Slider controls** for precise time period adjustment
s



##  Quick Start

### Prerequisites
- Python 3.11 or higher
- pip or uv package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/well-test-analysis.git
   cd well-test-analysis
   ```

2. **Install dependencies**
   
   Using pip:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using uv (recommended):
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   
   The app will automatically open at `http://localhost:8501`

## 📝 How to Use

### Step 1: Upload Data
1. Prepare your CSV file with columns: `Date`, `Time`, `Press`
2. Upload via the sidebar file uploader
3. Adjust well parameters in the sidebar:
   - Formation volume factor (bo)
   - Oil viscosity (muo)
   - Flow rate (qo)
   - Net pay thickness (h)
   - Effective porosity (PHIE)
   - Initial pressure (Pi)
   - Total compressibility (ct)
   - Wellbore radius (rw)

### Step 2: Select Time Limits
1. Use the sliders on the left to adjust:
   -  **DD Start**: Beginning of drawdown period
   -  **DD End**: End of drawdown / Start of buildup
   - **BU End**: End of buildup period
2. Watch the plot update in real-time
3. Click " Confirm Limits & Proceed"

### Step 3: Calculate Derivative
1. Click "Calculate Derivative"
2. Wait for the Bourdet derivative calculation to complete

### Step 4: Analyze Results
1. Adjust the **Derivative Level (m)** slider to estimate permeability and skin
2. Adjust the **Unit Slope Derivative** slider to estimate wellbore storage
3. Watch the parameter estimates update in real-time
4. Review final results in the summary table

## 📊 Input Data Format

Your CSV file should have the following structure:

```csv
Date,Time,Press
2024-01-01,00:00:00,5400
2024-01-01,00:15:00,5350
2024-01-01,00:30:00,5300
...
```

### Required Columns:
- **Date**: Date in any standard format (YYYY-MM-DD recommended)
- **Time**: Time in HH:MM:SS format
- **Press**: Pressure readings in psi

## 🔬 Theory & Methodology

### Bourdet Derivative
The application uses the Bourdet derivative method, which is the industry standard for well test interpretation. The derivative helps identify flow regimes and reduces the ambiguity in pressure analysis.

**Key equation:**
```
t(dp/dt) = [(dp_right - dp_center) * Δt_left + (dp_center - dp_left) * Δt_right] / (Δt_left + Δt_right)
```

### Parameters Estimated

1. **Permeability (k)** - in millidarcies (md)
   - Estimated from the horizontal derivative stabilization
   - Formula: k = 70.6 × q × Bo × μo / (h × m)

2. **Skin Factor (s)** - dimensionless
   - Indicates well damage (s > 0) or stimulation (s < 0)
   - Calculated using Horner analysis equations

3. **Wellbore Storage (C)** - in bbl/psi
   - Estimated from the unit slope line on the derivative plot
   - Formula: C = q × Bo × 0.0001 / (24 × Der)

##  Dependencies

```
matplotlib>=3.10.8
numpy>=2.4.3
pandas>=3.0.1
plotly>=6.6.0
streamlit>=1.19.0
altair<5
```


## 📚 References

This application implements methods from:
- IHS Energy Well Test Documentation
- Bourdet, D. et al. (1989). "A New Set of Type Curves Simplifies Well Test Analysis"
- Horner, D.R. (1951). "Pressure Build-Up in Wells"

### Useful Links:
- [IHS Energy Documentation](https://www.ihsenergy.ca/support/documentation_ca/WellTest/)
- [Derivative Analysis](https://www.ihsenergy.ca/support/documentation_ca/WellTest/2019_1/content/html_files/analysis_types/conventional_test_analyses/derivative_analyses.htm)
- [Radial Flow Analysis](https://www.ihsenergy.ca/support/documentation_ca/WellTest/content/html_files/analysis_types/conventional_test_analyses/radial_flow_analysis.htm)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created with  for petroleum engineers


## 🔮 Future Enhancements

- [ ] Export results to PDF repor
- [ ] Type curve matching




For questions, issues, or feature requests, please open an issue on GitHub.

---

**Note**: This tool is intended for petroleum engineering professionals. Results should be validated by qualified engineers before making operational decisions.
