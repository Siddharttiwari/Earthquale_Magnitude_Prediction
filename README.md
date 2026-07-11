# 🌍 Earthquake Magnitude Prediction

An advanced, data-driven machine learning application that uses live geospatial telemetry to predict earthquake magnitudes and evaluate regional infrastructure threat profiles. 

Instead of relying on static inputs or simulated parameters, this production-ready dashboard interfaces dynamically with the **United States Geological Survey (USGS) REST API** to stream real-time environmental context for localized predictions.

---

## 🚀 Key Features

* **Live USGS API Integration:** Connects to global seismic arrays to calculate real-time average depth and tracking station metrics based on recent localized tremors.
* **Custom Lookback Filters:** Includes an interactive historical data window selector (1 to 90 days) to adjust structural sampling depth cycles.
* **Geospatial Hazard Mapping:** Dynamically plots precise latitude and longitude epicenters of active fault ripples on an interactive map.
* **Model Inference Engine:** Runs an optimized **Random Forest Regressor** to evaluate non-linear tectonic coordinates contextually against baseline monitoring sizes.
* **Automated Risk Profiling:** Translates mathematical magnitude predictions into standard seismological threat levels (Nominal, Elevated, Critical) and maps localized safety boundaries.

---

## 📊 Core Model Mechanics & Logic

During initial architecture evaluations, multiple machine learning approaches were benchmarked to identify the best production engine:

| Model | Evaluation & Performance Verdict |
| :--- | :--- |
| **Random Forest** | **Best Choice (Deployed):** Captures irregular, non-linear geographical fault lines and handles highly correlated variables (like station count vs. magnitude scale) with extreme stability. |
| **Support Vector Machine** | **Runner-Up:** Competent at boundary mapping, but sensitive to spatial scale variances and noise without heavy feature normalization. |
| **Linear Regression** | **Unusable:** Fails completely due to its straight-line mathematical constraints, producing massive out-of-bounds mathematical anomalies on spatial map clusters. |

### Dataset Inputs (Features)
1. **`Latitude(deg)` & `Longitude(deg)`**: Maps precise location coordinates relative to global tectonic plate boundaries.
2. **`Depth(km)`**: Vertical distance down to the underground hypocenter, indicating surface energy dissipation scales.
3. **`No_of_Stations`**: Number of active global seismometers tracking the displacement, acting as a strong proxy for earthquake shockwave propagation scale.

---

## 🛠️ Installation & Setup

### 1. Clone or Move Files into Directory
Ensure your local project directory contains the following core files:
```text
📦 Earthquake-Prediction
 ┣ 📂 .streamlit
 ┃ ┗ 📜 config.toml           # UI Theme configurations
 ┣ 📜 earthquake_app.py       # Core Streamlit dashboard engine
 ┣ 📜 random_forest.pkl       # Trained serialized model artifact
 ┗ 📜 README.md               # Documentation