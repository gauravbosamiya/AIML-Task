### Name - Bosamiya Gauravkumar
### Mobile No. - 7698099022

## Prerequisites
- Python 3.9 or above
- uv (package manager)

## Install uv (if not installed)
pip install uv

# Step 1 — Clone the project
git clone https://github.com/gauravbosamiya/AIML-Task.git
cd AIML-Task

# Step 2 — Create virtual environment
uv venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Step 3 — Install dependencies
uv pip install -r requirements.txt

# Step 4 — Add saved_model files
Make sure saved_model/ folder contains:
├── saved_model/
│   ├── mobilenet_extractor.h5
│   ├── xgb_demand_model.pkl
│   └── meta.json

# Step 5 — Run FastAPI backend (Terminal 1)
cd AIML-Task
.venv\Scripts\activate
uvicorn api:app --reload

# Step 6 — Run Streamlit frontend (Terminal 2)
cd AIML-Task
.venv\Scripts\activate
streamlit run main.py