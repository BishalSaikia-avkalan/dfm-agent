# dfm-agent: Agentic DFM Analysis for Additive Manufacturing

An agentic AI application that takes a raw 3D geometry file as input and returns actionable Design for Manufacturability (DFM) feedback through a conversational, multi-turn interface.

## Features

- **File Upload**: Accept STL, OBJ, or OFF files via drag-and-drop.
- **3D Viewer**: Interactive in-browser rendering of the uploaded geometry.
- **Metric Prediction**: Automated analysis of overhang angles, support volume, surface area, and build volume.
- **Orientation Optimization**: Compute and visualize the optimal build direction to minimize support material.
- **Conversational Agent**: LLM-powered chat interface for follow-up questions and design suggestions.
- **Report Export**: Generate PDF DFM reports for your geometry.

## Preview

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/ed0c6658-4b08-489f-82b2-61a9338f74da" />



<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/9c932674-59ff-4542-b4aa-0ecad6c077fe" />


<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/9cedf6fe-7637-4baa-9bbb-017d3e7468d7" />



## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI / Python
- **LLM**: Claude API (Anthropic) for conversational reasoning
- **Geometry**: Trimesh, PyTorch (for metric prediction models)

## Getting Started

### Prerequisites

- Python 3.8+
- An OpenAI API Key (configured in `.env`)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd webapp
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory and add your API key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the App

Start the Streamlit application:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.
