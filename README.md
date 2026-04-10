# dfm-agent: Agentic DFM Analysis for Additive Manufacturing

An agentic AI application that takes a raw 3D geometry file as input and returns actionable Design for Manufacturability (DFM) feedback through a conversational, multi-turn interface.

## Features

- **File Upload**: Accept STL files via drag-and-drop.
- **3D Viewer**: Interactive in-browser rendering of the uploaded geometry.
- **Metric Prediction**: Automated analysis of overhang angles, support volume, surface area, and build volume through Geometric Deep Learning.
- **Orientation Optimization**: Compute and visualize the optimal build direction to minimize support material.
- **Autonomous DFM Agent**: An LLM-powered assistant capable of acting on the geometry:
  - Automatically **healing** broken meshes using `trimesh.repair`.
  - Instantly **estimating manufacturing costs** relative to material choice.
  - Generating optimized **slicer settings** (infill %, supports) based on prediction constraints.
- **Model Showcase**: An interactive dashboard highlighting training metrics across multiple deep learning architectures (DiffusionNet, GAT, MeshCNN, PointNet++).
- **Report Export**: Generate PDF DFM reports for your geometry.

## Tech Stack

- **Frontend**: Streamlit
- **Backend Model**: Custom DiffusionNet PyTorch model
- **LLM**: OpenAI API (GPT-4o) with Function Calling capabilities
- **Geometry Processing**: Trimesh, `robust-laplacian`

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
