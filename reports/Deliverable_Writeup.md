# Agentic DFM Analyzer: 2-Page Writeup

## 1. Architecture Decisions

### 1.1 Mesh-Native Inference via DiffusionNet
A core architectural decision was processing raw 3D mesh data directly rather than converting it to intermediate voxel or point-cloud representations (which lose geometric surface topology and exact edge constraints). **DiffusionNet** was selected because it leverages the spectral decomposition of the Laplace-Beltrami operator to compute discretization-agnostic, intrinsic mesh convolutions. This provides high predictive accuracy across varied mesh triangulations directly from raw STL/OBJ outputs.

The model outputs both a **regression target** (predicted volume) and **classification logits** for five DFM rules (surface area constraints, contour complexity, contour length limits, overhang limits, and overall pass/fail status). To offer deep model explainability, we explicitly modified the architecture with block-sparse gradient tracking to generate **GradCAM saliency maps**. This allows the network not only to output a binary constraint check, but also to visually spotlight the exact vertices triggering a defect prediction.

### 1.2 Frontend & Backend Structure
The user interface and inference server were consolidated into a **Streamlit** dashboard. Streamlit’s rapid prototyping environment integrates seamlessly with `Plotly` (for the 3D viewer interacting directly with `trimesh` geometry) and the PyTorch backend. This unified architecture avoids network serialization overhead for large meshes during 3D rendering.

## 2. Agent Design

The application utilizes an OpenAI `gpt-4o` powered **conversational agent** enhanced with function calling. Unlike typical LLM agents that rely on unstructured RAG for domain knowledge, this agent connects explicitly with deterministic mesh algorithms and AI model outputs via the following tools:

- `get_dfm_analysis`: Surfaces the numerical predictions from DiffusionNet and bounding box logic.
- `get_design_suggestions`: Maps specific metric failures (e.g. "Contour Complexity") to actionable human engineering advice (e.g. "Simplify geometry").
- `get_orientation_advice`: Calculates the exact bounding box and formulates rotation schemes.

Crucially, the agent acts symmetrically on both text and UI surfaces. When the agent uses the `get_orientation_advice` tool, it doesn't only return string-based suggestions ("orient alongside the shortest axis"). It actively saves a payload of recommended Euler transformations to the application session state. The Streamlit render loop detects this signal and displays three distinct rotation visualisations embedded inside the chat thread dynamically.

## 3. Demo Walkthrough

1. **File Upload & Automated Inference:** A user uploads a raw `100034_binary.stl` part. The application immediately calculates the mesh laplacian operators and forwards them through DiffusionNet.
2. **Dashboard Review:** The dashboard updates detailing vertex counts, volume predictions, and constraint badges.
3. **Model Explainability (Saliency Map):** An engineer notices the "Pass/Fail" constraint says "FAIL". Using the "View Mode" toggle on the 3D Viewer, the engineer selects the "Pass/Fail Saliency" map. The application backpropagates gradients to the mesh vertices and colours the precise regions on the 3D plot responsible for the failure (via Plotly's vertex `intensity` mapping in Jet colorscale).
4. **Agent Interaction:** The engineer asks the chatbot: *"What is the best orientation to minimise support material?"*
5. **Agentic Visualisation:** The GPT-4o agent intercepts the query, formulates optimal transformations based on the part's shortest dimensions (Z build time), and replies with analytical reasoning. Simultaneously, the UI catches the agent's tool-call metadata and plots three embedded 3D models showing the user exactly how the new orientation looks on the print bed.
6. **PDF Report:** The engineer clicks "Download PDF Report", generating a full artifact describing the findings and design recommendations for later use.

## 4. Future Scope & Extensions

- **Auto-Fixing/Generative Optimisation:** Instead of merely highlighting failures using Saliency Maps, the system could employ iterative generative smoothing algorithms around high-saliency vertices, running them back through DiffusionNet until a passing threshold is achieved.
- **Material-specific Adjustments:** Expanding the inference model to accept categorical inputs for target materials (e.g. PLA vs. ABS vs. TPU). The agent would recommend different overhang constraints dynamically.
- **Process Agnosticism:** Adding models trained on different constraint sets (SLS powder removal paths or Metal PBF thermal residual stress fields) to make the Analyzer a unified DFM toolkit for all advanced manufacturing workflows.
