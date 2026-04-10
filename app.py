"""
Agentic DFM Analyzer — Streamlit Application
=============================================
Upload an STL file, view 3D mesh, get DFM predictions from DiffusionNet,
and chat with an AI agent for design-for-manufacturing advice.
"""

import os
import sys
import tempfile
import textwrap
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from inference.predict import run_inference, run_comparative_inference
from agent.dfm_agent import DFMAgent
from reports.pdf_generator import generate_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic DFM Analyzer",
    page_icon="🏭",
    layout="wide",
)

# ── Premium Theme CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

    /* ---- Global Styling ---- */
    html, body, .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #ffffff !important;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        color: #1e293b !important;
        letter-spacing: -0.02em;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    }
    .metric-label { 
        color: #64748b; 
        font-size: 0.75rem; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 0.05em; 
    }
    .metric-value { 
        color: #0f172a; 
        font-size: 1.5rem; 
        font-weight: 700; 
        margin-top: 4px;
        font-family: 'Outfit', sans-serif;
    }

    /* ---- Pass / Fail badges ---- */
    .pass-badge {
        background: #dcfce7; color: #166534; padding: 4px 12px;
        border-radius: 9999px; font-weight: 600; font-size: 0.75rem;
        border: 1px solid #bbf7d0;
    }
    .fail-badge {
        background: #fee2e2; color: #991b1b; padding: 4px 12px;
        border-radius: 9999px; font-weight: 600; font-size: 0.75rem;
        border: 1px solid #fecaca;
    }

    /* ---- Constraint list ---- */
    .constraint-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    }
    .constraint-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 0; border-bottom: 1px solid #f1f5f9;
    }
    .constraint-row:last-child { border-bottom: none; }
    .constraint-name { color: #334155; font-size: 0.9rem; font-weight: 500; }
    .constraint-conf { color: #94a3b8; font-size: 0.75rem; margin-left: 8px; font-family: 'JetBrains Mono', monospace; }

    /* ---- Comparative Table (Premium Light Theme) ---- */
    .comp-table-container {
        border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden;
        margin-top: 1.5rem; background: #ffffff; padding: 0px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .dfm-table {
        border-collapse: collapse; width: 100%; font-size: 0.82rem;
        background-color: #ffffff; color: #1e293b; font-family: 'JetBrains Mono', monospace;
    }
    .dfm-table th {
        padding: 14px 12px; text-align: center; font-weight: 700; 
        font-family: 'Outfit', sans-serif; text-transform: uppercase; letter-spacing: 0.02em;
        border-bottom: 2px solid #e2e8f0;
    }
    .dfm-table td {
        padding: 12px; text-align: center; border-bottom: 1px solid #f1f5f9;
        color: #475569;
    }
    .dfm-table tr:hover td { background: #f8fafc; }
    .dfm-table td:first-child { 
        text-align: left; color: #1e293b; font-family: 'Inter', sans-serif; 
        font-weight: 600; padding-left: 20px; background: #f8fafc;
    }
    
    .th-metric { background-color: #f1f5f9; color: #475569 !important; }
    .th-gat    { background-color: #dbeafe; color: #1e40af !important; }
    .th-pnpp   { background-color: #f1f5f9; color: #475569 !important; }
    .th-mcnn   { background-color: #fef3c7; color: #92400e !important; }
    .th-dn     { background-color: #ccfbf1; color: #115e59 !important; }

    .val-pass  { color: #166534; font-weight: 700; }
    .val-fail  { color: #991b1b; font-weight: 700; }
    .best-highlight { color: #0d9488 !important; background: #f0fdfa; font-weight: 800; }
    .efficient-highlight { color: #1d4ed8 !important; background: #eff6ff; font-weight: 800; }

    /* ---- Summary cards (Matching Light Theme) ---- */
    .summary-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1.25rem; text-align: center;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
    }
    .sc-label { font-size: 0.7rem; color: #64748b; margin-bottom: 6px; text-transform: uppercase; font-weight: 700; }
    .sc-model { font-size: 1.1rem; font-weight: 800; color: #ffffff; margin-bottom: 4px; font-family: 'Outfit', sans-serif; }
    .sc-value { font-size: 0.8rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; }
    
    /* Model specific accents */
    .sc-dn { border-color: #0d9488 !important; border-width: 2px; }
    .sc-dn .sc-model { color: #0d9488; }
    .sc-gat { border-color: #2563eb !important; border-width: 2px; }
    .sc-gat .sc-model { color: #2563eb; }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 48px; border-radius: 8px 8px 0 0; 
        background-color: #f1f5f9; border: 1px solid #e2e8f0;
        padding: 0 24px; font-weight: 600; color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important; border-bottom: 2px solid #2563eb !important;
        color: #2563eb !important;
    }

    /* ---- Chat ---- */
    .stChatMessage { border: 1px solid #e2e8f0; background: #ffffff; border-radius: 12px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────────────────
def _init_state():
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = DFMAgent()
        except Exception as e:
            st.session_state.agent = None
            st.session_state.agent_error = str(e)
    if "file_name" not in st.session_state:
        st.session_state.file_name = None

_init_state()


# ── Helpers ──────────────────────────────────────────────────────────────────

def render_3d_mesh(mesh, intensity=None):
    """Return a Plotly figure for the trimesh object."""
    vertices = mesh.vertices
    faces = mesh.faces

    mesh_args = dict(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        flatshading=True,
        lighting=dict(ambient=0.45, diffuse=0.65, specular=0.3, roughness=0.5),
        lightposition=dict(x=100, y=200, z=300),
    )

    if intensity is not None:
        mesh_args.update(dict(
            intensity=intensity,
            intensitymode='vertex',
            colorscale='Jet',
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(title=dict(text='Saliency', side='right'), thickness=14, len=0.45, x=1.01)
        ))
    else:
        mesh_args.update(dict(color='#3b82f6', opacity=0.75))

    fig = go.Figure(data=[go.Mesh3d(**mesh_args)])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=450,
    )
    return fig


def metric_card(label, value):
    return textwrap.dedent(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """)


def constraint_row(name, info):
    badge_cls = "pass-badge" if info["passed"] else "fail-badge"
    status = "PASS" if info["passed"] else "FAIL"
    return textwrap.dedent(f"""
    <div class="constraint-row">
        <span class="constraint-name">{name.replace('_',' ').title()}</span>
        <span>
            <span class="{badge_cls}">{status}</span>
            <span class="constraint-conf">{info['confidence']:.0f}%</span>
        </span>
    </div>
    """)



def render_comp_dashboard():
    """Static HTML table for architecture comparison based on notebook benchmarks."""
    models = ["GAT", "PointNet++", "MeshCNN", "DiffusionNet"]
    
    data = {
        "Volume R² ↑":      ["0.2057", "-0.0418", "0.4329", "0.4474"],
        "Volume MSE ↓":     ["3.2192", "4.2224", "2.2983", "2.2395"],
        "Area Accuracy":   ["77.7%", "77.7%", "77.7%", "77.7%"],
        "Contour Count":   ["60.7%", "60.7%", "60.7%", "60.7%"],
        "Contour Length":  ["83.1%", "83.1%", "83.1%", "83.1%"],
        "Overhang":        ["69.6%", "69.6%", "69.6%", "69.6%"],
        "Pass Fail ↑":     ["30.1%", "30.1%", "30.1%", "69.9%"],
        "Parameters":      ["45,766", "516,038", "~200k", "284,550"],
        "Epochs Trained":  ["21", "30", "60", "60"]
    }

    th_html = "<th class='th-metric' style='text-align:left; padding-left:20px;'>Metric</th>"
    for m in models:
        th_cls = "th-" + ("pnpp" if m == "PointNet++" else m.lower())
        th_html += f"<th class='{th_cls}'>{m}</th>"

    rows_html = ""
    for metric, values in data.items():
        row = f"<td>{metric}</td>"
        for i, val in enumerate(values):
            cell_cls = ""
            # Highlighting logic based on the photo
            if metric in ["Volume R² ↑", "Volume MSE ↓", "Pass Fail ↑"] and i == 3: # DiffusionNet wins
                cell_cls = "best-highlight"
            if metric == "Parameters" and i == 0: # GAT is most efficient
                cell_cls = "efficient-highlight"
            
            row += f"<td class='{cell_cls}'>{val}</td>"
        rows_html += f"<tr>{row}</tr>"

    html = f"""
    <div class="comp-table-container">
        <table class="dfm-table">
            <thead><tr>{th_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Benchmark Data (Static) ──────────────────────────────────────────────────
BENCHMARKS = [
    {"label": "Best R² (Volume)",  "model": "DiffusionNet", "value": "0.4474", "cls": "sc-dn"},
    {"label": "Lowest Volume MSE", "model": "DiffusionNet", "value": "2.2395", "cls": "sc-dn"},
    {"label": "Most Efficient",    "model": "GAT",          "value": "45,766 params", "cls": "sc-gat"},
    {"label": "Pass/Fail F1 Score","model": "DiffusionNet", "value": "69.9% Overall", "cls": "sc-dn"},
]

def render_summary_cards():
    cols = st.columns(len(BENCHMARKS))
    for i, b in enumerate(BENCHMARKS):
        with cols[i]:
            st.markdown(f"""
            <div class="summary-card {b['cls']}">
                <div class="sc-label">{b['label']}</div>
                <div class="sc-model">{b['model']}</div>
                <div class="sc-value" style="color:#64748b; font-weight:700;">{b['value']}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Sidebar: Control Panel ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 DFM Control Panel")
    st.markdown("Upload STL file for **Agentic DFM Analysis**.")
    st.divider()

    uploaded = st.file_uploader("Upload STL", type=["stl"], label_visibility="collapsed")

    if uploaded:
        if st.button("🚀 Run Analysis", width="stretch", type="primary"):
            with st.spinner("Processing Mesh..."):
                try:
                    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name

                    result = run_inference(tmp_path)
                    st.session_state.analysis = result
                    st.session_state.file_name = uploaded.name
                    st.session_state.messages = []
                    os.unlink(tmp_path)
                    st.success("Part Analyzed Successfully!")
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")

    if st.session_state.analysis:
        st.divider()
        st.markdown("### 📄 Reports")
        pdf_bytes = generate_report(st.session_state.analysis, st.session_state.file_name or "part")
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name=f"DFM_Report_{st.session_state.file_name}.pdf",
            mime="application/pdf",
            width="stretch",
        )


# ── Main Header ──────────────────────────────────────────────────────────────
st.markdown("<h1 style='margin-bottom: 0px;'>Agentic DFM Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 1.1rem; margin-bottom: 24px;'>Intelligence-driven manufacturability for additive manufacturing</p>", unsafe_allow_html=True)

if not st.session_state.analysis:
    st.info("👋 Welcome! Please upload an STL file in the sidebar to begin your manufacturing audit.")

# Handle Agent Commands (Mesh Repair)
if "agent_commands" in st.session_state and st.session_state.analysis:
    for cmd in st.session_state.agent_commands:
        if cmd == "repair_mesh":
            mesh = st.session_state.analysis["raw_mesh"]
            if not mesh.is_watertight:
                import trimesh
                st.toast("🔧 Agent is repairing mesh topology...")
                trimesh.repair.fill_holes(mesh)
                trimesh.repair.fix_normals(mesh)
                st.session_state.analysis["mesh_info"]["is_watertight"] = mesh.is_watertight
                if mesh.is_watertight:
                    st.session_state.analysis["mesh_info"]["volume_mm3"] = float(mesh.volume)
                st.success("✅ Part successfully converted to watertight manifold.")
    st.session_state.agent_commands = []


tab1, tab2, tab3 = st.tabs(["📊 DFM Audit", "💰 Manufacturing & Costs", "🧪 Architecture Comparison"])

# ═ TAB 1: DFM AUDIT ══════════════════════════════════════════════════════════
with tab1:
    if st.session_state.analysis:
        res = st.session_state.analysis
        mesh_info = res["mesh_info"]
        constraints = res["constraints"]

        col_mesh, col_data = st.columns([3, 2], gap="large")

        with col_mesh:
            st.markdown("### 🧩 3D Geometry Analysis")
            
            # Feature map selector
            map_opts = ["Geometric Neutral"]
            if "saliency_maps" in res:
                map_opts.extend([
                    f"D-Net Saliency: {k.replace('_', ' ').title()}" 
                    for k in res["saliency_maps"].keys() 
                    if k != "contour_count"
                ])
            
            view_mode = st.selectbox("Render Mode", map_opts, index=0)

            intensity = None
            if "Saliency" in view_mode:
                key = next(k for k in res["saliency_maps"].keys() if k.replace('_', ' ').title() in view_mode)
                intensity = res["saliency_maps"][key]

            fig = render_3d_mesh(res["raw_mesh"], intensity=intensity)
            st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})

        with col_data:
            st.markdown("### 🔢 Geometric Predictions")
            
            st.markdown(metric_card("Predicted Build Volume", f"{res['predicted_volume_mm3']:,.1f} mm³"), unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(metric_card("Vertices", f"{mesh_info['vertices']:,}"), unsafe_allow_html=True)
                sa = mesh_info.get('surface_area_mm2')
                st.markdown(metric_card("Surface Area", f"{sa:,.0f} mm²" if sa else "N/A"), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("Faces", f"{mesh_info['faces']:,}"), unsafe_allow_html=True)
                bb = mesh_info.get('bounding_box_mm', [0, 0, 0])
                st.markdown(metric_card("Max Extent", f"{max(bb):.1f} mm"), unsafe_allow_html=True)
            
            st.markdown("#### ⚖️ Constraint Compliance")
            badges_html = ""
            for name, info in constraints.items():
                badges_html += constraint_row(name, info)
            st.markdown(f'<div class="constraint-panel">{badges_html}</div>', unsafe_allow_html=True)

        # Chat
        st.divider()
        st.markdown("### 💬 DFM Agent Consultation")
        
        def render_agent_orientations(data):
            if not data: return
            st.markdown("#### Recommended Orientations (Generated by Agent)")
            cols = st.columns(3)
            import trimesh as _trimesh
            for i, o in enumerate(data):
                m = res["raw_mesh"].copy()
                rot = _trimesh.transformations.euler_matrix(np.radians(o['euler'][0]), np.radians(o['euler'][1]), np.radians(o['euler'][2]))
                m.apply_transform(rot)
                with cols[i]:
                    st.markdown(f"<p style='text-align:center; font-weight:600;'>{o['name']}</p>", unsafe_allow_html=True)
                    ofig = render_3d_mesh(m)
                    ofig.update_layout(height=180)
                    st.plotly_chart(ofig, width="stretch", config={'displayModeBar': False})

        # Scrollable chat history container — keeps input pinned below
        chat_container = st.container(height=420)
        with chat_container:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if m.get("orientations_data"): render_agent_orientations(m["orientations_data"])

        # Input box — always renders below the container
        prompt = st.chat_input("Ask about print orientation, material choice, or cost...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with chat_container:
                st.chat_message("user").markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Agent analyzing requirements..."):
                        reply = st.session_state.agent.chat(st.session_state.messages, res)
                        orientations = None
                        if 'recommended_orientations' in st.session_state:
                            orientations = st.session_state['recommended_orientations']
                            del st.session_state['recommended_orientations']

                        st.markdown(reply)
                        if orientations: render_agent_orientations(orientations)
                        st.session_state.messages.append({"role": "assistant", "content": reply, "orientations_data": orientations})

            if "agent_commands" in st.session_state and st.session_state.agent_commands:
                st.rerun()


# ═ TAB 2: MANUFACTURING & COSTS ══════════════════════════════════════════════
with tab2:
    st.markdown("### 💰 Production Estimator")
    if not st.session_state.analysis:
        st.info("Upload geometry to generate cost estimates.")
    else:
        v = st.session_state.analysis.get("predicted_volume_mm3", 0.0)
        
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                mat = st.selectbox("Build Material", ["PLA", "ABS", "PETG", "TPU", "NYLON", "PC"])
                m_data = {"PLA": (1.24, 1500), "ABS": (1.04, 1600), "PETG": (1.27, 1800), "TPU": (1.21, 2500), "NYLON": (1.14, 4000), "PC": (1.20, 3500)}
                density, base_inr = m_data[mat]
            with c2:
                inr_kg = st.number_input("Filament Cost (₹/kg)", value=float(base_inr))
            with c3:
                margin = st.slider("Service Margin (%)", 0, 100, 20)
            
            weight = (v / 1000.0) * density
            raw_cost = (weight / 1000.0) * inr_kg
            total_cost = raw_cost * (1 + margin/100)
            
            st.divider()
            mc1, mc2, mc3 = st.columns(3)
            with mc1: st.metric("Predicted Weight", f"{weight:.1f} g")
            with mc2: st.metric("Raw Material Cost", f"₹{raw_cost:.2f}")
            with mc3: st.metric("Total Estimate (incl. Margin)", f"₹{total_cost:.2f}", delta=f"₹{total_cost-raw_cost:.1f} margin")

        st.info("💡 Pro-Tip: Ask the DFM Agent to generate a **Full Manufacturing Report** including slicing parameters for this specific part.")


# ═ TAB 3: ARCHITECTURE COMPARISON ════════ (STATIC BENCHMARKS) ═══════════════
with tab3:
    st.markdown("### 🧪 Architecture Benchmark Highlights")
    st.markdown("Comparison of **Geometric Deep Learning** models trained on the DFM dataset.")
    
    render_comp_dashboard()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("#### 📈 Global Architecture Performance Sumary")
    render_summary_cards()