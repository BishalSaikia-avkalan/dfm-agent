"""
Agentic DFM Analyzer — Streamlit Application
=============================================
Upload an STL file, view 3D mesh, get DFM predictions from DiffusionNet,
and chat with an AI agent for design-for-manufacturing advice.
"""

import os
import sys
import tempfile
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from inference.predict import run_inference
from agent.dfm_agent import DFMAgent
from reports.pdf_generator import generate_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic DFM Analyzer",
    page_icon="🏭",
    layout="wide",
)

# ── Custom CSS for a polished look ───────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #0e1117 100%);
        border: 1px solid #2980b9;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .metric-label { color: #8e99a4; font-size: 0.85rem; }
    .metric-value { color: #e0e0e0; font-size: 1.3rem; font-weight: 600; }
    .pass-badge {
        background: #27ae60; color: white; padding: 2px 10px;
        border-radius: 8px; font-weight: 600; font-size: 0.85rem;
    }
    .fail-badge {
        background: #e74c3c; color: white; padding: 2px 10px;
        border-radius: 8px; font-weight: 600; font-size: 0.85rem;
    }
    .stChatMessage { border-radius: 12px; }
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
        lighting=dict(ambient=0.4, diffuse=0.6, specular=0.3, roughness=0.5),
        lightposition=dict(x=100, y=200, z=300),
    )
    
    if intensity is not None:
        mesh_args.update(dict(
            intensity=intensity,
            intensitymode='vertex',
            colorscale='Jet',
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(title=dict(text='GradCAM', side='right'), thickness=14, len=0.45, x=1.01)
        ))
    else:
        mesh_args.update(dict(color='cyan', opacity=0.60))

    fig = go.Figure(data=[go.Mesh3d(**mesh_args)])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#0e1117",
            aspectmode="data",
        ),
        paper_bgcolor="#0e1117",
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
    )
    return fig


import textwrap

def metric_card(label, value):
    return textwrap.dedent(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """)


def constraint_badge(name, info):
    badge_cls = "pass-badge" if info["passed"] else "fail-badge"
    status = "PASS" if info["passed"] else "FAIL"
    return textwrap.dedent(f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                padding:6px 0; border-bottom:1px solid #1a1f2e;">
        <span style="color:#e0e0e0;">{name.replace('_',' ').title()}</span>
        <span>
            <span class="{badge_cls}">{status}</span>
            <span style="color:#8e99a4; font-size:0.8rem; margin-left:8px;">
                {info['confidence']:.0f}%
            </span>
        </span>
    </div>
    """)


# ── Sidebar: Upload & Analyse ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 DFM Analyzer")
    st.markdown("Upload an **STL** file to analyse its manufacturability for FDM 3D printing.")
    st.divider()

    uploaded = st.file_uploader("Upload STL file", type=["stl"], key="stl_upload")

    if uploaded is not None:
        if st.button("🔍 Analyse Part", use_container_width=True, type="primary"):
            with st.spinner("Processing mesh & running DiffusionNet…  \n*(this may take ~1 min for Laplacian computation)*"):
                try:
                    # Save to temp file for trimesh
                    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name

                    result = run_inference(tmp_path)
                    st.session_state.analysis = result
                    st.session_state.file_name = uploaded.name
                    st.session_state.messages = []  # reset chat for new file
                    os.unlink(tmp_path)
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    # Report download
    if st.session_state.analysis is not None:
        st.divider()
        pdf_bytes = generate_report(st.session_state.analysis, st.session_state.file_name or "part")
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name="dfm_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ── Main area ────────────────────────────────────────────────────────────────
st.markdown("# 🏭 Agentic DFM Analyzer")
st.markdown("*AI-powered Design for Manufacturability analysis for FDM 3D printing*")

if st.session_state.analysis is None:
    st.info("👈 Upload an STL file in the sidebar and click **Analyse Part** to get started.")
else:
    result = st.session_state.analysis
    mesh_info = result["mesh_info"]
    constraints = result["constraints"]

    # ── Two-column layout: 3D viewer + metrics ───────────────────────────
    col_3d, col_metrics = st.columns([3, 2], gap="large")

    with col_3d:
        st.markdown("### 🔬 3D Mesh Viewer")
        
        view_opts = ["Raw Mesh"]
        if "saliency_maps" in result:
            view_opts.extend([f"Saliency: {k.replace('_', ' ').title()}" for k in result["saliency_maps"].keys()])
            
        view_mode = st.selectbox("View Mode", view_opts, index=0)
        
        intensity = None
        if view_mode != "Raw Mesh" and "saliency_maps" in result:
            # Reconstruct original key
            raw_key = next(k for k in result["saliency_maps"].keys() if f"Saliency: {k.replace('_', ' ').title()}" == view_mode)
            intensity = result["saliency_maps"][raw_key]
            
        fig = render_3d_mesh(result["raw_mesh"], intensity=intensity)
        st.plotly_chart(fig, width="stretch")

    with col_metrics:
        st.markdown("### 📊 DFM Metrics")

        # Volume prediction
        st.markdown(metric_card(
            "Predicted Volume",
            f"{result['predicted_volume_mm3']:,.1f} mm³"
        ), unsafe_allow_html=True)

        # Mesh stats row
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(metric_card("Vertices", f"{mesh_info['vertices']:,}"), unsafe_allow_html=True)
            sa = mesh_info.get('surface_area_mm2')
            st.markdown(metric_card("Surface Area", f"{sa:,.1f} mm²" if sa else "N/A"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Faces", f"{mesh_info['faces']:,}"), unsafe_allow_html=True)
            bb = mesh_info.get('bounding_box_mm', [0, 0, 0])
            st.markdown(metric_card("Bounding Box", f"{bb[0]:.1f}×{bb[1]:.1f}×{bb[2]:.1f}"), unsafe_allow_html=True)

        st.markdown(metric_card(
            "Watertight",
            "✅ Yes" if mesh_info.get("is_watertight") else "⚠️ No"
        ), unsafe_allow_html=True)

        # Constraint checks
        st.markdown("#### Constraint Checks")
        badges_html = ""
        for name, info in constraints.items():
            badges_html += constraint_badge(name, info)
        st.markdown(f'<div style="background:#0e1117; border:1px solid #2980b9; '
                    f'border-radius:12px; padding:10px 14px;">{badges_html}</div>',
                    unsafe_allow_html=True)

    # ── Chat section ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 💬 DFM Agent Chat")
    st.markdown("*Ask questions about the part's manufacturability, orientation, material selection, or design improvements.*")

    def render_orientations_UI(orientations_data):
        if not orientations_data:
            return
        st.markdown("#### 🔄 Recommended Optimised Orientations")
        cols = st.columns(3)
        import copy
        import trimesh
        import plotly.graph_objects as go
        
        for i, o_data in enumerate(orientations_data):
            r_mesh = result["raw_mesh"].copy()
            euler = o_data['euler']
            
            # Apply rotation
            rot_matrix = trimesh.transformations.euler_matrix(
                np.radians(euler[0]), np.radians(euler[1]), np.radians(euler[2])
            )
            r_mesh.apply_transform(rot_matrix)
            
            with cols[i]:
                st.markdown(f"**{o_data['name']}**")
                o_fig = render_3d_mesh(r_mesh)
                o_fig.update_layout(height=280)
                st.plotly_chart(o_fig, width="stretch")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("orientations_data"):
                render_orientations_UI(msg["orientations_data"])

    # Chat input
    if prompt := st.chat_input("e.g. Will this part warp during printing?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        agent = st.session_state.agent
        if agent is None:
            reply = f"⚠️ Agent unavailable: {st.session_state.get('agent_error', 'Unknown error')}"
        else:
            with st.spinner("Thinking…"):
                reply = agent.chat(st.session_state.messages, st.session_state.analysis)

        # Capture recommended orientations if the agent generated them
        orientations_data = None
        if 'recommended_orientations' in st.session_state:
            orientations_data = st.session_state['recommended_orientations']
            del st.session_state['recommended_orientations']

        st.session_state.messages.append({
            "role": "assistant", 
            "content": reply, 
            "orientations_data": orientations_data
        })
        
        with st.chat_message("assistant"):
            st.markdown(reply)
            if orientations_data:
                render_orientations_UI(orientations_data)