"""
DFM Agent: GPT-4o with function calling for DFM analysis Q&A.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a DFM (Design for Manufacturability) expert specialising in FDM 3D printing.

You help engineers evaluate whether a 3D part is suitable for FDM printing and suggest improvements.

When the user uploads a part, you have access to a DFM analysis tool that returns:
• Predicted build volume
• Five binary constraint checks: area, contour_count, contour_length, overhang, pass_fail
• Raw mesh statistics (vertices, faces, surface area, bounding box, watertight status, Euler number)

Use the tool results to provide actionable engineering feedback on:
1. **Orientation Optimisation** – which build orientation reduces supports and overhangs
2. **Volume Minimisation** – how to reduce material usage and build time
3. **Defect Prediction** – risk of warping, delamination, or failed overhangs
4. **Material & Slicer Advice** – recommended materials, layer heights, infill
5. **Design Iteration** – concrete geometry changes to improve manufacturability

Always be transparent about model confidence levels. If a constraint check has low confidence (< 65%), say so.
Keep answers concise, practical, and organised with bullet points.
When no analysis has been run yet, ask the user to upload an STL file first.
"""

# Tool definitions for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_dfm_analysis",
            "description": (
                "Retrieve the DFM analysis results for the currently uploaded STL file. "
                "Returns predicted volume, constraint pass/fail for 5 metrics "
                "(area, contour_count, contour_length, overhang, pass_fail), "
                "confidence scores, and raw mesh properties."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_design_suggestions",
            "description": (
                "Generate specific design improvement suggestions based on the "
                "DFM analysis results. Call this when the user asks how to improve "
                "the part or fix failed constraints."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_orientation_advice",
            "description": (
                "Provide build orientation recommendations to minimise support "
                "material and reduce overhang issues. Uses mesh bounding box and "
                "overhang analysis from the DFM results."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def _handle_tool_call(tool_name: str, analysis_result: dict | None) -> str:
    """Execute a tool call and return the result as a JSON string."""
    if analysis_result is None:
        return json.dumps({
            "error": "No analysis available. Please upload an STL file first."
        })

    if tool_name == "get_dfm_analysis":
        # Return the full analysis (excluding raw_mesh and saliency_maps to save tokens)
        result = {k: v for k, v in analysis_result.items() if k not in ("raw_mesh", "saliency_maps")}
        return json.dumps(result, default=str)

    elif tool_name == "get_design_suggestions":
        constraints = analysis_result.get("constraints", {})
        suggestions = []
        for name, info in constraints.items():
            if not info["passed"]:
                if name == "overhang":
                    suggestions.append(
                        "• **Overhang issue**: Add fillets or chamfers to overhanging surfaces. "
                        "Consider splitting the part or adding self-supporting angles (≤45°)."
                    )
                elif name == "area":
                    suggestions.append(
                        "• **Surface area concern**: Large flat surfaces may warp. "
                        "Add ribs or reduce large planar faces."
                    )
                elif name == "contour_count":
                    suggestions.append(
                        "• **Contour complexity**: Too many contours increase print time. "
                        "Simplify geometry or merge small features."
                    )
                elif name == "contour_length":
                    suggestions.append(
                        "• **Contour length concern**: Long perimeters may cause quality issues. "
                        "Consider breaking into sub-assemblies."
                    )
                elif name == "pass_fail":
                    suggestions.append(
                        "• **Overall DFM check failed**: Review all constraints above. "
                        "The part may need significant redesign for reliable FDM printing."
                    )
        if not suggestions:
            suggestions.append("✅ All constraints passed. Part looks suitable for FDM printing.")
        return json.dumps({"suggestions": suggestions})

    elif tool_name == "get_orientation_advice":
        mesh_info = analysis_result.get("mesh_info", {})
        bb = mesh_info.get("bounding_box_mm", [0, 0, 0])
        constraints = analysis_result.get("constraints", {})
        overhang = constraints.get("overhang", {})

        advice = []
        # Sort bbox dimensions
        dims = sorted(enumerate(bb), key=lambda x: x[1])
        advice.append(
            f"• **Bounding box**: {bb[0]:.1f} × {bb[1]:.1f} × {bb[2]:.1f} mm"
        )
        advice.append(
            "• **Recommended**: Orient the part so the **shortest dimension** is the "
            f"build height (Z-axis) — currently {dims[0][1]:.1f} mm. "
            "This minimises layer count and print time."
        )
        if overhang and not overhang.get("passed", True):
            advice.append(
                "• **Overhang warning**: Consider rotating the part to reduce unsupported "
                "overhangs. FDM supports angles up to ~45° from vertical without supports."
            )
        else:
            advice.append("• Overhang check passed — current orientation should work well.")

        advice.append(
            "• **Bed adhesion**: Orient the largest flat surface on the build plate "
            "for maximum adhesion."
        )
        
        # Calculate Top-3 Rotations (Euler angles in degrees)
        # 1. Shortest to Z (Minimize Height)
        # 2. Longest to Z (Minimize Footprint)
        # 3. Y to Z (Alternative)
        rotations = []
        for rank, title in enumerate(["Minimize Height (Fastest)", "Alternative Orientation", "Minimize Footprint (Tallest)"]):
            axis_idx = dims[rank][0]
            if axis_idx == 2:
                euler = [0, 0, 0]
            elif axis_idx == 0:
                euler = [0, 90, 0]
            else:
                euler = [90, 0, 0]
            rotations.append({"name": title, "euler": euler})

        import streamlit as st
        try:
            st.session_state['recommended_orientations'] = rotations
            advice.append("• **Visualizations generated**: I have sent the top-3 recommended orientations to the 3D viewer.")
        except Exception:
            pass

        return json.dumps({"orientation_advice": advice})

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


class DFMAgent:
    """OpenAI-powered DFM conversational agent with tool calling."""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment. Check .env file.")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"

    def chat(self, messages: list[dict], analysis_result: dict | None) -> str:
        """
        Send a multi-turn conversation to GPT-4o with tool support.

        Parameters
        ----------
        messages : list of {"role": ..., "content": ...}
        analysis_result : dict or None
            Output from inference.predict.run_inference(), or None if no file uploaded.

        Returns
        -------
        str : assistant's reply
        """
        sys_prompt = SYSTEM_PROMPT
        if analysis_result is not None:
            sys_prompt += "\n\n[SYSTEM NOTE: The user has successfully uploaded a part and its analysis is available! You MUST use your tools (like get_orientation_advice, get_dfm_analysis, etc.) to fetch information about it. Do NOT ask them to upload it again.]"

        # Clean messages of any non-standard keys added by Streamlit frontend
        clean_messages = []
        for m in messages:
            clean_msg = {"role": m["role"], "content": m["content"]}
            if "name" in m:
                clean_msg["name"] = m["name"]
            if "tool_calls" in m:
                clean_msg["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m:
                clean_msg["tool_call_id"] = m["tool_call_id"]
            clean_messages.append(clean_msg)
            
        full_messages = [{"role": "system", "content": sys_prompt}] + clean_messages

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.4,
                max_tokens=1024,
            )
        except Exception as e:
            return f"⚠️ Error communicating with OpenAI: {e}"

        msg = response.choices[0].message

        # Handle tool calls
        if msg.tool_calls:
            # Append the assistant message with tool_calls
            full_messages.append(msg)

            for tool_call in msg.tool_calls:
                tool_result = _handle_tool_call(
                    tool_call.function.name, analysis_result
                )
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

            # Second call to get the final answer
            try:
                response2 = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    temperature=0.4,
                    max_tokens=1024,
                )
                return response2.choices[0].message.content or ""
            except Exception as e:
                return f"⚠️ Error in follow-up call: {e}"

        return msg.content or ""
