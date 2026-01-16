import json
import os
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

import base64

logo_path = "/Users/stevenmauricesmith/Desktop/cloudburst/images/ChatGPT Image Jan 16, 2026, 05_07_12 PM.png"
with open(logo_path, "rb") as f:
    data = base64.b64encode(f.read()).decode("utf-8")
# ----------------------------
# LLM
# ----------------------------
llm = ChatOllama(model="qwen3:1.7b")  # change as needed


# ----------------------------
# Optional: structured planner output
# ----------------------------
class CostLineItem(BaseModel):
    item: str = Field(..., description="Short name of the cost item")
    min_usd: float = Field(..., description="Lower bound estimate in USD")
    likely_usd: float = Field(..., description="Most likely estimate in USD")
    max_usd: float = Field(..., description="Upper bound estimate in USD")


class PlannerOutput(BaseModel):
    executive_summary: str
    phases: Dict[str, str] = Field(
        ..., description="Phase name -> actions (0-6, 6-18, 18-36 months)"
    )
    cost_estimates: list[CostLineItem]
    assumptions: list[str]
    next_steps: list[str]


# ----------------------------
# Helpers: dataset loading
# ----------------------------
def safe_json_dumps(obj: Any, max_chars: int = 160_000) -> str:
    txt = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    if len(txt) > max_chars:
        return txt[:max_chars] + "\n\n[TRUNCATED]"
    return txt


def load_dataset_from_upload(uploaded_file) -> Dict[str, Any]:
    """
    Accepts file from st.file_uploader and returns a compact dict.
    Supports: CSV / JSON / JSONL / TXT.
    """
    if uploaded_file is None:
        return {"note": "No dataset file uploaded."}

    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower().strip(".")

    if ext == "csv":
        df = pd.read_csv(uploaded_file)

        summary = {
            "file": filename,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": list(map(str, df.columns)),
        }

        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            desc = numeric.describe().to_dict()
            summary["numeric_describe"] = json.loads(json.dumps(desc, default=str))

        nonnum = df.select_dtypes(exclude="number")
        if not nonnum.empty:
            cat_info: Dict[str, Any] = {}
            for col in list(nonnum.columns)[:10]:
                vc = nonnum[col].astype(str).value_counts(dropna=False).head(10)
                cat_info[str(col)] = vc.to_dict()
            summary["categorical_top_values"] = cat_info

        summary["sample_rows_head_20"] = df.head(20).to_dict(orient="records")

        return {"csv_summary": summary}

    if ext == "json":
        data = json.load(uploaded_file)
        return data if isinstance(data, dict) else {"data": data}

    if ext == "jsonl":
        rows = []
        for i, line in enumerate(uploaded_file):
            if i >= 2000:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_raw": line})
        return {"jsonl_rows_loaded": len(rows), "rows": rows}

    # fallback: text
    content = uploaded_file.read().decode("utf-8", errors="replace")
    if len(content) > 160_000:
        content = content[:160_000] + "\n\n[TRUNCATED]"
    return {"text_file": filename, "content": content}


# ----------------------------
# Agents
# ----------------------------
def climate_specialist(concern: str, dataset: Dict[str, Any], location: str) -> str:
    messages = [
        {
            "role": "system",
            "content": f"""
You are a Climate Specialist working on flood mitigation via cloudburst hubs (green infrastructure).
Context: We are promoting cloudburst hubs in Corona and Kissena in Queens for flood mitigation (DEP proposal).
Cloudbursts are short, intense rainfall causing flash/inland flooding, especially near Flushing.
Location: {location}

TASK:
1) Use the dataset to infer future trends and risk drivers.
2) Address the user concern directly.
3) Recommend mitigation/adaptation strategies for communities.
4) Write a concise report that a Climate Scientist can audit.

OUTPUT FORMAT (use these headings):
- Key findings
- Risk characterization (what/where/when)
- Mitigation & adaptation options (ranked, with pros/cons)
- Data quality notes (assumptions, missing data, uncertainties)
""".strip(),
        },
        {
            "role": "user",
            "content": f"""CONCERN:
{concern}

DATASET (JSON-like):
{safe_json_dumps(dataset)}
""",
        },
    ]
    return llm.invoke(messages).content


def climate_scientist(concern: str, dataset: Dict[str, Any], specialist_report: str, location: str) -> str:
    messages = [
        {
            "role": "system",
            "content": f"""
You are a Climate Scientist. Your job is to validate and improve the Climate Specialist report.
Location: {location}

TASK:
- Critique assumptions and methods
- Add scientific rigor (extremes, uncertainty, return periods if applicable)
- Identify data gaps / what should be measured
- Provide confidence level (High/Medium/Low) + justification
- Give recommendations suitable for planning

OUTPUT FORMAT (use these headings):
- Scientific assessment
- Confidence level (High/Medium/Low) + justification
- Key uncertainties & missing data
- Scientist recommendations to planning team
""".strip(),
        },
        {
            "role": "user",
            "content": f"""CONCERN:
{concern}

DATASET:
{safe_json_dumps(dataset)}

CLIMATE SPECIALIST REPORT:
{specialist_report}
""",
        },
    ]
    return llm.invoke(messages).content


def planner_agent(concern: str, dataset: Dict[str, Any], scientist_report: str, location: str):
    system_prompt = f"""
You are a Planner Agent for climate adaptation infrastructure projects (NYC-style public works context).
Location: {location}

TASK:
Convert the scientific assessment into an actionable plan for what the user/community should do concerning the dataset.
Include:
- Phased plan: 0–6 months, 6–18 months, 18–36 months
- Stakeholders
- Rough cost ranges (CapEx + OpEx) MIN/LIKELY/MAX in USD
- Dependencies, permitting, engagement, risks
- Concrete next steps

Return JSON matching this schema:
PlannerOutput = {{
  "executive_summary": str,
  "phases": {{"0-6 months": str, "6-18 months": str, "18-36 months": str}},
  "cost_estimates": [{{"item": str, "min_usd": float, "likely_usd": float, "max_usd": float}}, ...],
  "assumptions": [str, ...],
  "next_steps": [str, ...]
}}
""".strip()

    # Try structured output first
    try:
        structured_llm = llm.with_structured_output(PlannerOutput)
        result: PlannerOutput = structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"""CONCERN:
{concern}

DATASET:
{safe_json_dumps(dataset)}

CLIMATE SCIENTIST REPORT:
{scientist_report}
""",
                },
            ]
        )
        return result.model_dump(), None

    except Exception as e:
        # Fallback plain text
        fallback = llm.invoke(
            [
                {"role": "system", "content": system_prompt + "\n\nIf JSON fails, return a well-structured markdown plan."},
                {
                    "role": "user",
                    "content": f"""CONCERN:
{concern}

DATASET:
{safe_json_dumps(dataset)}

CLIMATE SCIENTIST REPORT:
{scientist_report}

NOTE: structured output error: {repr(e)}
""",
                },
            ]
        ).content
        return None, fallback


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Cloudburst Advisor", layout="wide")

st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{data}" width="500">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Cloudburst")
st.write("Upload your dataset + describe your concern. Then the climate workflow will generate recommendations and budgeting.")


with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama model", value="qwen3:1.7b")
    location = st.text_input("Location context", value="Queens, NY")
    show_intermediate = st.checkbox("Show intermediate agent reports", value=True)
    run_button = st.button("Run Workflow 🚀")

# update model dynamically if user changes
llm = ChatOllama(model=model_name)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Your concern")
    concern = st.text_area(
        "Describe your climate/flood concern and what you want to understand.",
        height=220,
        placeholder="Example: We have frequent street flooding near XYZ. Based on this rainfall/drainage dataset, what should we prioritize and how much might it cost?"
    )

with col2:
    st.subheader("2) Upload your dataset")
    uploaded = st.file_uploader(
        "Upload CSV, JSON, JSONL, or TXT",
        type=["csv", "json", "jsonl", "txt"]
    )

    if uploaded is not None:
        st.success(f"Uploaded: {uploaded.name}")

dataset = load_dataset_from_upload(uploaded)

if run_button:
    if not concern.strip():
        st.error("Please enter a concern message before running.")
        st.stop()

    with st.status("Running Climate Specialist → Scientist → Planner workflow...", expanded=True) as status:
        st.write("🧠 Climate Specialist analyzing dataset + concern...")
        specialist_report = climate_specialist(concern, dataset, location)
        st.write("✅ Climate Specialist complete.")

        st.write("🔬 Climate Scientist validating and improving report...")
        scientist_report = climate_scientist(concern, dataset, specialist_report, location)
        st.write("✅ Climate Scientist complete.")

        st.write("📋 Planner generating phased plan + budget...")
        planner_structured, planner_fallback = planner_agent(concern, dataset, scientist_report, location)
        st.write("✅ Planner complete.")

        status.update(label="Workflow complete ✅", state="complete", expanded=False)

    st.divider()
    st.header("✅ Final Recommendations + Budget")

    if planner_structured:
        st.subheader("Executive Summary")
        st.write(planner_structured["executive_summary"])

        st.subheader("Phased Plan")
        phases = planner_structured["phases"]
        for phase_name, phase_text in phases.items():
            with st.expander(phase_name, expanded=True):
                st.write(phase_text)

        st.subheader("Cost Estimates (USD)")
        cost_rows = []
        for item in planner_structured["cost_estimates"]:
            cost_rows.append({
                "Item": item["item"],
                "Min (USD)": item["min_usd"],
                "Likely (USD)": item["likely_usd"],
                "Max (USD)": item["max_usd"],
            })
        st.dataframe(cost_rows, use_container_width=True)

        st.subheader("Assumptions")
        st.write("\n".join([f"- {a}" for a in planner_structured["assumptions"]]))

        st.subheader("Next Steps")
        st.write("\n".join([f"- {n}" for n in planner_structured["next_steps"]]))

        # Download JSON
        st.download_button(
            "Download plan as JSON",
            data=json.dumps(planner_structured, indent=2),
            file_name="cloudburst_plan.json",
            mime="application/json"
        )

    else:
        st.markdown(planner_fallback)

    if show_intermediate:
        st.divider()
        st.header("Intermediate Agent Reports")

        with st.expander("Climate Specialist Report", expanded=False):
            st.markdown(specialist_report)

        with st.expander("Climate Scientist Report", expanded=False):
            st.markdown(scientist_report)

    st.divider()
    st.subheader("Dataset Summary Sent to Agents")
    st.code(safe_json_dumps(dataset), language="json")