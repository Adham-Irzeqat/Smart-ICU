# icu_ai_cloud.py

import json
import re
from datetime import datetime, timezone

from .models import PatientSummary
import streamlit as st

# LangChain 1.0.2 imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI



# Database and data handling
from pymongo import MongoClient
import pandas as pd
import numpy as np

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Token counting
import tiktoken

import os
from langsmith import Client

# Set up LangSmith tracing and environment variables
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "ICU AI Cloud")

# Optional: initialize a client if you want to log custom runs
client = Client(api_key=st.secrets["LANGCHAIN_API_KEY"])



# =========================
# --- MongoDB Setup ---
# =========================
@st.cache_resource
def get_mongodb_client():
    """Initialize MongoDB client"""
    mongo_uri = st.secrets.get("MONGO_URI", "mongodb://localhost:27017/")
    return MongoClient(mongo_uri)

def get_database(): 
    """Get ICU database"""
    return get_mongodb_client()['icu_database']

# =========================
# --- Data Storage ---
# =========================
def store_patient_data(patient: PatientSummary, patient_data: dict):
    """Store patient snapshot"""
    db = get_database()
    collection = db['patient_data']
    record = {
        "patient_id": patient.patient_id,
        "timestamp": datetime.now(timezone.utc),
        "type": "patient_data",
        "data": patient_data
    }
    return str(collection.insert_one(record).inserted_id)

def store_doctor_recommendation(patient_id: str, recommendation: dict, patient_data: dict):
    """Store doctor's recommendation"""
    db = get_database()
    collection = db['patient_data']
    record = {
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc),
        "type": "doctor_rec",
        "recommendation": recommendation,
        "data": patient_data
    }
    return str(collection.insert_one(record).inserted_id)

def store_ai_recommendation(patient_id: str, recommendation: dict, patient_data: dict):
    """Store AI recommendation"""
    db = get_database()
    collection = db['patient_data']
    record = {
        "patient_id": patient_id,
        "timestamp": datetime.now(timezone.utc),
        "type": "ai_rec",
        "recommendation": recommendation,
        "data": patient_data
    }
    return str(collection.insert_one(record).inserted_id)

# =========================
# --- Data Retrieval ---
# =========================
def get_patient_history(patient_id: str, limit: int = None):
    """Retrieve all patient data snapshots"""
    db = get_database()
    collection = db['patient_data']
    cursor = collection.find({"patient_id": patient_id}).sort("timestamp", -1)
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)

def get_recommendation_history(patient_id: str, limit: int = None):
    """Retrieve all recommendations (doctor + AI)"""
    db = get_database()
    collection = db['patient_data']
    cursor = collection.find({
        "patient_id": patient_id,
        "type": {"$in": ["doctor_rec", "ai_rec"]}
    }).sort("timestamp", -1)
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)

def search_patients(query_filters: dict):
    """Search patients with optional date range"""
    db = get_database()
    collection = db['patient_data']
    mongo_query = {}
    if "patient_id" in query_filters:
        mongo_query["patient_id"] = query_filters["patient_id"]
    if "date_from" in query_filters:
        mongo_query["timestamp"] = {"$gte": query_filters["date_from"]}
    if "date_to" in query_filters:
        if "timestamp" not in mongo_query:
            mongo_query["timestamp"] = {}
        mongo_query["timestamp"]["$lte"] = query_filters["date_to"]
    results = collection.find(mongo_query).sort("timestamp", -1)
    return list(results)

# =========================
# --- Comparison & Trends ---
# =========================
def compare_patient_states(patient_id: str, timestamp1: datetime, timestamp2: datetime):
    """Compare two patient states"""
    db = get_database()
    collection = db['patient_data']
    state1 = collection.find_one(
        {"patient_id": patient_id, "timestamp": {"$lte": timestamp1}},
        sort=[("timestamp", -1)]
    )
    state2 = collection.find_one(
        {"patient_id": patient_id, "timestamp": {"$lte": timestamp2}},
        sort=[("timestamp", -1)]
    )
    if not state1 or not state2:
        return None
    return {
        "state1": state1,
        "state2": state2,
        "comparison": _generate_comparison(state1["data"], state2["data"])
    }

def _generate_comparison(data1: dict, data2: dict):
    """Compare vitals, labs, medications"""
    comparison = {"vitals_changes": [], "labs_changes": [], "meds_changes": [], "imaging_changes": []}

    # Vitals
    if data1.get("vitals") and data2.get("vitals"):
        vitals1 = {(v.get("label") or v.get("code") or "Unknown"): v.get("value") for v in data1["vitals"][-5:]}
        vitals2 = {(v.get("label") or v.get("code") or "Unknown"): v.get("value") for v in data2["vitals"][-5:]}
        for name in set(vitals1.keys()) | set(vitals2.keys()):
            v1, v2 = vitals1.get(name), vitals2.get(name)
            if v1 is not None and v2 is not None and v1 != v2:
                comparison["vitals_changes"].append({
                    "name": name, "previous": v1, "current": v2,
                    "change": v2 - v1 if isinstance(v1, (int,float)) and isinstance(v2,(int,float)) else None
                })

    # Labs
    if data1.get("labs") and data2.get("labs"):
        labs1 = {l.get("name", "Unknown"): l.get("value") for l in data1["labs"][-10:]}
        labs2 = {l.get("name", "Unknown"): l.get("value") for l in data2["labs"][-10:]}
        for name in set(labs1.keys()) | set(labs2.keys()):
            l1, l2 = labs1.get(name), labs2.get(name)
            if l1 is not None and l2 is not None and l1 != l2:
                comparison["labs_changes"].append({
                    "name": name, "previous": l1, "current": l2,
                    "change": l2 - l1 if isinstance(l1, (int,float)) and isinstance(l2,(int,float)) else None
                })

    # Medications
    meds1_set = set(m.get("name", "Unknown") for m in data1.get("meds", []))
    meds2_set = set(m.get("name", "Unknown") for m in data2.get("meds", []))
    comparison["meds_changes"] = {"added": list(meds2_set - meds1_set),
                                  "removed": list(meds1_set - meds2_set),
                                  "continued": list(meds1_set & meds2_set)}
    return comparison

def generate_trend_report(patient_id: str, days: int = 7):
    """Trends for vitals, labs, meds, recommendations"""
    db = get_database()
    cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days)
    patient_data = list(db['patient_data'].find({"patient_id": patient_id,"type": "patient_data", "timestamp":{"$gte":cutoff}}).sort("timestamp",1))
    recommendations = list(db['patient_data'].find({"patient_id": patient_id, "type":{"$in":["doctor_rec","ai_rec"]}, "timestamp":{"$gte":cutoff}}).sort("timestamp",1))
    if not patient_data:
        return None
    return {
        "patient_id": patient_id,
        "period_days": days,
        "total_updates": len(patient_data),
        "total_recommendations": len(recommendations),
        "vitals_trend": _extract_vitals_trend(patient_data),
        "labs_trend": _extract_labs_trend(patient_data),
        "meds_trend": _extract_meds_trend(patient_data),
        "recommendation_patterns": _analyze_recommendation_patterns(recommendations),
        "raw_data": patient_data,
        "raw_recommendations": recommendations
    }

def _extract_vitals_trend(data):
    vitals_by_name = {}
    for record in data:
        ts = record["timestamp"]
        for vital in record["data"].get("vitals", []):
            name = vital.get("label") or vital.get("code") or vital.get("name","Unknown")
            vitals_by_name.setdefault(name, []).append({"timestamp": ts, "value": vital["value"], "unit": vital.get("unit","")})
    return vitals_by_name

def _extract_labs_trend(data):
    labs_by_name = {}
    for record in data:
        ts = record["timestamp"]
        for lab in record["data"].get("labs", []):
            name = lab.get("name","Unknown")
            labs_by_name.setdefault(name, []).append({"timestamp": ts, "value": lab["value"], "unit": lab.get("unit","")})
    return labs_by_name

def _extract_meds_trend(data):
    meds_timeline = []
    for record in data:
        ts = record["timestamp"]
        meds = [m.get("name","Unknown") for m in record["data"].get("meds",[])]
        meds_timeline.append({"timestamp": ts, "medications": meds, "count": len(meds)})
    return meds_timeline

def _analyze_recommendation_patterns(recs):
    if not recs:
        return {}
    action_types, priorities = {}, {}
    for rec in recs:
        rec_data = rec["recommendation"]
        for action_type in rec_data.get("type_of_action", []):
            action_types[action_type] = action_types.get(action_type, 0) + 1
        priority = rec_data.get("priority", "Unknown")
        priorities[priority] = priorities.get(priority,0)+1
    avg_conf = sum(r["recommendation"].get("confidence",0) for r in recs)/len(recs)
    return {"action_type_distribution": action_types, "priority_distribution": priorities, "average_confidence": avg_conf}

# =========================
# --- Serialization ---
# =========================
def _serialize_patient(patient: PatientSummary):
    def _fmt(dt): return dt.strftime("%Y-%m-%d %H:%M") if dt else None
    age = None
    if getattr(patient,"date_of_birth",None):
        today = datetime.now(timezone.utc).date()
        dob = patient.date_of_birth.date()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return {
        "patient_id": patient.patient_id,
        "pseudo_name": patient.pseudo_name,
        "bed": patient.bed,
        "sex": patient.sex,
        "age": age,
        "date_of_birth": _fmt(patient.date_of_birth),
        "admission_dx": patient.admission_dx,
        "last_update": _fmt(patient.last_update),
        "vitals": [{**v.model_dump(), "timestamp": _fmt(v.timestamp)} for v in sorted(patient.vitals,key=lambda x:x.timestamp)],
        "labs": [{**l.model_dump(), "timestamp": _fmt(l.timestamp)} for l in sorted(patient.labs,key=lambda x:x.timestamp)],
        "meds": [{**m.model_dump(), "start_time": _fmt(m.start_time), **({"stop_time": _fmt(m.stop_time)} if hasattr(m,"stop_time") else {})} for m in sorted(patient.meds,key=lambda x:x.start_time)],
        "imaging": [{**i.model_dump(), "datetime": _fmt(i.datetime)} for i in sorted(patient.imaging,key=lambda x:x.datetime)],
        "events": [{**e.model_dump(),"timestamp":_fmt(e.timestamp),"description":getattr(e,"description",getattr(e,"details","N/A"))} for e in sorted(patient.events,key=lambda x:x.timestamp)],
        "nurse_note": getattr(patient,"nurse_note",None),
    }

def _extract_json_from_text(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except json.JSONDecodeError: return None
    return None

# =========================
# --- LangChain & RAG ---
# =========================
import tiktoken  # For token counting (works with OpenAI models)

def _estimate_tokens(text: str, model_name="gpt-4-turbo"):
    """Estimate number of tokens in text for a given model"""
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

def get_patient_context_vectorstore(patient_id: str, new_data: dict, max_tokens=3000, top_k=10):
    """
    Retrieve past patient data, doctor recs, and similar cases, truncated to fit within token limit.
    Compatible with langchain_core + langchain_community FAISS.
    """
    db = get_database()

    # --- Prepare documents ---
    past_entries = list(db['patient_data'].find({"patient_id": patient_id, "type": "patient_data"}).sort("timestamp", 1))
    past_docs = [Document(page_content=json.dumps(e['data']), metadata={"type": e['type'], "timestamp": str(e['timestamp'])}) for e in past_entries]

    doctor_rec = db['patient_data'].find_one({"patient_id": patient_id, "type": "doctor_rec"}, sort=[("timestamp",-1)])
    if doctor_rec:
        past_docs.append(Document(page_content=json.dumps(doctor_rec['recommendation']),
                                  metadata={"type": "doctor_rec", "timestamp": str(doctor_rec['timestamp'])}))

    other_entries = list(db['patient_data'].find({"patient_id": {"$ne": patient_id}, "type": "patient_data"}))
    similar_docs = [Document(page_content=json.dumps(e['data']), metadata={"type": e['type'], "patient_id": e['patient_id'], "timestamp": str(e['timestamp'])}) for e in other_entries]

    all_docs = past_docs + similar_docs

    # --- Build FAISS vector store ---
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vector_store = FAISS.from_documents(all_docs, embeddings)

    # --- Embed query and search ---
    query_vector = embeddings.embed_query(json.dumps(new_data))
    D, I = vector_store.index.search(np.array([query_vector], dtype=np.float32), k=top_k)

    # --- Retrieve documents safely ---
    retrieved_docs = []
    for idx in I[0]:
        if idx >= 0:
            doc_id = str(idx)  # FAISS stores keys as strings in this version
            if doc_id in vector_store.docstore._dict:
                retrieved_docs.append(vector_store.docstore._dict[doc_id])

    # --- Token-aware truncation ---
    context_text = ""
    tokens_used = 0
    for doc in retrieved_docs[::-1]:  # start from most relevant
        doc_tokens = _estimate_tokens(doc.page_content)
        if tokens_used + doc_tokens > max_tokens:
            break
        context_text = doc.page_content + "\n" + context_text
        tokens_used += doc_tokens

    return context_text
    # Token-aware truncation

def get_similar_cases_with_outcomes(current_patient_data: dict, patient_id: str, top_k=5, max_tokens=2000):
    """
    Find semantically similar past cases from ALL patients.
    Returns cases with their outcomes and what worked.
    """
    db = get_database()
    
    # Get all past cases with both AI and doctor recommendations (these have outcomes)
    all_cases = list(db['patient_data'].find({
        "patient_id": {"$ne": patient_id},  # Exclude current patient
        "type": "patient_data"
    }))
    
    # For each case, try to find its corresponding doctor recommendation (outcome)
    cases_with_outcomes = []
    for case in all_cases:
        # Find doctor recommendation for this case
        doctor_rec = db['patient_data'].find_one({
            "patient_id": case['patient_id'],
            "type": "doctor_rec",
            "timestamp": {"$gte": case['timestamp']}
        }, sort=[("timestamp", 1)])
        
        if doctor_rec:
            # Create rich case description
            case_text = f"""
Patient Profile:
- Age: {case['data'].get('age', 'N/A')} | Sex: {case['data'].get('sex', 'N/A')}
- Diagnosis: {case['data'].get('admission_dx', 'N/A')}

Clinical Presentation:
Vitals: {', '.join([f"{v.get('label', v.get('code', 'Unknown'))}: {v.get('value')} {v.get('unit', '')}" for v in case['data'].get('vitals', [])[:8]])}

Labs: {', '.join([f"{l.get('name')}: {l.get('value')} {l.get('unit', '')}" for l in case['data'].get('labs', [])[:8]])}

Medications: {', '.join([m.get('name') for m in case['data'].get('meds', [])])}

Doctor's Actions Taken:
{chr(10).join([f"- {action}" for action in doctor_rec['recommendation'].get('recommended_action', [])[:5]])}

Assessment: {doctor_rec['recommendation'].get('assessment', 'N/A')[:200]}

Outcome: {doctor_rec['recommendation'].get('expected_outcome', 'N/A')[:150]}
Priority: {doctor_rec['recommendation'].get('priority', 'N/A')}
"""
            cases_with_outcomes.append({
                "text": case_text,
                "patient_id": case['patient_id'],
                "timestamp": case['timestamp'],
                "priority": doctor_rec['recommendation'].get('priority'),
                "confidence": doctor_rec['recommendation'].get('confidence', 0)
            })
    
    if not cases_with_outcomes:
        return ""
    
    # Create documents for vector search
    docs = [Document(
        page_content=c['text'],
        metadata={
            "patient_id": c['patient_id'],
            "timestamp": str(c['timestamp']),
            "priority": c['priority']
        }
    ) for c in cases_with_outcomes]
    
    # Build vector store
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Create query from current patient
    query_text = f"""
Age: {current_patient_data.get('age', 'N/A')} | Sex: {current_patient_data.get('sex', 'N/A')}
Diagnosis: {current_patient_data.get('admission_dx', 'N/A')}
Vitals: {', '.join([f"{v.get('label', v.get('code', 'Unknown'))}: {v.get('value')}" for v in current_patient_data.get('vitals', [])])}
Labs: {', '.join([f"{l.get('name')}: {l.get('value')}" for l in current_patient_data.get('labs', [])])}
Medications: {', '.join([m.get('name') for m in current_patient_data.get('meds', [])])}
"""
    
    # Semantic search
    similar_docs = vector_store.similarity_search(query_text, k=top_k)
    
    # Build context with token limit
    context = "\n=== SIMILAR PAST CASES FROM YOUR ICU ===\n"
    context += f"Found {len(similar_docs)} similar cases. Learn from what worked:\n\n"
    
    tokens_used = _estimate_tokens(context)
    
    for idx, doc in enumerate(similar_docs, 1):
        case_text = f"Similar Case #{idx}:\n{doc.page_content}\n{'='*60}\n\n"
        case_tokens = _estimate_tokens(case_text)
        
        if tokens_used + case_tokens > max_tokens:
            break
            
        context += case_text
        tokens_used += case_tokens
    
    return context

from langsmith.run_helpers import traceable
from langsmith import Client
import langsmith

@traceable(name="ICU_AI_Recommendation_Generation")
def generate_recommendations_openai(patient: PatientSummary, language="English"):
    """
    Generate ICU recommendations with:
    1. Patient's own history (last 5 visits)
    2. Similar cases from other patients (RAG)
    3. Last doctor's recommendation
    """
    
    # Initialize LangSmith client
    ls_client = Client()
    
    # Serialize patient for storage
    patient_data = _serialize_patient(patient)
    store_patient_data(patient, patient_data)
    
    # ========================
    # --- Build Comprehensive Context ---
    # ========================
    db = get_database()
    
    # 1. Patient's own history (last 5 visits with details)
    with langsmith.trace(name="Step_1_Patient_History", inputs={"patient_id": patient.patient_id}) as history_trace:
        past_entries = list(
            db['patient_data'].find({"patient_id": patient.patient_id, "type": "patient_data"})
            .sort("timestamp", -1).limit(5)
        )
        
        history_context = "=== THIS PATIENT'S HISTORY (Last 5 visits) ===\n\n"
        for idx, e in enumerate(reversed(past_entries), 1):
            data = e['data']
            history_context += f"Visit {idx} - {e['timestamp'].strftime('%Y-%m-%d %H:%M')}:\n"
            vitals_text = ', '.join(
                f"{v.get('label', v.get('code', 'Unknown'))}: {v.get('value')}"
                for v in data.get('vitals', [])[:5]
            )
            history_context += f"  Vitals: {vitals_text}\n"
            labs_text = ', '.join(
                f"{l.get('name', 'Unknown')}: {l.get('value')}"
                for l in data.get('labs', [])[:5]
            )
            history_context += f"  Labs: {labs_text}\n"
            meds_text = ', '.join(
                m.get('name', 'Unknown')
                for m in data.get('meds', [])
            )
            history_context += f"  Meds: {meds_text}\n\n"
        
        # Output to LangSmith
        history_trace.end(
            outputs={
                "visits_found": len(past_entries),
                "full_history_context": history_context
            }
        )
    
    # 2. Last doctor's recommendation
    with langsmith.trace(name="Step_2_Doctor_Recommendation", inputs={"patient_id": patient.patient_id}) as doctor_trace:
        doctor_rec = db['patient_data'].find_one(
            {"patient_id": patient.patient_id, "type": "doctor_rec"},
            sort=[("timestamp", -1)]
        )
        
        doctor_context = ""
        if doctor_rec:
            doctor_context = "\n=== LAST DOCTOR'S RECOMMENDATION ===\n"
            doctor_context += f"Date: {doctor_rec['timestamp'].strftime('%Y-%m-%d %H:%M')}\n"
            doctor_context += f"Assessment: {doctor_rec['recommendation'].get('assessment', 'N/A')}\n"
            doctor_context += "Actions taken:\n"
            for action in doctor_rec['recommendation'].get('recommended_action', [])[:5]:
                doctor_context += f"  - {action}\n"
            doctor_context += f"Outcome: {doctor_rec['recommendation'].get('expected_outcome', 'N/A')}\n\n"
        
        # Output to LangSmith
        doctor_trace.end(
            outputs={
                "doctor_rec_found": bool(doctor_rec),
                "doctor_recommendation_full": doctor_context if doctor_context else "No previous doctor recommendation",
                "timestamp": doctor_rec['timestamp'].isoformat() if doctor_rec else None
            }
        )
    
    # 3. Similar cases using RAG (semantic search)
    with langsmith.trace(name="Step_3_RAG_Similar_Cases", inputs={"patient_id": patient.patient_id, "top_k": 5}) as rag_trace:
        similar_cases_context = get_similar_cases_with_outcomes(
            patient_data, 
            patient.patient_id, 
            top_k=5,
            max_tokens=2000
        )
        
        # Output to LangSmith
        rag_trace.end(
            outputs={
                "similar_cases_found": "Yes" if "Similar Case" in similar_cases_context else "No",
                "similar_cases_context_full": similar_cases_context,
                "context_length": len(similar_cases_context)
            }
        )
    
    # 4. Combine all contexts
    with langsmith.trace(name="Step_4_Combine_Context", inputs={"patient_id": patient.patient_id}) as combine_trace:
        full_context = history_context + doctor_context + similar_cases_context
        
        # Token management - truncate if needed
        context_tokens = _estimate_tokens(full_context)
        if context_tokens > 3000:
            truncated_history = history_context[:int(len(history_context) * 0.3)]
            full_context = truncated_history + doctor_context + similar_cases_context
        
        # Output to LangSmith
        combine_trace.end(
            outputs={
                "total_context_tokens": context_tokens,
                "truncated": context_tokens > 3000,
                "final_combined_context": full_context
            }
        )

    # ========================
    # --- Initialize LLM ---
    # ========================
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4-turbo",
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        max_tokens=4000
    )

    # ========================
    # --- Enhanced ICU Prompt ---
    # ========================
    template = """
You are an experienced ICU physician with 30 years of critical care expertise.

IMPORTANT: You have access to similar past cases from your ICU. Learn from what worked in those cases!

{patient_context}

Goal: Support structured, evidence-based, multidisciplinary ICU care by:
- Reviewing the patient's current status across all core ICU domains: **1. Feeding 2. Analgesia 3. Sedation  4. Thromboprophylaxis 5. Head-Up 6. Ulcer Prophylaxis 7.Glycemic Control,
 8. SBT 9. Bowel Movement 10. Indwelling Catheter 11. Drug De-escalation**
- Learning from similar past cases above - what actions were taken and their outcomes.
- Comparing current patient to their own history and similar cases.
- Recommending individualized adjustments based on proven approaches.

Instructions:
- Analyze all patient data (vitals, labs, meds, imaging, events) and nurse notes.
- Consider similar cases: what worked, what was the priority, what were the outcomes.
- If similar cases showed good outcomes with specific interventions, consider them with CoT.
- Provide 10 extremely detailed, numeric ICU-ready orders with name, dose, route, frequency, lab targets.
- You MUST show your **CoT in the rationale**, explaining how similar cases influenced your decisions.
- You MUST **EXPLAIN** your **CoT in the rationale**, clarifying the reasoning behind choosing the name, dose, route, frequency, lab targets.

Current patient data:
{patient_data}

Return ONLY JSON in this format:
{{
"assessment": "Include how this case compares to similar cases",
  "recommended_action": ["10 specific actions, if any antibiotics are needed, you MUST consider the **Gold Criteria**"],
  "type_of_action": ["Action types"],
  "priority": "High(if the case is too severe and life-threatening)/Medium(needs urgent treatment but not life threatening)/Low(requires attention but not urgent) ",
  "confidence": 0.0-1.0, 
  "rationale": "**EXPLAIN** your **CoT** in a sentence for each recommended action.",
  "supporting_evidence": ["Evidence including lessons from similar cases"],
  "follow_up": "",
  "notes": "Any insights from similar cases",
  "expected_outcome": "Based on similar case outcomes",
  "dependencies": "",
  "timeframe": ""
}}
"""
    
    formatted_prompt = template.format(
        patient_context=full_context,
        patient_data=json.dumps(patient_data, indent=2)
    )

    # ========================
    # --- Call LLM and parse JSON ---
    # ========================
    with langsmith.trace(name="Step_5_LLM_Call", inputs={"prompt_tokens": _estimate_tokens(formatted_prompt)}) as llm_trace:
        try:
            messages = [{"role": "user", "content": formatted_prompt}]
            response = llm.invoke(messages)
            data = _extract_json_from_text(response.content)

            # Ensure all fields exist
            default_fields = {
                "assessment": "",
                "recommended_action": [""]*10,
                "type_of_action": [""],
                "priority": "",
                "confidence": 0.0,
                "rationale": "",
                "supporting_evidence": [""],
                "follow_up": "",
                "notes": "",
                "expected_outcome": "",
                "dependencies": "",
                "timeframe": ""
            }
            if not data:
                data = default_fields
            else:
                for k, v in default_fields.items():
                    if k not in data or data[k] is None:
                        data[k] = v

            # Store AI recommendation
            store_ai_recommendation(patient.patient_id, data, patient_data)
            
            # Output to LangSmith
            llm_trace.end(
                outputs={
                    "recommendation_generated": True,
                    "priority": data.get("priority"),
                    "confidence": data.get("confidence"),
                    "full_response": data
                }
            )

            return data

        except Exception as e:
            llm_trace.end(
                outputs={
                    "error": str(e),
                    "recommendation_generated": False
                }
            )
            
            return {
                "assessment": "Error",
                "recommended_action": ["Error generating recommendations"],
                "confidence": 0.0,
                "rationale": str(e),
                "type_of_action": ["Error"],
                "priority": "N/A",
                "supporting_evidence": [],
                "follow_up": "N/A",
                "notes": str(e),
                "expected_outcome": "N/A",
                "dependencies": "N/A",
                "timeframe": "N/A"
            }
        


# =========================
# --- Visualization ---
# =========================
def plot_vitals_trend(vitals_trend: dict):
    if not vitals_trend: return None
    fig = make_subplots(rows=len(vitals_trend), cols=1, subplot_titles=list(vitals_trend.keys()), vertical_spacing=0.1)
    for idx, (name, data_points) in enumerate(vitals_trend.items(),1):
        fig.add_trace(go.Scatter(x=[d["timestamp"] for d in data_points], y=[d["value"] for d in data_points], mode="lines+markers", name=name, line=dict(width=2)), row=idx, col=1)
    fig.update_layout(height=300*len(vitals_trend), showlegend=False, title_text="Vital Signs Trends")
    return fig

def plot_labs_trend(labs_trend: dict):
    if not labs_trend: return None
    fig = make_subplots(rows=len(labs_trend), cols=1, subplot_titles=list(labs_trend.keys()), vertical_spacing=0.08)
    for idx, (name, data_points) in enumerate(labs_trend.items(),1):
        fig.add_trace(go.Scatter(x=[d["timestamp"] for d in data_points], y=[d["value"] for d in data_points], mode="lines+markers", name=name, line=dict(width=2), marker=dict(size=8)), row=idx, col=1)
    fig.update_layout(height=250*len(labs_trend), showlegend=False, title_text="Laboratory Values Trends")
    return fig

def plot_recommendation_patterns(patterns: dict):
    if not patterns: return None
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Action Type Distribution","Priority Distribution"), specs=[[{"type":"pie"},{"type":"pie"}]])
    if patterns.get("action_type_distribution"):
        fig.add_trace(go.Pie(labels=list(patterns["action_type_distribution"].keys()), values=list(patterns["action_type_distribution"].values()), name="Action Types"), row=1, col=1)
    if patterns.get("priority_distribution"):
        fig.add_trace(go.Pie(labels=list(patterns["priority_distribution"].keys()), values=list(patterns["priority_distribution"].values()), name="Priorities"), row=1, col=2)
    fig.update_layout(height=400, showlegend=True)
    return fig
