import sys
import os

# Add project root dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Imports from local package
from icu_local_ai.models import PatientSummary
from icu_local_ai.icu_ai_cloud import (
    generate_recommendations_openai,
    get_patient_history,
    get_recommendation_history,
    search_patients,
    compare_patient_states,
    generate_trend_report,
    plot_vitals_trend,
    plot_labs_trend,
    plot_recommendation_patterns,
    _serialize_patient,
    store_doctor_recommendation
)
import streamlit as st
import json
from typing import List
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta, timezone


# =======================
# --- Streamlit Setup ---
# =======================
st.set_page_config(page_title="ICU Recommendation System", layout="wide")
st.title(" Hebron Public Hospital-ICU")

# =======================
# --- Sidebar ---
# =======================
st.sidebar.header("ICU Patients")

uploaded = st.sidebar.file_uploader("Upload patient JSON file", type=["json"])
patients_raw = []
if uploaded:
    try:
        patients_raw = json.load(uploaded)
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {e}")
# else:
#     if st.sidebar.button("upload file"):
#         try:
#             with open("patients_sample.json", "r", encoding="utf-8") as f:
#                 patients_raw = json.load(f)
#         except Exception as e:
#             st.sidebar.error(f"Cannot load sample file: {e}")

# =======================
# --- Convert JSON ---
# =======================

patient = None  # default

patients: List[PatientSummary] = []
for pr in patients_raw:
    try:
        patients.append(PatientSummary.model_validate(pr))
    except Exception as e:
        st.error(f"Error parsing patient record: {e}")

if not patients:
    st.info("No data uploaded yet.")
    st.stop()

def calculate_age(dob: datetime) -> int:
    today = datetime.now().date()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

# 🔍 Search input for patient by name
search_query = st.text_input("Search patient by name").strip().lower()

# Filter patients by name
filtered_patients = [
    p for p in patients
    if search_query in p.pseudo_name.lower()
]

if not filtered_patients and search_query:
    st.warning("No matching patients found.")
elif not search_query:
    st.info("Type a patient's first or last name to view details.")
else:
    # Create readable display names for filtered patients
    patient_options = {
        f"{p.patient_id} — {p.pseudo_name} (Bed {p.bed})": p for p in filtered_patients
    }

    # Dropdown for matches
    selected_patient_key = st.selectbox(
        "Select a patient",
        list(patient_options.keys())
    )
    patient = patient_options[selected_patient_key]

if patient is not None:
    # --- Patient Header ---

    if 'patient' in locals():  # Only show header if a patient is selected
        st.markdown(
            f"""
            <div style="
                background-color: #1C1C1C;
                padding: 1.5rem;
                border-radius: 12px;
                margin-top: 1rem;
                box-shadow: 0 2px 5px rgba(0,0,0,0.15);
            ">
                <h3 style="margin-bottom: 0.5rem; color: #FFD700;"> Patient: 
                    <span style="color:#00BFFF;">{patient.patient_id}</span> — 
                    {patient.pseudo_name} (Bed <b>{patient.bed}</b>)
                </h3>
                <p style="margin: 0.3rem 0; font-size: 1rem; color: #EEE;">
                    <b>Sex:</b> {getattr(patient,'sex','N/A')}  &nbsp;|&nbsp;
                    <b>Age:</b> {calculate_age(patient.date_of_birth)}  &nbsp;|&nbsp;
                    <b>Admission Dx:</b> {patient.admission_dx}
                </p>
                <p style="margin: 0.3rem 0; font-size: 0.9rem; color: #AAA;">
                    <b>Last update:</b> {patient.last_update}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    # else:
    #     st.info("Please select a patient to view details.")


    # =======================
    # --- Main Tabs ---
    # =======================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Current Data", 
        "🤖 Recommendations", 
        "📊 Trend Reports", 
        "🔄 Compare States", 
        "📚 History"
    ])

    # =======================
    # TAB 1: Current Data
    # =======================
    with tab1:
        # --- Unified Timeline ---
        timeline_data = []
        for v in patient.vitals:
            timeline_data.append({
                "Type": "Vital",
                "Name": v.label or v.code,
                "Value": str(v.value) + (v.unit or ''),
                "Timestamp": v.timestamp.isoformat(),
                "Details": f"{v.label or v.code}: {v.value}{v.unit or ''}"
            })
        for l in patient.labs:
            timeline_data.append({
                "Type": "Lab",
                "Name": l.name,
                "Value": str(l.value) + (l.unit or ''),
                "Timestamp": l.timestamp.isoformat(),
                "Details": f"{l.name}: {l.value}{l.unit or ''} (ref: {l.reference_range or 'N/A'})"
            })
        for i in patient.imaging:
            timeline_data.append({
                "Type": "Imaging",
                "Name": i.modality,
                "Value": "",
                "Timestamp": i.datetime.isoformat(),
                "Details": f"{i.modality}: {i.report_summary}"
            })
        for e in patient.events:
            timeline_data.append({
                "Type": "Event",
                "Name": "Event",
                "Value": str(getattr(e, "value", "")),
                "Timestamp": e.timestamp.isoformat(),
                "Details": getattr(e, "details", "N/A")
            })

        df_timeline = pd.DataFrame(timeline_data).sort_values("Timestamp")
        if not df_timeline.empty:
            st.subheader("🩺 Unified Clinical Timeline")
            fig = px.scatter(
                df_timeline, x="Timestamp", y="Type", color="Type", symbol="Type",
                hover_data=["Name", "Value", "Details"], title="Chronological Overview"
            )
            fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(template="plotly_white", hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Detailed Data Log")
            st.dataframe(df_timeline[["Timestamp","Type","Name","Value","Details"]].reset_index(drop=True))

        # --- Medications ---
        st.subheader("Medications")
        if patient.meds:
            for m in patient.meds:
                st.write(f"- {m.name} {m.dose} {m.frequency or ''} — started {m.start_time.isoformat()}")
        else:
            st.info("No medications recorded")

        # --- Nurse Notes ---
        if getattr(patient, "nurse_note", None):
            st.subheader("Nurse Notes")
            st.write(patient.nurse_note)

    # =======================
    # TAB 2: Recommendations
    # =======================
    with tab2:
        st.subheader("AI Recommendations (12-Hour ICU Plan)")
        
        # Show context from history
        try:
            history_count = len(get_patient_history(patient.patient_id, limit=5))
            rec_count = len(get_recommendation_history(patient.patient_id, limit=5))
            if history_count > 0 or rec_count > 0:
                st.info(f"📊 Found {history_count} previous data uploads and {rec_count} past recommendations for this patient")
        except:
            pass
        
        if st.button("🔄 Generate Recommendations", type="primary"):
            with st.spinner(f"Generating recommendations for {patient.pseudo_name}..."):
                res = generate_recommendations_openai(patient)
            
            st.success("✅ Recommendations generated and stored in database!")

            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Assessment")
                st.write(res.get("assessment", "N/A"))

                st.markdown("### Recommended Actions")
                actions = res.get("recommended_action", [])
                if isinstance(actions, list):
                    for idx, action in enumerate(actions, 1):
                        st.markdown(f"{idx}. {action}")
                else:
                    st.write(actions)

                st.markdown("### Rationale")
                st.write(res.get("rationale", "N/A"))

                st.markdown("### Supporting Evidence")
                evidence = res.get("supporting_evidence", [])
                if isinstance(evidence, list):
                    for ev in evidence:
                        st.markdown(f"- {ev}")
                else:
                    st.write(evidence)

                st.markdown("### Expected Outcome")
                st.write(res.get("expected_outcome", "N/A"))

                st.markdown("### Follow-up")
                st.write(res.get("follow_up", "N/A"))

            with col2:
                st.markdown("### Quick Info")
                
                # Priority with color
                priority = res.get("priority", "N/A")
                priority_colors = {
                    "High": "🔴",
                    "Medium": "🟡",
                    "Low": "🟢",
                    "Urgent": "🔴",
                    "Routine": "🟢"
                }
                priority_icon = priority_colors.get(priority, "⚪")
                st.markdown(f"**Priority:** {priority_icon} {priority}")
                
                # Confidence
                confidence = res.get("confidence", 0)
                st.metric("Confidence", f"{confidence * 100:.1f}%")
                
                # Type of action
                st.markdown("**Action Types:**")
                action_types = res.get("type_of_action", [])
                if isinstance(action_types, list):
                    for at in action_types:
                        st.markdown(f"- {at}")
                else:
                    st.write(action_types)
                
                # Timeframe
                st.markdown(f"**Timeframe:** {res.get('timeframe', 'N/A')}")
                
                # Dependencies
                st.markdown("**Dependencies:**")
                st.write(res.get("dependencies", "N/A"))

            st.markdown("---")
            st.markdown("### Additional Notes")
            st.write(res.get("notes", "N/A"))

            # Optional: show raw JSON
            if st.checkbox("Show raw JSON"):
                st.json(res)


        # --------------------------------------------
        # Step 3️⃣ Doctor Enters Recommendation (always shown)
        # --------------------------------------------
        st.markdown("### 🩺 Doctor's Recommendation")

    with st.form("doctor_rec_form"):
        st.info("Enter your professional assessment and treatment plan below.")

        assessment = st.text_area("Assessment / Findings")
        actions = st.text_area(
            "Recommended Actions (one per line)", 
            placeholder="e.g., Increase IV fluids to 75 mL/hr\nCheck urine output hourly"
        )
        types = st.multiselect(
            "Type of Action", 
            ["fluid", "monitoring", "medication", "nutrition", "prophylaxis", "other"]
        )
        priority = st.selectbox("Priority", ["low", "medium", "high"])
        supporting_evidence = st.text_area(
            "Supporting Evidence (one per line)", 
            placeholder="e.g., BP trend, lab results, urine output"
        )
        follow_up = st.text_input("Follow-up Instructions", placeholder="Reassess in 4 hours")

        submit = st.form_submit_button("💾 Submit Doctor's Recommendation")

        if submit:
            if not patient:
                st.error("❌ No patient loaded. Cannot save recommendation.")
            else:
                try:
                    patient_data = _serialize_patient(patient)

                    doctor_rec = {
                        "assessment": assessment,
                        "recommended_action": [a.strip() for a in actions.split("\n") if a.strip()],
                        "type_of_action": types if types else ["other"],
                        "priority": priority,
                        "confidence": 1.0,
                        "rationale": "Entered by attending physician",
                        "supporting_evidence": [e.strip() for e in supporting_evidence.split("\n") if e.strip()],
                        "follow_up": follow_up,
                    }

                    inserted_id = store_doctor_recommendation(patient.patient_id, doctor_rec, patient_data)
                    st.success(f"✅ Doctor's recommendation saved successfully! (ID: {inserted_id})")

                except Exception as e:
                    st.error(f"❌ Failed to save doctor recommendation: {e}")
    # =======================
    # TAB 3: Trend Reports
    # =======================
    with tab3:
        st.subheader("📊 Trend Analysis")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            days_back = st.slider("Days to analyze", 1, 30, 7)
        with col2:
            st.info(f"Analyzing trends for the last {days_back} days")
        
        if st.button("📈 Generate Trend Report", type="primary"):
            with st.spinner("Generating comprehensive trend report..."):
                report = generate_trend_report(patient.patient_id, days=days_back)
            
            if not report:
                st.warning("No historical data found for this patient. Upload more data over time to see trends.")
            else:
                # Summary metrics
                st.markdown("### Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Data Points", report['total_updates'])
                with col2:
                    st.metric("Recommendations", report['total_recommendations'])
                with col3:
                    st.metric("Period", f"{report['period_days']} days")
                with col4:
                    avg_conf = report['recommendation_patterns'].get('average_confidence', 0)
                    st.metric("Avg Confidence", f"{avg_conf * 100:.1f}%")
                
                st.markdown("---")
                
                # --- Vitals trends in 3 columns per row ---
                st.markdown("### 📈 Vital Signs Trends")
                vitals_data = report['vitals_trend']  # dictionary: {type: values}

                vital_keys = list(vitals_data.keys())
                for i in range(0, len(vital_keys), 3):
                    cols = st.columns(3)
                    for j, key in enumerate(vital_keys[i:i+3]):
                        fig = plot_vitals_trend({key: vitals_data[key]})
                        with cols[j]:
                            st.subheader(key)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(f"No data for {key}")
                
                st.markdown("---")
                
                # --- Labs trends in 3 columns per row ---
                st.markdown("### 🧪 Laboratory Values Trends")
                labs_data = report['labs_trend']  # dictionary: {type: values}

                lab_keys = list(labs_data.keys())
                for i in range(0, len(lab_keys), 3):
                    cols = st.columns(3)
                    for j, key in enumerate(lab_keys[i:i+3]):
                        fig = plot_labs_trend({key: labs_data[key]})
                        with cols[j]:
                            st.subheader(key)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(f"No data for {key}")
                
                st.markdown("---")

                
                # Medication timeline
                st.markdown("### 💊 Medication Timeline")
                meds_trend = report['meds_trend']
                if meds_trend:
                    meds_df = pd.DataFrame([
                        {
                            'Timestamp': m['timestamp'],
                            'Medication Count': m['count'],
                            'Medications': ', '.join(m['medications'][:5]) + ('...' if len(m['medications']) > 5 else '')
                        }
                        for m in meds_trend
                    ])
                    st.dataframe(meds_df, use_container_width=True)
                else:
                    st.info("No medication data available")
                
                st.markdown("---")
                
                # Recommendation patterns
                st.markdown("### 🎯 Recommendation Patterns")
                patterns_fig = plot_recommendation_patterns(report['recommendation_patterns'])
                if patterns_fig:
                    st.plotly_chart(patterns_fig, use_container_width=True)
                
                # Show pattern details
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Action Type Distribution**")
                    action_dist = report['recommendation_patterns'].get('action_type_distribution', {})
                    for action, count in action_dist.items():
                        st.write(f"- {action}: {count}")
                
                with col2:
                    st.markdown("**Priority Distribution**")
                    priority_dist = report['recommendation_patterns'].get('priority_distribution', {})
                    for priority, count in priority_dist.items():
                        st.write(f"- {priority}: {count}")

    # =======================
    # TAB 4: Compare States
    # =======================
    with tab4:
        st.subheader("🔄 Compare Patient States")
        st.info("Compare patient data between two different time points to see changes")
        
        # Get available timestamps
        try:
            history = get_patient_history(patient.patient_id, limit=50)
            
            if len(history) < 2:
                st.warning("Need at least 2 historical data points to compare. Upload more data over time.")
            else:
                # Create timestamp options
                timestamp_options = {
                    f"{h['timestamp'].strftime('%Y-%m-%d %H:%M')} ({i+1} of {len(history)})": h['timestamp']
                    for i, h in enumerate(history)
                }
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Earlier State**")
                    earlier_key = st.selectbox("Select earlier timestamp", list(timestamp_options.keys()), index=min(1, len(timestamp_options)-1))
                    earlier_ts = timestamp_options[earlier_key]
                
                with col2:
                    st.markdown("**Later State**")
                    later_key = st.selectbox("Select later timestamp", list(timestamp_options.keys()), index=0)
                    later_ts = timestamp_options[later_key]
                
                if st.button("📊 Compare States", type="primary"):
                    comparison = compare_patient_states(patient.patient_id, earlier_ts, later_ts)
                    
                    if not comparison:
                        st.error("Could not retrieve comparison data")
                    else:
                        comp_data = comparison['comparison']
                        
                        st.markdown("### Comparison Results")
                        st.markdown(f"**Period:** {earlier_ts.strftime('%Y-%m-%d %H:%M')} → {later_ts.strftime('%Y-%m-%d %H:%M')}")
                        
                        # Vitals changes
                        st.markdown("#### 📈 Vital Signs Changes")
                        vitals_changes = comp_data.get('vitals_changes', [])
                        if vitals_changes:
                            vitals_df = pd.DataFrame(vitals_changes)
                            st.dataframe(vitals_df, use_container_width=True)
                        else:
                            st.info("No significant vital signs changes detected")
                        
                        # Labs changes
                        st.markdown("#### 🧪 Laboratory Values Changes")
                        labs_changes = comp_data.get('labs_changes', [])
                        if labs_changes:
                            labs_df = pd.DataFrame(labs_changes)
                            st.dataframe(labs_df, use_container_width=True)
                        else:
                            st.info("No significant lab value changes detected")
                        
                        # Medication changes
                        st.markdown("#### 💊 Medication Changes")
                        meds_changes = comp_data.get('meds_changes', {})
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("**Added**")
                            added = meds_changes.get('added', [])
                            if added:
                                for med in added:
                                    st.write(f"✅ {med}")
                            else:
                                st.write("None")
                        
                        with col2:
                            st.markdown("**Removed**")
                            removed = meds_changes.get('removed', [])
                            if removed:
                                for med in removed:
                                    st.write(f"❌ {med}")
                            else:
                                st.write("None")
                        
                        with col3:
                            st.markdown("**Continued**")
                            continued = meds_changes.get('continued', [])
                            if continued:
                                for med in continued[:5]:  # Show first 5
                                    st.write(f"➡️ {med}")
                                if len(continued) > 5:
                                    st.write(f"... and {len(continued)-5} more")
                            else:
                                st.write("None")
        
        except Exception as e:
            st.error(f"Error loading comparison data: {e}")

    # =======================
    # TAB 5: History
    # =======================
    with tab5:
        st.subheader("📚 Patient History & Search")
        
        # Search section
        with st.expander("🔍 Search Patient Data", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                search_patient_id = st.text_input("Patient ID", value=patient.patient_id)
            with col2:
                days_back_search = st.number_input("Days back", min_value=1, max_value=365, value=30)
            
            if st.button("🔎 Search", type="primary"):
                date_from = datetime.now(timezone.utc) - timedelta(days=days_back_search)
                results = search_patients({
                    "patient_id": search_patient_id,
                    "date_from": date_from
                })
                
                st.markdown(f"### Search Results: {len(results)} records found")
                
                if results:
                    for idx, result in enumerate(results[:10], 1):  # Show first 10
                        with st.expander(f"Record {idx} - {result['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                            data = result['data']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Basic Info**")
                                st.write(f"Patient: {data.get('pseudo_name', 'N/A')}")
                                st.write(f"Bed: {data.get('bed', 'N/A')}")
                                st.write(f"Age: {data.get('age', 'N/A')}")
                                st.write(f"Diagnosis: {data.get('admission_dx', 'N/A')}")
                            
                            with col2:
                                st.markdown("**Data Summary**")
                                st.write(f"Vitals: {len(data.get('vitals', []))} readings")
                                st.write(f"Labs: {len(data.get('labs', []))} results")
                                st.write(f"Medications: {len(data.get('meds', []))} active")
                                st.write(f"Imaging: {len(data.get('imaging', []))} studies")
                            
                            if st.checkbox(f"Show full data for record {idx}", key=f"show_data_{idx}"):
                                st.json(data)
        
        st.markdown("---")
        
        # Recommendation history
        st.markdown("### 🤖 Recommendation History")

        try:
            rec_history = get_recommendation_history(patient.patient_id, limit=50)
            
            if not rec_history:
                st.info("No recommendations found for this patient yet. Generate your first recommendation in the Recommendations tab.")
            else:
                # Separate AI vs Doctor recommendations by rationale
                ai_recs = [r for r in rec_history if r['recommendation'].get('rationale','') != "Entered by attending physician"]
                doctor_recs = [r for r in rec_history if r['recommendation'].get('rationale','') == "Entered by attending physician"]

                # Display AI recommendations
                st.markdown("#### 🤖 AI Recommendations")
                if ai_recs:
                    st.write(f"Found {len(ai_recs)} AI recommendations")
                    for idx, rec_record in enumerate(ai_recs, 1):
                        timestamp = rec_record['timestamp'].strftime('%Y-%m-%d %H:%M')
                        rec = rec_record['recommendation']
                        
                        priority = rec.get('priority', 'N/A')
                        confidence = rec.get('confidence', 0)
                        priority_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "Urgent": "🔴"}
                        priority_icon = priority_colors.get(priority, "⚪")
                        
                        with st.expander(f"{idx}. {timestamp} - {priority_icon} Priority: {priority} (Confidence: {confidence*100:.0f}%)"):
                            st.markdown("**Assessment:**")
                            st.write(rec.get('assessment', 'N/A'))
                            
                            st.markdown("**Recommended Actions:**")
                            actions = rec.get('recommended_action', [])
                            if isinstance(actions, list):
                                for action in actions:
                                    st.write(f"- {action}")
                            else:
                                st.write(actions)
                            
                            st.markdown("**Rationale:**")
                            st.write(rec.get('rationale', 'N/A'))
                            
                            if st.checkbox(f"Show full AI recommendation {idx}", key=f"show_ai_rec_{idx}"):
                                st.json(rec)
                else:
                    st.info("No AI recommendations found.")

                # Display Doctor recommendations
                st.markdown("#### 🩺 Doctor Recommendations")
                if doctor_recs:
                    st.write(f"Found {len(doctor_recs)} doctor-entered recommendations")
                    for idx, rec_record in enumerate(doctor_recs, 1):
                        timestamp = rec_record['timestamp'].strftime('%Y-%m-%d %H:%M')
                        rec = rec_record['recommendation']
                        
                        priority = rec.get('priority', 'N/A')
                        confidence = rec.get('confidence', 0)
                        priority_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "Urgent": "🔴"}
                        priority_icon = priority_colors.get(priority, "⚪")
                        
                        with st.expander(f"{idx}. {timestamp} - {priority_icon} Priority: {priority} (Confidence: {confidence*100:.0f}%)"):
                            st.markdown("**Assessment:**")
                            st.write(rec.get('assessment', 'N/A'))
                            
                            st.markdown("**Recommended Actions:**")
                            actions = rec.get('recommended_action', [])
                            if isinstance(actions, list):
                                for action in actions:
                                    st.write(f"- {action}")
                            else:
                                st.write(actions)
                            
                            st.markdown("**Rationale:**")
                            st.write(rec.get('rationale', 'N/A'))
                            
                            if st.checkbox(f"Show full doctor recommendation {idx}", key=f"show_doc_rec_{idx}"):
                                st.json(rec)
                else:
                    st.info("No doctor recommendations found.")

        except Exception as e:
            st.error(f"Error loading recommendation history: {e}")

        st.markdown("---")

        
        # Data upload history
        st.markdown("### 📊 Data Upload History")
        try:
            data_history = get_patient_history(patient.patient_id, limit=20)
            
            if not data_history:
                st.info("No historical data found. Data will be automatically saved when you generate recommendations.")
            else:
                st.write(f"Found {len(data_history)} data uploads")
                
                history_df = pd.DataFrame([
                    {
                        'Timestamp': h['timestamp'].strftime('%Y-%m-%d %H:%M'),
                        'Vitals': len(h['data'].get('vitals', [])),
                        'Labs': len(h['data'].get('labs', [])),
                        'Medications': len(h['data'].get('meds', [])),
                        'Imaging': len(h['data'].get('imaging', []))
                    }
                    for h in data_history
                ])
                
                st.dataframe(history_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading data history: {e}")
