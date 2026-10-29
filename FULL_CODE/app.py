
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
BASELINE_PATH = APP_DIR / "baseline.py"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import baseline

st.set_page_config(
    page_title="Frontline Health Companion",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f5f8f6; }
    .block-container { padding-top: 2rem; max-width: 1150px; }
    .hero {
        background: linear-gradient(120deg, #145c43, #248464);
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; margin: 0; }
    .hero p { color: #e8f5ee; margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🩺 Frontline Health-Worker Companion</h1>
      <p>Protocol reference and household follow-up support for ASHA and ANM workers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Offline-capable baseline demo . Verify clinical decisions against approved protocols."
)

protocol_dir = PROJECT_ROOT / "data" / "protocols"
household_path = PROJECT_ROOT / "data" / "households.csv"

if not protocol_dir.exists():
    st.error(
        "Protocol data was not found. Place the authorised problem-pack data "
        "in the project's data/protocols folder, as described in README.md."
    )
    st.stop()

try:
    sections = baseline.load_sections()
except Exception as exc:
    st.error(f"Could not load protocol sections: {exc}")
    st.stop()

if not sections:
    st.error("No protocol sections were loaded. Check the protocol data files.")
    st.stop()

tab_ask, tab_visits, tab_about = st.tabs(
    ["💭 Ask a protocol question", "📝 Household visit list", "ℹ️ About"]
)

with tab_ask:
    st.subheader("Ask a health-protocol question")
    st.write(
        "Enter a question from the authorised protocol scope. "
        "Danger signs must be escalated immediately."
    )

    question = st.text_area(
        "Question",
        placeholder="Example: What should I do if a pregnant woman has heavy bleeding?",
        height=110,
        key="health_question",
    )

    if st.button("Find protocol guidance", type="primary"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                tokens = baseline.tokenize(question)
                is_danger = any(
                    phrase in question.lower()
                    for phrase in baseline.DANGER
                )
                is_refused = any(
                    phrase in question.lower()
                    for phrase in baseline.REFUSE
                )

                # Safety-first gate: show the escalation message before
                # displaying any routine retrieved protocol text.
                if is_danger:
                    st.error(
                        "🚨🚑 DANGER SIGN: Escalate immediately to the "
                        "appropriate medical professional or emergency service. "
                        "Do not delay referral while using this app."
                    )

                if is_refused:
                    st.warning(
                        "This app cannot safely provide that requested advice. "
                        "Please consult the Medical Officer or authorised "
                        "clinical protocol."
                    )
                elif is_danger:
                    st.info(
                        "Follow the approved emergency referral protocol. "
                        "This prototype is not a substitute for clinical judgement."
                    )
                else:
                    bm25 = baseline.BM25(sections)
                    section_id = baseline.choose_section(question, bm25)

                    # Check weak keyword-search matches.
                    hits = bm25.search(question, k=1)

                    if hits:
                        score, candidate_id = hits[0]

                        # Preserve topic-specific rules.
                        # Filter only weak BM25 fallback matches.
                        if section_id == candidate_id:
                            question_tokens = set(
                                baseline.tokenize(question)
                            )
                            section_tokens = set(
                                baseline.tokenize(sections[candidate_id])
                            )

                            stop_words = {
                                "i", "am", "is", "are", "the", "a", "an",
                                "to", "of", "for", "from", "in", "on",
                                "what", "how", "do", "does", "should",
                                "my", "me", "we", "our", "can", "you"
                            }

                            matched_tokens = (
                                question_tokens & section_tokens
                            )
                            meaningful_matches = (
                                matched_tokens - stop_words
                            )

                            if (
                                score < 10
                                or len(meaningful_matches) < 2
                            ):
                                section_id = None
                    else:
                        section_id = None

                    if section_id and section_id in sections:
                        st.success("Protocol section retrieved")
                        st.markdown("**Source section:** " + section_id)
                        st.write(sections[section_id])
                    else:
                        st.warning(
                            "No suitable protocol section was found. "
                            "Please consult the authorised protocol or Medical Officer."
                        )
            except Exception as exc:
                st.error(f"Could not process the question: {exc}")

with tab_visits:
    st.subheader("Household follow-up support")
    st.write("View the generated visit list when it is available.")

    generated_csv = PROJECT_ROOT / "submission" / "H5_visit_list.csv"
    if not generated_csv.exists():
        generated_csv = (
            PROJECT_ROOT / "FINAL_SUBMISSION" / "submission" / "H5_visit_list.csv"
        )

    if generated_csv.exists():
        try:
            visits = pd.read_csv(generated_csv)
            st.metric("Visit-list entries", len(visits))
            if "priority" in visits.columns:
                st.dataframe(
                    visits,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.dataframe(visits, use_container_width=True, hide_index=True)
            st.download_button(
                "Download visit list CSV",
                data=visits.to_csv(index=False).encode("utf-8"),
                file_name="H5_visit_list.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not read the visit list: {exc}")
    else:
        st.info(
            "The visit list is not generated yet. Run the baseline using "
            "the authorised data before opening this tab."
        )

with tab_about:
    st.subheader("About this prototype")
    st.markdown(
        """
        - **Retrieval:** BM25 keyword search over supplied protocol sections.
        - **Safety:** Keyword-based danger-sign escalation and out-of-scope refusal.
        - **Visit list:** Displays the baseline-generated household follow-up CSV.
        - **Limitations:** This is a prototype, not a clinically validated medical system.
        - **Connectivity:** The app uses local files and does not require a remote AI API,
          but the complete offline workflow still needs to be tested on the target device.
        """
    )
    st.warning(
        "For urgent symptoms, follow approved clinical escalation procedures. "
        "Never rely on this prototype as the sole basis for medical decisions."
    )

