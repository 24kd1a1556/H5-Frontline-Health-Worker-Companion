# H5 — Frontline Health-Worker Companion for Low-Connectivity Areas



## 1. Project Overview



The **Frontline Health-Worker Companion** is an offline-first prototype designed to support frontline health workers, such as ASHA and ANM workers, who need quick access to health protocols while visiting households in areas with limited or unreliable internet connectivity.



The project focuses on three core tasks:



1. **Protocol-based question answering:** Retrieve relevant health-protocol content, provide a section citation, and identify questions that require escalation or refusal.

2. **Household visit-list generation:** Process household-register data and generate a prioritised list of follow-up activities.

3. **Offline operation and footprint measurement:** Run locally without a neural model or an internet connection during inference, and report execution latency and model footprint.



The project is developed for the H5 problem statement of the AP Talent Mega Hackathon 2026.



> **Important:** This is a hackathon prototype, not a clinically validated medical system. It does not replace professional medical judgement, official health protocols, or emergency referral procedures.



## 2. Problem Statement



Frontline health workers may need to consult maternal-health, newborn-care, immunisation, nutrition, and chronic-disease protocols while visiting families. In low-connectivity areas, accessing online information may be difficult or impossible.



The proposed solution aims to make relevant protocol information available locally, flag danger-related questions for escalation, decline unsupported questions, and help organise household follow-ups.



The system is designed around the following requirements:



- Work with limited or no internet connectivity.

- Retrieve relevant information from the available protocol content.

- Include protocol-section citations with answers.

- Apply a rule-based danger-sign escalation mechanism.

- Refuse questions that fall outside the supported scope.

- Generate a structured household visit list.

- Report the implementation's measured resource footprint.



## 3. Solution Architecture



The current implementation uses a lightweight, rule-based and retrieval-based approach rather than a neural language model.



### 3.1 Protocol question answering



The answering pipeline uses keyword-based retrieval with BM25-style ranking to identify a relevant protocol section. A small Hinglish-to-English word map helps match some common expressions to English medical terms.



The implementation also includes keyword-based rules for danger-sign escalation and out-of-scope refusal.



The main steps are:



1. Read the supplied question data.

2. Apply the implemented safety and scope rules.

3. Retrieve the most relevant protocol section.

4. Construct the answer and attach a section citation.

5. Record the escalation and refusal flags in the output.



Keyword matching has limitations: different wording, misspellings, ambiguous questions, or missing protocol information can lead to an incorrect or incomplete result. The prototype must therefore not be treated as an autonomous clinical decision-maker.



### 3.2 Household visit-list generation



The visit-list generator processes the available household-register data and applies implemented rules for selected follow-up categories, including antenatal care, vaccination, non-communicable disease follow-up, and MUAC-related checks.



It produces a CSV containing the member identifier, due item, and priority.



**Known limitation:** The current visit-list logic is simplified. It does not implement every rule in the full visit-list specification, including some HBNC visits, Td rules, upper-age limits, complex vaccine codes, and high-risk prioritisation. The generated list must not be assumed to be a complete clinical schedule.



### 3.3 Offline execution and footprint



The current baseline does not use a neural model. Its inference and retrieval logic runs locally using Python and the required local data files.



The footprint report records:



- Model name.

- Model size in megabytes.

- Peak RAM measurement, when available.

- Median CPU latency in milliseconds.

- Whether the implementation operates offline.



A reported model size of zero means that no neural model is used; it does not mean the complete program consumes zero memory or disk space.



## 4. Technology Stack



- **Language:** Python 3.9 or later

- **Data processing:** pandas

- **Retrieval:** BM25-style keyword ranking

- **Safety and scope handling:** Rule-based keyword checks

- **Output formats:** JSONL, CSV, and JSON

- **Runtime:** Local CPU execution; no neural-model download is required by the baseline



The current implementation is command-line based. A Streamlit web interface is included in `FULL_CODE/app.py`. It provides a browser-based interface for asking frontline health-worker protocol questions and viewing household follow-up information. The interface uses the project's local protocol resources and does not require a hosted AI API.



## 5. Project Structure



The final submission package is organised as follows:



```text

H5_Frontline_Health_Worker_Companion/
├── README.md
├── .gitignore
├── FULL_CODE/
│   ├── app.py
│   ├── baseline.py
│   ├── requirements.txt
│   └── tests/
│       └── test_baseline.py
└── submission/
    ├── H5_answers.jsonl
    ├── H5_footprint.json
    └── H5_visit_list.csv

```



The supplied data files and development evaluator are kept outside this minimal submission structure unless the organisers explicitly require them to be included.



For local execution, place the supplied `data/` directory at the project root. The directory should contain the required question and household-register files.



```text

H5_Frontline_Health_Worker_Companion/

├── data/

│   ├── questions_dev.csv

│   ├── questions_test.csv

│   ├── household_register.csv

│   ├── households.csv

│   └── ashas.csv

├── README.md

├── .gitignore

├── submission/

└── FULL_CODE/

```



Use the complete data directory provided with the hackathon pack; the illustration above is not an exhaustive file listing.



## 6. Prerequisites



Install Python 3.9 or later and ensure that Python is available from your terminal.



Install the dependencies required to run the baseline and development evaluator:



```bash

python -m pip install pandas numpy

```



The baseline requires `pandas`. `numpy` is required by the supplied evaluation script.



No neural model download is needed for this baseline.



## 7. How to Run the Project



Run the following commands from the project root directory.



### 7.1 Run the Streamlit web app

From the project root, install the application dependencies:

```bash
python -m pip install -r FULL_CODE/requirements.txt
```

Start the web interface:

```bash
python -m streamlit run FULL_CODE/app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

The app needs the required local protocol and household data files in the locations expected by the implementation. Installing the Python packages does not install or provide the hackathon dataset.

### 7.2 Generate development outputs

Run the baseline against the development questions:

```bash
python FULL_CODE/baseline.py --split dev --out submission_dev
```

This generates development outputs in `submission_dev/`.

### 7.3 Evaluate development outputs

If the supplied development evaluator and its truth file are available, run:

```bash
python evaluate.py --submission submission_dev --truth data/dev_truth.jsonl
```

Development scores are not hidden-test scores. Do not use the development truth file to claim that the hidden test questions have been evaluated.

### 7.4 Generate the final test submission

```bash
python FULL_CODE/baseline.py --split test --out submission
```

This creates or overwrites the output files in `submission/`. Run this only when you intentionally want to regenerate the final test outputs.

## 8. Submission Output Files



The `submission/` directory must contain all three required files.



### `H5_answers.jsonl`



Contains one JSON object per question, with fields such as:



- `question_id` — question identifier.

- `answer_text` — generated answer text.

- `cited_section_ids` — cited protocol-section identifiers.

- `escalate` — whether the implemented rules flag escalation.

- `refused` — whether the implemented rules refuse the question.

- `latency_ms` — recorded response latency.



The expected test output contains 200 answer records, one per test question.



### `H5_visit_list.csv`



Contains the generated household follow-up list with these required columns:



- `member_id`

- `due_item`

- `priority`



The current rule engine is simplified, so the list should be reviewed against the complete specification before any real-world use.



### `H5_footprint.json`



Contains the footprint measurements, including model information, reported memory usage, median CPU latency, and offline status.



The footprint values describe the current implementation and the environment in which it was measured. They should not be interpreted as universal performance guarantees.



## 9. Safety and Responsible Use



Health-related outputs require particular care. This prototype has rule-based escalation and refusal mechanisms, but keyword rules cannot reliably recognise every danger sign or every unsafe request.



Accordingly:



- Treat emergency symptoms and danger signs as requiring appropriate professional assessment and referral.

- Do not use the prototype to independently diagnose patients or prescribe medicines.

- Do not assume that an answer is correct merely because it includes a citation.

- Verify clinical content against the applicable official protocol.

- Escalate uncertain, ambiguous, or unsupported cases to an authorised healthcare professional.

- Do not use the simplified visit list as a substitute for official follow-up schedules.

- Protect household and patient information; use only data authorised for the hackathon.



The scenario and supplied data are for the hackathon exercise. This prototype has not been established as clinically validated or approved for deployment.



## 10. Current Validation Status



During local development, the current `FULL_CODE/baseline.py` was executed against the development split and generated the expected three output files.



The supplied development evaluator reported a primary score of **1.0000** for that development run, with the escalation gate passing.



The final test-output structure was also checked locally:



- 200 answer records.

- 200 unique question IDs.

- No missing answer text in the structural check.

- No missing citations in the structural check.



These checks establish successful execution and structural compliance with the checked fields. They do **not** establish the medical correctness of every answer, completeness of the visit list, or the score on hidden test data.



## 11. Limitations and Future Improvements



Potential improvements include:



1. Add a Streamlit interface suitable for low-end devices and frontline workflows.

2. Improve protocol retrieval for synonyms, spelling variations, and code-mixed language.

3. Expand and validate danger-sign and out-of-scope detection.

4. Implement the complete visit-list rules, including missing categories and priority conditions.

5. Add tests for missing files, ambiguous questions, incorrect citations, and safety-rule edge cases.

6. Validate all health-related outputs against the supplied official protocol documents with qualified review.

7. Test resource usage on representative low-resource hardware.

8. Consider safe local data handling and synchronisation when connectivity becomes available.



These improvements should be implemented and tested before describing the corresponding features as completed.



## 12. Reproducibility



For reproducible results:



- Use the supplied hackathon data files.

- Run commands from the project root.

- Keep the original question identifiers and expected output schema unchanged.

- Record the Python and dependency versions used for testing.

- Keep the development and hidden-test workflows separate.

- Regenerate submission files only when intentional, because the generation command overwrites existing files in the selected output directory.


### Running the tests

From the project root, run:

```bash
python -m unittest discover -s FULL_CODE/tests -v
```

The current test suite covers basic text tokenisation, BM25 retrieval, result limits, and protocol-section selection. These tests do not establish clinical safety or validate every danger-sign and refusal scenario.

Before submission, run the tests against the final project structure and confirm that they pass.

## 13. AI Usage

ChatGPT was used as a development assistant for project planning, README editing, code review, debugging support, and test design. The team reviewed and tested the implementation during development.

The application uses local protocol resources and does not call a hosted AI API at runtime. Its retrieval and safety rules are heuristic-based and have not been clinically validated.


## 14. Acknowledgement



Developed for the **AP Talent Mega Hackathon 2026 — H5: Frontline Health-Worker Companion for Low-Connectivity Areas**.



The implementation is intended as a demonstration of offline retrieval, rule-based safety handling, structured household follow-up generation, and lightweight execution—not as a replacement for professional healthcare services.





