# AeroLLM Dataset

## Overview

`AeroLLM_Data` is the curated aviation-domain dataset prepared for the **AeroLLM** project.

The purpose of this dataset is to provide reliable aviation-related information for the downstream **Retrieval-Augmented Generation (RAG)** pipeline. The collected data covers aviation incidents, accidents, maintenance issues, FAA guidance, and aviation maintenance knowledge.

The dataset is organized into source-specific directories so that the next stages of the project can independently perform:

* Document chunking
* Text preprocessing
* Embedding generation
* Vector database creation
* Semantic retrieval
* RAG-based question answering

---

## Dataset Sources

The dataset currently contains information from the following major sources:

### 1. NASA ASRS

The Aviation Safety Reporting System (ASRS) data contains aviation incident and safety reports.

```text
ASRS/
├── asrs_train.csv
├── asrs_validation.csv
└── asrs_test.csv
```

These files provide aviation safety-related incident information that can be used for training, validation, testing, retrieval, and domain analysis.

---

### 2. FAA Service Difficulty Reports

```text
FAA_SDR/
└── faa_sdr_clean.csv
```

The FAA Service Difficulty Reports contain aviation maintenance-related reports describing service difficulties and technical problems.

These records are particularly useful for retrieving information related to:

* Aircraft component problems
* Maintenance issues
* Mechanical failures
* Recurring service difficulties
* Aircraft system problems

---

### 3. NTSB Accident Data

The National Transportation Safety Board (NTSB) data was extracted from the NTSB aviation accident database.

```text
NTSB/
├── aircraft_clean.csv
├── events_clean.csv
├── findings_clean.csv
└── narratives_clean.csv
```

The selected tables represent different aspects of aviation accident information:

* **Aircraft** — aircraft-related records
* **Events** — accident/event information
* **Findings** — investigation findings
* **Narratives** — narrative descriptions associated with investigations

These datasets provide detailed accident and investigation information for the AeroLLM knowledge base.

---

### 4. FAA Advisory Circulars

```text
FAA_Advisory_Circulars/
├── AC_120-16G_clean.txt
├── AC_120-72A_clean.txt
├── AC_121-22D_clean.txt
├── AC_20-77B_clean.txt
├── AC_43-12A_clean.txt
└── AC_43-216A_clean.txt
```

FAA Advisory Circulars provide official FAA guidance and technical information.

The selected circulars cover aviation operational, maintenance, safety, and technical guidance relevant to the AeroLLM domain.

The PDF documents were converted into machine-readable text. OCR processing was applied where required to improve text extraction from scanned or poorly encoded documents.

---

### 5. FAA Aviation Maintenance Handbooks

```text
FAA_Handbooks/
├── FAA-H-8083-30B_General_clean.txt
├── FAA-H-8083-31B_Aviation_Maintenance_Technician_Handbook_clean.txt
└── amt_powerplant_handbook_clean.txt
```

These handbooks provide aviation maintenance knowledge covering areas such as:

* General aviation maintenance
* Airframe maintenance
* Powerplant maintenance
* Aircraft systems
* Maintenance procedures
* Technical concepts

The source PDFs were converted into text and cleaned to make them suitable for downstream NLP processing.

---

## Dataset Organization

The complete processed dataset follows this structure:

```text
AeroLLM_Data/
└── processed/
    ├── ASRS/
    ├── FAA_Advisory_Circulars/
    ├── FAA_Handbooks/
    ├── FAA_SDR/
    ├── NTSB/
    └── metadata.csv
```

`metadata.csv` acts as the central index for the processed documents.

It contains information such as:

* `document_id`
* `dataset`
* `document_name`
* `document_type`
* `authority`
* `source_file`
* `status`

This allows the downstream pipeline to identify the source and type of each document before chunking and embedding.

---

## Data Preparation and Cleaning

The dataset was prepared as part of the **AeroLLM data collection and preprocessing stage**.

The main work performed includes:

### Data Collection

Collected aviation-domain datasets and documents from:

* NASA ASRS
* FAA
* NTSB

### Dataset Organization

Separated the collected material into source-specific directories:

```text
ASRS
FAA_SDR
NTSB
FAA_Advisory_Circulars
FAA_Handbooks
```

### PDF Text Extraction

FAA documents and handbooks were converted from PDF format into machine-readable text.

### OCR Processing

OCR was applied to scanned or poorly encoded FAA documents where normal PDF text extraction produced corrupted text.

This improved extraction of technical terms and sentences from the source documents.

### Text Cleaning

The extracted documents were cleaned to remove unnecessary PDF/OCR artifacts and improve readability for downstream NLP processing.

Examples of cleanup included:

* Removing page-number artifacts
* Removing unnecessary formatting
* Reducing extraction noise
* Correcting obvious OCR corruption where identified
* Producing consistent text files for downstream processing

### NTSB Database Extraction

Relevant NTSB database tables were extracted into CSV format so that they can be processed by the RAG pipeline.

### Metadata Creation

A centralized `metadata.csv` file was created to track all processed datasets and their corresponding file paths.

The metadata currently contains **17 processed dataset/document entries**.

---

## Current Dataset Inventory

| Source                 |  Files | Format |
| ---------------------- | -----: | ------ |
| NASA ASRS              |      3 | CSV    |
| FAA SDR                |      1 | CSV    |
| NTSB                   |      4 | CSV    |
| FAA Advisory Circulars |      6 | TXT    |
| FAA Handbooks          |      3 | TXT    |
| **Total**              | **17** |        |

---

## Data Quality Validation

The processed dataset was validated to ensure that:

1. All expected dataset files are present.
2. `metadata.csv` contains the expected 17 dataset entries.
3. Every `source_file` referenced by `metadata.csv` points to an existing processed file.
4. FAA PDF text extraction was checked after OCR processing.
5. NTSB database tables were successfully exported to CSV.

The metadata file contains 18 lines in total:

```text
1 header
17 dataset/document entries
```

---

## Role in the AeroLLM Pipeline

This folder represents the **data collection and preprocessing stage** of the AeroLLM project.

The overall pipeline is:

```text
Official Aviation Sources
        │
        ▼
Data Collection
        │
        ▼
Data Cleaning & Organization
        │
        ▼
AeroLLM_Data
        │
        ▼
Document Chunking
        │
        ▼
Embeddings
        │
        ▼
Vector Database
        │
        ▼
Semantic Retrieval
        │
        ▼
LLM
        │
        ▼
Aviation-domain RAG Answers
```

The current folder completes the **data collection, extraction, cleaning, organization, and metadata stage**.

The next stage can consume the files under `processed/` directly.

---

## Intended Use

The processed datasets are intended to support aviation-domain information retrieval and question answering.

Example questions that the resulting AeroLLM system may eventually answer include:

* What are common causes of aircraft maintenance difficulties?
* What findings are associated with a particular type of aviation accident?
* What FAA guidance applies to a particular maintenance procedure?
* What information is available about a particular aircraft system?
* What patterns appear across aviation incident and accident reports?

The actual answering capability will depend on the subsequent chunking, embedding, retrieval, and LLM stages.

---

## Important Notes

### Raw Data

Raw source files and intermediate processing files are kept separate from the final `processed/` dataset.

The `processed/` directory should be treated as the input dataset for the downstream RAG pipeline.

### Data Provenance

The `metadata.csv` file provides the source authority and document identity for each processed dataset/document.

### Large Files

Some aviation datasets, particularly NTSB narrative data, are relatively large. Repository storage limits should be considered before committing the complete dataset to GitHub.

Where necessary, large datasets should be distributed separately or downloaded through documented source instructions rather than committed directly to Git.

---

## Project Contribution

### Data Collection & Preprocessing

The data preparation work for this component included:

* Collecting aviation-domain datasets
* Organizing datasets by source
* Extracting NTSB database tables
* Converting FAA PDFs into machine-readable text
* Applying OCR to scanned/poorly extracted FAA documents
* Cleaning extracted text
* Organizing the final processed datasets
* Creating and validating `metadata.csv`
* Verifying that all metadata paths point to valid processed files

This prepared the aviation-domain knowledge base for the next stage of the AeroLLM RAG pipeline.

---

## Status

**Dataset Collection & Preprocessing: Completed**

Next stage:

**Chunking → Embeddings → Vector Database → Retrieval → RAG**

