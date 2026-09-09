import pandas as pd
from pathlib import Path
import json

INPUT_FILE = Path("processed/faa_sdr_clean.csv")
OUTPUT_FILE = Path("rag/data/documents/faa_sdr_documents.jsonl")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 10000

print("Starting FAA SDR preprocessing...")
print(f"Input: {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")

total = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for chunk_no, df in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )
    ):

        print(f"Processing chunk {chunk_no + 1}...")

        for _, row in df.iterrows():

            def clean(value):
                if pd.isna(value):
                    return "Not available"
                return str(value).strip()

            report_id = clean(row["id"])

            document = f"""
FAA SERVICE DIFFICULTY REPORT

Report ID: {report_id}
Source: {clean(row["source"])}
Date: {clean(row["DifficultyDate"])}
Year: {clean(row["Year"])}

AIRCRAFT
Make: {clean(row["AircraftMake"])}
Model: {clean(row["AircraftModel"])}

SYSTEM
JASC Code: {clean(row["JASCCode"])}

COMPONENT
Part: {clean(row["PartName"])}
Part Number: {clean(row["PartNumber"])}
Component: {clean(row["ComponentName"])}
Component Part Number: {clean(row["ComponentPartNumber"])}

CONDITION
{clean(row["NatureOfConditionA"])}

PROCEDURE
{clean(row["PrecautionaryProcedureA"])}

DISCREPANCY
{clean(row["Discrepancy"])}

NARRATIVE
{clean(row["narrative"])}
""".strip()

            record = {
                "id": report_id,
                "text": document,
                "metadata": {
                    "source": "FAA_SDR",
                    "date": clean(row["DifficultyDate"]),
                    "year": clean(row["Year"]),
                    "aircraft_make": clean(row["AircraftMake"]),
                    "aircraft_model": clean(row["AircraftModel"]),
                    "jasc_code": clean(row["JASCCode"]),
                    "part_name": clean(row["PartName"]),
                    "part_number": clean(row["PartNumber"]),
                    "component_name": clean(row["ComponentName"]),
                    "component_part_number": clean(
                        row["ComponentPartNumber"]
                    )
                }
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1

        print(f"Records processed so far: {total}")

print("\n================================")
print("PREPROCESSING COMPLETE")
print("================================")
print(f"Total documents: {total}")
print(f"Output file: {OUTPUT_FILE}")