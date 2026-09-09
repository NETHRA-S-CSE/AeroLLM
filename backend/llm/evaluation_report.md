# AeroLLM RAG -> LLM Grounding & Faithfulness Evaluation Report

## Overview
- **Evaluated Queries**: 10
- **Average Top-1 Similarity Score**: 0.5679
- **Report ID Grounding Rate**: 100.0%
- **Aircraft Detail Grounding Rate**: 100.0%
- **Component Detail Grounding Rate**: 100.0%
- **Schema Compliance Rate**: 100.0%
- **Overall Grounding Pass Rate**: 100.0%

## Detailed Query Evaluation Results

| # | Topic | Query | Top Similarity | Cited Report IDs | Valid IDs? | Grounded? |
|---|---|---|---|---|---|---|
| 1 | Hydraulic Systems | hydraulic system pressure loss during flight | 0.5338 | FAA_SDR_1351, FAA_SDR_2769, FAA_SDR_7251, FAA_SDR_4530, FAA_SDR_6179 | Yes | PASS |
| 2 | Engine Vibration | engine vibration and component replacement | 0.4996 | FAA_SDR_6456, FAA_SDR_6599, FAA_SDR_3495, FAA_SDR_5106, FAA_SDR_6572 | Yes | PASS |
| 3 | Electrical Failures | generator control unit electrical power failure | 0.5390 | FAA_SDR_5804, FAA_SDR_5943, FAA_SDR_5048, FAA_SDR_2344, FAA_SDR_3380 | Yes | PASS |
| 4 | Emergency Lighting | emergency exit lighting inoperative and battery replacement | 0.6353 | FAA_SDR_6556, FAA_SDR_9429, FAA_SDR_6232, FAA_SDR_9245, FAA_SDR_6018 | Yes | PASS |
| 5 | Component Replacement | fuel pump failure and component replacement | 0.6087 | FAA_SDR_8306, FAA_SDR_9852, FAA_SDR_9633, FAA_SDR_1109, FAA_SDR_7169 | Yes | PASS |
| 6 | Fluid Leakage | hydraulic fluid leakage from main landing gear line | 0.5360 | FAA_SDR_1684, FAA_SDR_3652, FAA_SDR_6461, FAA_SDR_4667, FAA_SDR_1351 | Yes | PASS |
| 7 | Pressure Problems | cabin altitude pressure warning and valve inspection | 0.5901 | FAA_SDR_2782, FAA_SDR_5292, FAA_SDR_7761, FAA_SDR_5259, FAA_SDR_5828 | Yes | PASS |
| 8 | Wiring Problems | aircraft electrical wiring chafing and short circuit | 0.5019 | FAA_SDR_6520, FAA_SDR_3964, FAA_SDR_3700, FAA_SDR_4998, FAA_SDR_9714 | Yes | PASS |
| 9 | Aircraft Systems | flight control elevator trim system actuator malfunction | 0.6591 | FAA_SDR_7846, FAA_SDR_4323, FAA_SDR_7850, FAA_SDR_6647, FAA_SDR_6446 | Yes | PASS |
| 10 | Maintenance Observations | maintenance observation of crack in wing spar fitting | 0.5755 | FAA_SDR_2772, FAA_SDR_7627, FAA_SDR_4812, FAA_SDR_4742, FAA_SDR_1593 | Yes | PASS |
