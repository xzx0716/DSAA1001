# DSAA1001 Assessment 1: Chicago Food Inspections

**Submission notebook:** [Assessment_1_EDA.ipynb](Assessment_1_EDA.ipynb).

Predict whether a food inspection will Fail using information available before
the inspection begins. This repository contains the Assessment 1 EDA notebook
and all of its saved inputs. No model has been trained for this submission.

## Contents and A1 Coverage

| Path / notebook section | Purpose |
| --- | --- |
| `Assessment_1_EDA.ipynb` | The single submission notebook at the repository root; executed outputs and Markdown explanations |
| Notebook Section 3 | Eight figure outputs, comparisons, findings, modelling implications and limitations |
| Notebook Section 4 | Evidence-based data-quality table covering 14 issues |
| Notebook Sections 5-6 | Feature availability, establishment overlap, baseline and time-respecting evaluation plan |
| `data/*.csv.gz` | Complete frozen raw snapshot, losslessly compressed |
| `data/download_metadata.json` | Original download source, time, date window, rows and columns |
| `data/documentation/` | Official data dictionary and source-verification record |
| `requirements.txt` | Python dependencies |
| `AI-USAGE.md` | Tools used, assistance scope and limits |

## Run Offline

Use Python 3.11 or later; the verified environment uses Python 3.13. Prepare
dependencies before disconnecting from the network:

```sh
python -m pip install -r requirements.txt
```

Open `Assessment_1_EDA.ipynb` in a Jupyter-compatible editor, select
the prepared Python environment and choose Restart Kernel / Run All.
Use the repository root as the working directory.

The notebook reads the included `.csv.gz` directly with pandas. No manual
decompression, data download, package installation or Git LFS is needed during
Run All. The original CSV exceeds GitHub's ordinary-file limit; gzip preserves
every CSV byte. The notebook verifies the decompressed original SHA-256 before
analysis. The metadata's `csv_file` refers to that original CSV filename.

Offline execution is verified using a fresh kernel with external TCP connections
blocked, including execution from a relocated project directory. This uses the
installed dependency environment rather than a newly provisioned environment.
All analysis code is contained in the notebook; no helper scripts are needed.

## Data Source and Version

- Official source: https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5/about_data
- Dataset ID: `4ijn-s7e5`.
- API: https://data.cityofchicago.org/resource/4ijn-s7e5.json
- Downloaded: **2026-10-25 03:44:36 UTC** / **2026-10-25 11:44:36 Asia/Shanghai**.
- Portal update recorded at download: 2026-10-24 09:08:34 UTC.
- Requested inspection window: 2016-01-01 through 2026-10-25 inclusive.
- Actual inspection dates: **2016-01-04 through 2026-10-23**.
- Raw records: **198,576 rows, 17 columns**.
- Original CSV size: **216,614,489 bytes**.
- Original CSV SHA-256: `6ebba026f5b8f6c21c2b87efac548b14d0dff764175b706482b46e71e0b436ca`.
- Compression changes storage only; no rows, columns or values were removed.

This pre-cutoff snapshot is fixed. The analysis does not download a newer
version. The filename's end date is a requested date, not actual coverage.
Downloading later and filtering inspection dates cannot reproduce which rows
were published at an earlier time. The saved dictionary and verification record
contain source URLs, verification time and the dictionary checksum.

## Supplementary Data and Documentation

**No supplementary dataset was used.** All calculations and figures use only the
saved City of Chicago Food Inspections snapshot. No census, weather, community
boundaries, external licence records or geocoding data were joined or used as
additional features. Historical features are derived from earlier rows of the
same snapshot; the Risk standardisation also uses only that snapshot.

The files in `data/documentation/` are supporting source documentation, not
additional observation datasets. The official dictionary explains field meanings;
`source_verification.json` records a later read-only count/date-range check and
source URLs. Neither adds rows or model features. Both are included because the
written interpretation cites them. Raw inputs read by the notebook are only the
compressed snapshot and its download metadata.

Acquisition reused the instructor's column mapping, CSV conversion and metadata
writer. Temporary download code added gzip transport, 5,000-row pagination,
retries and six concurrent requests after uncompressed requests timed out.
The final row count matched the official API count and inspection IDs were
unique. Neither download code nor network access is needed to reproduce the EDA.

## Labels and Analysis Boundaries

Fail = 1; Pass / Pass w/ Conditions = 0. Other operational outcomes are not
supervised labels, but eligible earlier visits may contribute historical counts.
There are 168,128 labelled visits in the full snapshot, including 37,987 Fail
visits (22.5941%). Duplicate IDs, exact duplicate rows and invalid dates are zero.

Outcome-based EDA uses 2016-2024: 140,762 labelled visits and 32,189 Fail visits
(22.87%). Raw coverage and quality checks use the full frozen snapshot. Prior
drafts explored some later outcomes; the modelling plan discloses this limitation
rather than claiming an untouched internal test. No official hidden-period
outcomes (2026-10-31 through 2026-11-27) were requested or used.

There are 7 NA licences and 464 literal zero licences. The dictionary does not
define zero as a missing code. Unknown identities retain their visits and valid
labels but do not share history. Histories exclude all current-day visits;
historical failure rates use prior labelled visits as their denominator.
Current Results, Violations and their post-start derivatives are not predictors.

## Raw Missing-Value Rates

The denominator is all 198,576 raw rows. Literal licence zero is a separate
identity issue and is not included in `isna()` missing counts.

| Field | Missing count | Missing rate |
| --- | ---: | ---: |
| Inspection ID | 0 | 0.0000% |
| DBA Name | 0 | 0.0000% |
| AKA Name | 498 | 0.2508% |
| License # | 7 | 0.0035% |
| Facility Type | 1,029 | 0.5182% |
| Risk | 39 | 0.0196% |
| Address | 0 | 0.0000% |
| City | 127 | 0.0640% |
| State | 73 | 0.0368% |
| Zip | 12 | 0.0060% |
| Inspection Date | 0 | 0.0000% |
| Inspection Type | 1 | 0.0005% |
| Results | 0 | 0.0000% |
| Violations | 56,661 | 28.5337% |
| Latitude | 690 | 0.3475% |
| Longitude | 690 | 0.3475% |
| Location | 690 | 0.3475% |


## Submission Scope

This is Assessment 1. Model fitting, hidden-test `predictions.csv`, the
commissioner brief and final presentation belong to Assessment 2. The A1
modelling plan is in the notebook rather than a separate report.

The GitHub upload does not submit anything to the course platform. Complete
any platform-specific group identification and submit the required notebook
or repository link before the A1 deadline: **30 October 2026, 23:59**.
