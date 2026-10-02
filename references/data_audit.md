# Local FI-TW data audit

The local archive contains 84 monthly files for 2018–2024, plus later 2025 files that were excluded. The 2018–2024 network files contain 41,672,120 rows, exceeding the paper's approximately 38.5 million. The network-wide totals differ; the author cohort can nevertheless be matched exactly after reproducing its column projection and deduplication.

## Exact cohort and row counts

Oulu central station is `stationShortCode == "OL"`, station UIC 370, stationName `Oulu asema`. Use `trainCategory == "Long-distance"`. Both `ARRIVAL` and `DEPARTURE` are required. Do not include other Oulu freight and technical station codes or Oulunkylä.

| Rule | Rows |
|---|---:|
| All Oulu long-distance events in 2018–2024 departure-month files | 102,217 |
| Nonmissing differenceInMinutes | 101,452 |
| Nonmissing and parsable actualTime | 101,148 |
| ActualTime present and target present | 101,148 |
| ActualTime present and cancelled=False | 101,147 |
| ActualTime present and stop_cancelled=False | 101,123 |
| Target present, cancelled=False, stop_cancelled=False | 101,137 |
| ActualTime valid + author retained-column deduplication | 101,146 |
| Paper reported | 101,146 |

The author actualTime cleaning plus deduplication on retained model-table columns reproduces exactly 101,146 observations. The projection contains actualTime, trainStopping, commercialStop, actualTime-derived month/hour/day-of-week/day-of-month, trainNumber, 11 raw weather fields, the three numeric delay targets, and cancelled, while omitting event type and scheduledTime. It removes two arrival/departure pairs that have the same actualTime and all retained features/targets: train 11942 on 2021-01-09 at 14:28:26 UTC and train 10349 on 2024-08-21 at 18:38:28 UTC. Deduplication before projection leaves both because event type differs. Omitting actualTime before deduplication would incorrectly remove 28 rows. All nonmissing actualTime values are ISO8601 UTC strings of length 24 and parse successfully; all have nonmissing target. There are no duplicates on the dated train/station/event-type/scheduled-time key, or on dated train/event-type/actual-time. There are nine arrival/departure pairs at the same actualTime, which are distinct events.

304 rows have a nonmissing target despite missing actualTime. Examples contain `estimateSource="LIIKE_AUTOMATIC"` and `liveEstimateTime`: their delay labels may represent estimates. There are two `unknownDelay=True` rows, both missing actualTime.

## Time and identifiers

Both scheduledTime and actualTime are UTC ISO strings. Monthly files are grouped by train `departureDate`, not necessarily Oulu event date. The selected scheduledTime range runs from 2018-01-01 04:54 UTC through 2025-01-01 06:26 UTC. Eight scheduled events fall after UTC midnight on January 1, 2025; eleven fall after Finnish local midnight. Keeping the author departure-month selection retains those boundary events. Filtering by event calendar years changes the count.

`trainNumber` contains 344 unique values (20–69,053) and is reused across departure dates. There are 76,006 distinct `(departureDate, trainNumber)` service runs. No raw `train_id` column exists: its exact derivation must come from author source or be recorded as an assumption. The target agrees with `(actualTime-scheduledTime)/60` to within half-minute rounding.

## Weather fields and missingness

No engineered cyclic time, trainDelayed, or weather-category columns exist in the raw parquet. They must be rebuilt. All source records already carry matched raw weather and 12/24/72-hour aggregate weather; rejoining weather is unnecessary. The exact source pressure name is `Pressure (msl)`.

Raw Oulu missingness out of 102,217 rows:

| Field | Missing |
|---|---:|
| Air temperature | 0 |
| Wind speed | 0 |
| Gust speed | 0 |
| Wind direction | 0 |
| Relative humidity | 0 |
| Dew-point temperature | 0 |
| Precipitation intensity | 0 |
| Pressure (msl) | 0 |
| Snow depth | 14 |
| Horizontal visibility | 29 |
| Cloud amount | 20,248 |
| Precipitation amount | 89,409 |
| closest_ems | 83,904 |

Cloud amount is entirely missing through May 2019, then broadly available. Thus the paper's statement that cloud was unavailable at Oulu does not describe this full archive snapshot. Precipitation amount is only sparsely available but its rolling aggregate fields are complete. `closest_ems` is either missing or Oulu Kaukovainio; raw weather remains complete in many records where station provenance is missing, so do not filter on closest_ems.

The schemas have the same base fields, some column reordering across months, and an additional mostly-null unknownDelay field in some months. A column-selective monthly reader avoids schema-order assumptions.

## Artifacts

- `data/oulu_raw.parquet`: 102,217 raw filtered records with required weather, flags, service IDs, and timestamps; source filename column `file`.
- `references/data_audit.json`: full monthly counts, canonical schema, variant differences, flags, and missingness.
- `references/data_filter_audit.json`: filter-combination counts near the reported cohort.
- `references/duplicate_audit.json`: exact retained-column duplicate projection and both removed pairs.
- `tmp/data_audit.py` and `tmp/data_filter_audit.py`: read-only source audit scripts.
