# Calibration sources

| Input | Value used | Source | Status |
|---|---|---|---|
| Planning-area resident population | 28 residential planning areas, June 2026 | Singapore Department of Statistics, *Singapore Residents by Planning Area / Subzone, Age Group, Sex and Floor Area of Residence, June 2026* (figures as tabulated on the Wikipedia "Planning areas of Singapore" page; top five cross-checked against SingStat's Statistics Singapore Newsletter Issue 2, 2026) | Calibrated |
| Planning-area -> region mapping | URA regions | Same table | Calibrated |
| Dwelling-type shares | HDB 1-2rm 7.3%, 3rm 16.6%, 4rm 31.2%, 5rm&Exec 22.1%, Condo 17.9%, Landed 4.7% (others 0.2% dropped) | SingStat, General Household Survey 2025 / Resident Households latest data | Calibrated (household shares used as a proxy for customer shares) |
| GST | 9% | IRAS (rate since 1 Jan 2024) | Fact |
| Public holidays | `holidays.Singapore` | Python `holidays` package | Library |
| CNY dates | 2023-01-22, 2024-02-10, 2025-01-29 | Calendar | Fact |
| Planning-area lat/lon | Approximate town-centre coordinates | Author's approximation, for Tableau symbol maps only | Approximate: verify or use URA MP19 boundaries from data.gov.sg |
| Age-band weights | Residents aged 18-64, June 2026: 18-24 10.9%, 25-34 21.4%, 35-44 23.7%, 45-54 22.5%, 55-64 21.6% | SingStat, *Singapore Residents by Age Group, Ethnic Group and Sex, End June* (Table M810011, the Population Trends T3 series), via data.gov.sg dataset d_3cf667d761b4bdc6d4d3d3aeec37dea5. The 19 five-year groups transcribed in `sg_residents_age_2026_5yr.csv` sum exactly to SingStat's published total of 4,231,524 | Calibrated. **Two assumptions remain:** (1) 18-19 is approximated as 2/5 of the 15-19 group; (2) the app base is restricted to ages 18-64 (65+ are 19.5% of residents but assumed not to use the app). `adoption_index` in `sg_age_2026.csv` is 1.0; lower it for older groups to model a younger user base |
| Basket multiplier by housing type | 0.8 / 0.9 / 1.0 / 1.2 / 1.4 | Assumed income proxy | **Assumption** |
| Purchase frequency, lifetime, price sensitivity, campaign response | Simulation parameters | Not observable in public data | **Assumption** |
