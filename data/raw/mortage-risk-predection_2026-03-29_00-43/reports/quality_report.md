# Data Quality Report: merged_dataset.csv

- Rows: **624**
- Columns: **40**
- Issue types detected: **2**

## Missing Values

|                                |   missing |
|:-------------------------------|----------:|
| submitted                      |       614 |
| state                          |       614 |
| links.self_iiif_sequence       |       614 |
| links.versions                 |       614 |
| metadata.access_right          |       614 |
| metadata.creators              |       614 |
| metadata.doi                   |       614 |
| metadata.license.id            |       614 |
| stats.views                    |       614 |
| metadata.resource_type.title   |       614 |
| metadata.resource_type.type    |       614 |
| metadata.title                 |       614 |
| modified                       |       614 |
| owners                         |       614 |
| metadata.relations.version     |       614 |
| updated                        |       614 |
| id                             |       614 |
| recid                          |       614 |
| revision                       |       614 |
| stats.downloads                |       614 |
| stats.unique_downloads         |       614 |
| stats.unique_views             |       614 |
| stats.version_downloads        |       614 |
| stats.version_unique_downloads |       614 |
| stats.version_unique_views     |       614 |
| stats.version_views            |       614 |
| conceptrecid                   |       614 |
| Credit_History                 |        60 |
| Self_Employed                  |        42 |
| LoanAmount                     |        32 |
| Dependents                     |        25 |
| Gender                         |        23 |
| Married                        |        13 |
| CoapplicantIncome              |        10 |
| ApplicantIncome                |        10 |
| Property_Area                  |        10 |
| Loan_ID                        |        10 |
| Education                      |        10 |

## Duplicates

Duplicate rows: **0**

## Outliers (IQR)

|                   |   count |    lower |    upper |
|:------------------|--------:|---------:|---------:|
| revision          |       1 |     2.5  |     6.5  |
| CoapplicantIncome |      18 | -3445.88 |  5743.12 |
| ApplicantIncome   |      50 | -1498.75 | 10171.2  |
| LoanAmount        |      39 |    -2    |   270    |
| Credit_History    |      89 |     1    |     1    |

