# EDA and Preprocessing

Experiments, one card each: [exp01_overview](experiments/exp01_overview.md), [exp02_time_zones](experiments/exp02_time_zones.md), [exp03_metric](experiments/exp03_metric.md), [exp04_anomalies](experiments/exp04_anomalies.md). [data_aggregation.ipynb](notebooks/data_aggregation.ipynb) builds the training and test tables.

## Data insights

Details per experiment are in [EDA step by step](#eda-step-by-step).

- 20.2M hourly readings from 2,380 building x meter series. 875 of 1,449 buildings have only one meter, almost always electricity.
- The target is very skewed: one steam meter (building 1099) has 78% of all energy. On a log scale it looks fine.
- Zeros are common: 27% of hot water readings, 16% of chilled water, 13% of steam and 4% of electricity.
- Chilled water peaks in summer, steam and hot water in winter. Use drops at night and at weekends.

## Metric

RMSLE measures relative error: a 10% miss costs about 0.095 at any scale. But it is harsh near zero: predicting 5 for a true 0 costs 1.79 ([exp03_metric](experiments/exp03_metric.md)).

Good:

- big buildings don't dominate the score
- it's plain MSE on `log1p`, so we can train on the metric directly

Bad:

- zeros are expensive, and many zeros are broken meters, so partly the score is about guessing outages
- it doesn't care whether errors cancel out over a day or a year
- negative predictions are undefined (we clip at 0)
- a unit mix-up becomes a big constant error (site 0 electricity is in kBTU)

Extra metrics we track:

- RMSLE on clean rows, to see the error on real consumption ([exp02_seasons](../validation/experiments/exp02_seasons.md))
- RMSLE per meter type, since electricity and hot water differ by 0.6 ([exp01_outputs](../mlp/experiments/exp01_outputs.md))
- spread over the four folds: a change must help in at least 3 of 4 seasons
- residual correlation between models, to see how much an ensemble can gain

Alternative target metric: CV(RMSE) per building x meter series, averaged over series. The competition is about estimating energy savings after retrofits, which needs errors in kWh. ASHRAE Guideline 14 uses CV(RMSE) for this. Averaging per series keeps big buildings from dominating, and zeros don't blow it up. We haven't computed it on our models.

## Preprocessing

We based our data cleaning on three prior write-ups:

- the data-cleaning approach of a top ASHRAE solution ([1](../README.md#references))
- the anomaly-correction approach of a BirdCLEF solution ([2](../README.md#references))
- site and building identification via internet search ([3](../README.md#references))

With Handout validation this looked like a big boost, but with the 4-fold seasonal validation it is only a small change ([exp02_building](../mlp/experiments/exp02_building.md)).

### Local time zone correction

As noted on the [competition forum](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/115698), weather timestamps are UTC, while meter timestamps are local. We checked it on our data: after shifting weather to each site's local time, temperature peaks at 14:00–16:00 everywhere. Site 14 is the exception, its meters are in UTC too ([exp02_time_zones](experiments/exp02_time_zones.md)).

The fix improved the LightGBM baseline from 1.290 to 1.286, and no fold got worse ([exp04_anomalies](experiments/exp04_anomalies.md)).

### Remove anomalies

Some meters are broken: zeros for long periods, spikes, the same value for days, etc. We remove three kinds of rows ([exp04_anomalies](experiments/exp04_anomalies.md)):

- every zero electricity reading, since a building always uses some power
- steam or hot water at 0 for a week or more while the week is cold (below 10 °C)
- chilled water at 0 for a week or more while the week is warm (above 20 °C)

This is 599,086 rows, 3.0% of the training data. We drop them from training only and still validate on all rows, because the Kaggle test set has broken meters as well ([data_aggregation.ipynb §4](notebooks/data_aggregation.ipynb)).

On clean rows this helps a lot: LightGBM goes from 1.140 to 1.048. On all rows, as Kaggle scores, it barely moves (1.286 → 1.281). Jan–Mar and Apr–Jun actually get worse, because a model that learned a meter's zeros wins when the outage goes on into the validation months. We still drop the anomalies.

### Removed features

- floor count: missing for 83% of rows ([data_aggregation.ipynb §3](notebooks/data_aggregation.ipynb))
- precipitation and cloud coverage: often missing and barely related to the target ([exp01_overview](experiments/exp01_overview.md))

### Filled-in features

- year_built: filled with the training median, plus a "missing" flag, since 60% of rows don't have it ([data_aggregation.ipynb §3](notebooks/data_aggregation.ipynb))
- air / dew temperature and sea level pressure: interpolated linearly over time within each site. A site that never reports a column (site 5 has no pressure) gets the mean of the other sites at the same hour ([data_aggregation.ipynb §2](notebooks/data_aggregation.ipynb))
- wind direction: turned into sin/cos, then interpolated linearly ([ashrae/features.py](../ashrae/features.py))

### Feature Engineering

The same features are built for training, validation and test ([ashrae/features.py](../ashrae/features.py)):

- `log1p(square_feet)`: floor area is the strongest building feature (ρ = 0.68 with electricity) ([exp01_overview](experiments/exp01_overview.md))
- weather: air and dew temperature, pressure, wind speed and direction (sin/cos)
- hour and day of week as sin/cos, in local time
- one-hot `meter` and `primary_use`
- building identity: either a learned 16-number embedding per building, or the mean `log1p` reading of each building x meter series ([exp02_building](../mlp/experiments/exp02_building.md))

### Target transformations

- We predict `log1p(meter_reading)` with MSE loss. RMSLE is just RMSE on `log1p`, so the loss is the metric ([exp03_metric](experiments/exp03_metric.md)). Also, `log1p` of the reading is bell-shaped, which is a good sign.
- Predictions are clipped at 0 ([mlp/src/predict.py](../mlp/src/predict.py)).
- Site 0 electricity is converted from kBTU to kWh for training and back for the submission ([data_aggregation.ipynb §1](notebooks/data_aggregation.ipynb)).

## EDA step by step

Each part: what we found → what it changes, with the card it comes from.

### Data overview ([exp01_overview](experiments/exp01_overview.md))

- 2016, hourly: 20.2M readings from 2,380 building × meter series.
- 875 of 1,449 buildings have one meter, almost always electricity (60% of rows).
- → Keep one row per building × meter × hour; a column per meter type would be mostly empty.

### Data quality ([exp01_overview](experiments/exp01_overview.md))

- Missing: `floor_count` 76% and `year_built` 53% of buildings; cloud coverage 49% and precipitation 36% of weather
  rows. Site 5 never reports pressure. Up to 330 hours missing at a site. No duplicates.
- → Fill weather gaps within each site; use other sites only when a site has no data for a column.

### Building metadata ([exp01_overview](experiments/exp01_overview.md))

- Education (549) and office (279) dominate. Floor area is bell-shaped on a log scale.
- Missing `year_built` / `floor_count` depends on the site: whole sites report none.
- → Use `log1p(square_feet)`.

### Target ([exp01_overview](experiments/exp01_overview.md))

- Extremely skewed: one steam meter (building 1099) holds 78% of all energy. Well-behaved on a log scale.
- Zeros: 27% of hot water, 16% of chilled water, 13% of steam, 4% of electricity readings.
- → Target is `log1p(meter_reading)`.

### Temporal patterns ([exp02_time_zones](experiments/exp02_time_zones.md))

- Chilled water peaks in summer, steam and hot water in winter. Use is lower at night and at weekends.
- → Hour and day-of-week features; the model must combine meter type with weather.

### Time zones ([exp02_time_zones](experiments/exp02_time_zones.md))

- Weather timestamps are UTC, meter timestamps local (site 14: UTC too). Checked on our data: after shifting, the
  temperature peaks at 14:00–16:00 at every site.
- → Shift weather by the site's UTC offset before merging (not at site 14). LightGBM, 4-fold average: 1.290 → 1.286
  (clean rows 1.147 → 1.140); no fold gets worse.

### Features vs target ([exp01_overview](experiments/exp01_overview.md))

- Floor area is the strongest building feature (Spearman ρ = 0.68 for electricity).
- Temperature: chilled water up (ρ = 0.43), steam and hot water down (ρ ≈ −0.45).
- → Drop precipitation and cloud coverage: barely related to the target, often missing.

### Metric ([exp03_metric](experiments/exp03_metric.md))

- RMSLE = RMSE of `log1p` values → train on `log1p` with MSE loss.
- Relative error: a 10% miss costs ≈ 0.095 at any scale. Harsh near zero: predicting 5 for a true 0 costs 1.79.
- A few wrong zeros can dominate the score. Also track RMSLE per meter and without anomaly rows.
- Alternative metric: MAE of `log1p`

### Anomalies ([exp04_anomalies](experiments/exp04_anomalies.md))

- Zero electricity = broken meter or data gap. Heating at zero for a week in the cold, or cooling in the heat, too.
- → Drop them from training only. LightGBM, 4-fold average on clean rows: 1.140 → 1.061 without zero electricity,
  → 1.048 with both rules, better in every fold. Shorter cut-offs work better: even a few hours of zeros teach the
  model that a building can use no power.
- On all rows the average barely moves (1.286 → 1.281): Jul–Dec improves, Jan–Jun gets worse, because there the
  broken meters stay broken into the validation months. Oct–Dec alone (1.383 → 1.168) overstated the gain.
- Anomaly rows stay in validation and cost up to 0.48 RMSLE per fold
  ([exp02_seasons](../validation/experiments/exp02_seasons.md)).

### What didn't work

- Two more anomaly rules: stuck meters (same non-zero value for 48+ hours) and spikes (> 10× the daily median). On
  top of both rules they change the 4-fold average by 0.003 at most; the spike rule mostly flagged normal on/off
  cycling. Both left out.

### What was removed from the notebook

- Summary tables (per-column statistics, per-meter percentiles), the `year_built` histogram, a box plot by building
  use (boxes overlapped too much), and per-site tables already shown by the time-zone plot.
