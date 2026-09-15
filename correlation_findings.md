# Correlation Findings

## Overview

This report summarizes the relationships found in the Beijing air-quality dataset across pollutant concentrations, weather measurements, and monitoring stations.

Correlation values range from -1 to 1:

- Values near 1 indicate a strong positive relationship.
- Values near -1 indicate a strong negative relationship.
- Values near 0 indicate a weak or limited linear or ranked relationship.

Correlation shows association, not causation.

## Pearson Correlation

Pearson correlation measures linear relationships among the pollutants and weather measurements.

Key findings:

- PM2.5 and PM10 have a very strong positive relationship: **0.884**.
- PM2.5 and CO have a strong positive relationship: **0.790**.
- PM2.5 and NO2 have a moderately strong positive relationship: **0.667**.
- NO2 and CO have a strong positive relationship: **0.705**.
- O3 and NO2 have a moderate negative relationship: **-0.472**.
- O3 and temperature have a strong positive relationship: **0.595**.
- Temperature and pressure have a very strong negative relationship: **-0.813**.
- Temperature and dew point have a very strong positive relationship: **0.820**.
- PM2.5 and wind speed have a weak-to-moderate negative relationship: **-0.272**.

The Pearson heatmap is saved as `PRSA_Data_20130301-20170228/pearson_correlation_heatmap.png`.

## Spearman Correlation

Spearman correlation measures ranked, monotonic relationships and can capture relationships that are not strictly linear.

Key findings:

- PM2.5 and PM10 have a very strong positive ranked relationship: **0.891**.
- PM2.5 and CO have a strong positive ranked relationship: **0.837**.
- PM2.5 and NO2 have a moderately strong positive ranked relationship: **0.658**.
- NO2 and CO have a strong positive ranked relationship: **0.741**.
- O3 and NO2 have a strong negative ranked relationship: **-0.614**.
- O3 and wind speed have a moderate positive ranked relationship: **0.427**.
- Temperature and pressure have a very strong negative ranked relationship: **-0.819**.
- Temperature and dew point have a very strong positive ranked relationship: **0.817**.
- PM2.5 and wind speed have a moderate negative ranked relationship: **-0.329**.

Compared with Pearson correlation, the stronger Spearman values for PM2.5 and CO and for O3 and NO2 suggest that rankings and possible non-linear patterns are important in these relationships.

The Spearman heatmap is saved as `PRSA_Data_20130301-20170228/spearman_correlation_heatmap.png`.

## Strongest Relationships

The strongest unique Pearson relationships are:

| Variables | Pearson correlation |
|---|---:|
| PM2.5 and PM10 | 0.884 |
| Temperature and dew point | 0.820 |
| Temperature and pressure | -0.813 |
| PM2.5 and CO | 0.790 |
| Pressure and dew point | -0.750 |
| NO2 and CO | 0.705 |
| PM10 and CO | 0.702 |
| PM2.5 and NO2 | 0.667 |
| PM10 and NO2 | 0.652 |
| O3 and temperature | 0.595 |

The strongest pollutant relationship is PM2.5 with PM10. This indicates that particulate pollution levels tend to rise and fall together across the combined dataset.

The strongest weather relationships are temperature with dew point and temperature with pressure. These are expected atmospheric relationships and should not be interpreted as pollution effects.

The chart is saved as `PRSA_Data_20130301-20170228/strongest_relationships.png`.

## PM2.5 and Other Pollutants by Station

PM2.5 has a strong positive relationship with PM10 at every station, ranging from **0.855** at Gucheng to **0.904** at Nongzhanguan.

PM2.5 and CO also show strong positive relationships at every station, ranging from **0.750** at Wanliu to **0.814** at Nongzhanguan and Wanshouxigong.

PM2.5 and NO2 show moderately strong positive relationships, ranging from **0.644** at Shunyi to **0.718** at Dingling.

PM2.5 and SO2 show moderate positive relationships, ranging from **0.397** at Tiantan to **0.547** at Dongsi.

PM2.5 and O3 show weak-to-moderate negative relationships at every station, ranging from **-0.064** at Huairou to **-0.191** at Wanshouxigong.

Overall, the station-level results are consistent across locations:

- PM10 is the pollutant most consistently associated with PM2.5.
- CO is the second strongest pollutant relationship with PM2.5.
- O3 generally moves in the opposite direction from PM2.5.
- The strength of the relationships varies by station, but the overall pattern remains similar.

The station heatmap is saved as `PRSA_Data_20130301-20170228/pm25_station_correlation_heatmap.png`.

## Conclusion

Particulate pollutants are strongly associated with one another, especially PM2.5, PM10, and CO. Nitrogen dioxide also follows the particulate pollution pattern. Ozone behaves differently, showing negative relationships with PM2.5, NO2, and CO. Weather variables have substantial relationships with one another, particularly temperature, pressure, and dew point, so they should be considered when interpreting pollutant correlations.
