# Beijing Air Quality and Weather Data Sources

The full downloaded source collection is kept locally in
`CNEMC_Figshare_2014-2025/`. The repository tracks only the Beijing
station-hour pollutant CSV and daily ERA5 feature CSV listed below; the other
large source and intermediate files remain local because they exceed
GitHub's normal repository file limits.

## Pollutant measurements

The source is the Figshare repository for *Data repository for “A decade of
China’s air quality monitoring data suggests health impacts are no longer
declining”* by Silver, Chen, Reddington, and Arnold (2025):

- [Figshare dataset (version 1)](https://doi.org/10.6084/m9.figshare.25689477.v1)

The source data are licensed CC BY 4.0. Cite the dataset and article when
reusing the data. The local Beijing station-hour CSV is a convenience export
from the protocol-adjusted Parquet data and covers May 13, 2014 through
January 4, 2025.

## Weather data

The local hourly weather data are ERA5 estimates retrieved for the project
station locations using the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
with the ERA5 model selected. They cover March 1, 2013 through January 4,
2025. These are reanalysis estimates, not direct measurements at the air
quality monitors.

Suggested citations:

- Zippenfenig, P. (2023). [Open-Meteo.com Weather API](https://doi.org/10.5281/zenodo.7970649).
- Hersbach, H. et al. (2023). [ERA5 hourly data on single levels from 1940 to present](https://doi.org/10.24381/cds.adbb2d47).

The local daily weather feature table is derived from the hourly ERA5 export.
The station-to-grid mapping is in the local `metadata/` folder.
