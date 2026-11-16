# swot-paper-repro

Figure generators and evaluation scripts for "SWOT impact on the end-to-end neural forecasting system" (Botvynko et al.).
The data paths point to the authors' cluster; the evaluation tables come from the OceanBench-based `oceanbench_eval` scripts.

- `sla/make_gain_maps.py`: map of the change in SLA RMSE with SWOT input per 2° box (fig-gain-maps).
- `currents/drifter_cyclo_eval.py`: hourly 15 m drifter points (48 h Butterworth low-pass), geostrophic and
  cyclogeostrophic (jaxparrow) currents of the forecasts, complex-ratio statistics (after Uchida et al. 2026, GRL).
- `currents/make_current_figures.py`: paper polar histograms (`polar_paper`: fig-currents-polar-swath/-boundary, systems in columns, Nadir and Nadir+SWOT rows), review PNGs per subset (`polar_by_subset`), change in RMS vector error against leadtime (`lead`, shared y axis), drifter rotary spectra (`rotary`). Run with /Odyssey/private/d21botvy/venvs/cyclo/bin/python: `python make_current_figures.py <paper figures dir>`.
- `currents/run_match.sh`: GPU batch job for the matching and analysis.
