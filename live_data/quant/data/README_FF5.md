# Kenneth French five-factor data

The live Quant provider defaults to:

`F-F_Research_Data_5_Factors_2x3_daily_CSV.zip`

The loader accepts the original Kenneth French daily ZIP directly and converts
percentage values to decimals. It also accepts a normalized CSV with columns:

`date,mkt_rf,smb,hml,rmw,cma,rf`

`rf` is used to form stock excess returns and is never included as a regression
beta. Override the location with the `FACTOR_DATA_PATH` environment variable.

When updating the data, rerun the complete backtest from January 2025. Do not mix
old JKP-based months with FF5-based months in one performance comparison.
