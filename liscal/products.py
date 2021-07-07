import xarray as xr
import pandas as pd

from liscal import binary_scores, hydro_stats


def create_products(cfg, subcatch, obj):

    # Long term run has run_id X
    run_id = 'X'

    # compute statistics (KGE, NSE, etc.)
    Q, stats = obj.compute_statistics(run_id)

    # compute monthly discharge data
    sim_monthly, obs_monthly = hydro_stats.split_monthly(Q.index.values, Q['Qsim'].values, Q['Qobs'].values, spinup=subcatch.spinup)

    # get return periods at station coordinates
    thresholds = xr.open_dataset(cfg.return_periods).sel(x=subcatch.data['LisfloodX'], y=subcatch.data['LisfloodY'])
    print(thresholds)

    # compute contingency table and export
    contingency_values = binary_scores.contingency_table(thresholds, Q)
    contingency_df = pd.DataFrame(data=contingency_values, index=subcatch.obsid)
    print(contingency_df)
    contingency_df.to_csv(path.join(subcatch.path_out, 'contingency_table.csv'))
