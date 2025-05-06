import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

ver = sys.version
ver = ver[:ver.find('(')-1]
if ver.find('3.') > -1:
  from configparser import ConfigParser as Parser # Python 3.8
else:
  from ConfigParser import SafeConfigParser as Parser # Python 2.7-15

# defines the plot groupings
plot_groupings = [
      [['rainUps', 'snowUps'], ['snowMeltUps'], ['frostUps'], ['actEvapo']], # plot 1 has 4 subplots
      [['dTopToSubUps'], ['qUzUps'], ['qLzUps'], ['percUZLZUps'], ['dSubToUzUps']],
      [['prefFlowUps'], ['infUps'], ['surfaceRunoffUps'], ['gwLossUps'], ['lzUps', 'uzUps']],
      [['th1aAvUps'], ['th1bAvUps'], ['th2AvUps']]
  ]

def construct_dfs(base_path, catchment_id, plot_groupings):
  observations_df = pd.read_csv(f"{base_path}/{catchment_id}/station/observations.csv")
  observations_df.columns = ['time', "dis"]
  observations_df['time'] = pd.to_datetime(observations_df['time'], format='%d/%m/%Y %H:%M')
  stations_info_df = pd.read_csv(f"{base_path}/{catchment_id}/station/station_data.csv", index_col=0).T
  stats_df = pd.read_csv(f"{base_path}/{catchment_id}/pHistoryWRanks.csv", index_col=0)
  simulations_df = pd.read_csv(f"{base_path}/{catchment_id}/out/streamflow_simulated_best.csv")
  simulations_df.columns = ['time', "dis"]
  simulations_df['time'] = pd.to_datetime(simulations_df['time'], format='%d/%m/%Y %H:%M')

  simulation_other_vars_path=f"{base_path}/{catchment_id}/out/long_term_run/"

  for plot_vars in plot_groupings:
      for subplot_vars in plot_vars:
          for var in subplot_vars:
              df = pd.read_csv(simulation_other_vars_path+f"{var}.tss", sep="\s+", header=None, skiprows=4, names=['step', var])
              df.drop(columns=['step'], inplace=True)
              if not df.index.equals(simulations_df.index):
                  raise ValueError("Indexes do not match. Cannot concatenate.")
              simulations_df = pd.concat([simulations_df, df], axis=1)
  
  return observations_df, simulations_df, stations_info_df, stats_df # decide if we use stats_df or not

def discharge_plot(observations_df, simulations_df, stations_info_df, save=True):
  fig = go.Figure()
  fig.add_trace(go.Scatter(
      x=simulations_df['time'],
      y=simulations_df['dis'],
      mode='lines',
      name='model'
  ))
  fig.add_trace(go.Scatter(
      x=observations_df['time'],
      y=observations_df['dis'],
      mode='lines',
      name='obs'
  ))
  fig.update_layout(
      title="{}, {}".format(stations_info_df['StationName'].values[0], stations_info_df['Country code'].values[0]),
      xaxis_title='Time',
      yaxis_title='Discharge',
      legend=dict(title='Datasets'),
      width=800,
      height=500
  )
  if save:
    fig.write_html(f"catchment{catchment_id}_dis.html")
  return fig

def other_var_plots(simulations_df, stations_info_df, plot_groupings, save=True):
  figs = []
  for plot_vars in plot_groupings:
    num_plots = len(plot_vars)
    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True)
    for i, subplot_vars in enumerate(plot_vars, start=1):
        for var in subplot_vars:
            fig.add_trace(
                go.Scatter(x=simulations_df['time'], y=simulations_df[var], mode='lines', name=var),
                row=i, col=1
            )
        fig.update_layout(
            **{f'yaxis{i}': dict(title=",".join(subplot_vars))}
        )
    fig.update_layout(
        title="{}, {}".format(stations_info_df['StationName'].values[0], stations_info_df['Country code'].values[0]),
        legend=dict(title='Variables'),
        width=1000,
        height=800,
        **{f'xaxis{num_plots}': dict(title="Time")},
    )
    figure_name = "_".join([
        var
        for subplot_vars in plot_vars
        for var in subplot_vars
    ]) + ".html"
    if save:
      fig.write_html(f"catchment{catchment_id}_{figure_name}")
    figs.append(fig)
  return figs

if __name__ == "__main__":
  parser = Parser()
  settings_file = os.path.normpath(sys.argv[1])
  catchments_to_process_file = os.path.normpath(sys.argv[2])

  parser.read(settings_file)
  base_path = parser.get('Path', 'subcatchment_path')

  catchments_to_process_df = pd.read_csv(catchments_to_process_file,header=None)
  catchment_ids = catchments_to_process_df[0]

  for catchment_id in catchment_ids:
    obs_df, sim_df, stn_df, _ = construct_dfs(base_path, catchment_id, plot_groupings)
    dis_fig = discharge_plot(obs_df, sim_df, stn_df)
    other_figs = other_var_plots(sim_df, stn_df, plot_groupings)