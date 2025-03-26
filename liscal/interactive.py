from hat.interactive.explorers import StationsExplorer
from hat.interactive.leaflet import StatsColormap
from hat.interactive.widgets import Widget, WidgetsManager
from liscal import evaluation, subcatchment, objective, products
import xarray as xr
from IPython.display import clear_output, display
from ipywidgets import Layout, Output, VBox
import ipywidgets
import os

class CalibPlotWidget(Widget):

    def __init__(self, stations_metadata, lisflood_config):
        self.stations_metadata = stations_metadata
        self.plotter = evaluation.SpeedometerPlot(lisflood_config.plot_params)
        self.config = lisflood_config

        self.figure = Output()
        output = VBox(
            [self.figure],
            layout=Layout(width="1000px", align_items="center"),
        )

        super().__init__(output)

    def update(self, *args, **kwargs):

        obsid = int(args[0])

        subcatch = subcatchment.SubCatchment(
        self.config, obsid, initialise=False
        )
        
        try:
            obj = objective.ObjectiveKGE(self.config, subcatch)

            with self.figure:
                clear_output(wait=True)
                figs = products.create_products(self.config, subcatch, obj, False)
                for fig in figs:
                    display(fig)
        except:
            print(f"Error generating plot for obsid {obsid}")



class CalibrationExplorer(StationsExplorer):
    def __init__(self, hat_config, lisflood_config):
        title = "Interactive Map Visualisation for Hydrological Model Performance"
        super().__init__(hat_config, title)

        widgets = {}
        self.loading_widget = ipywidgets.Label(value="")
        widgets["plot"] = CalibPlotWidget(self.stations_metadata, lisflood_config)
        self.widgets = WidgetsManager(widgets, hat_config["station_id_column_name"], self.loading_widget)

    def create_frame(self):
        main_layout = ipywidgets.Layout(
            justify_content="space-around",
            align_items="stretch",
            spacing="2px",
            width="2000px",
        )
        left_layout = ipywidgets.Layout(
            justify_content="space-around",
            align_items="center",
            spacing="10px",
            width="30%",
        )
        right_layout = ipywidgets.Layout(
            justify_content="center",
            align_items="center",
            spacing="10px",
            width="60%",
        )

        # # Frames
        top_left_frame = self.leafletmap.output(left_layout)
        top_right_frame = ipywidgets.VBox(
            [self.widgets["plot"].figure],
            layout=right_layout,
        )
        main_top_frame = ipywidgets.HBox([top_left_frame, top_right_frame])

        main_frame = ipywidgets.VBox(
            [self.title_label, main_top_frame],
            layout=main_layout,
        )
        return main_frame

    def plot(self, mp_colormap="viridis", colorby='Kling Gupta Efficiency'):
        stats = xr.Dataset(
            {
                colorby: (['station'], self.stations_metadata[colorby].astype('float'))
            },
            coords={
                'station': self.stations_metadata[self.config['station_id_column_name']],
            }
        )[colorby]

        colormap = StatsColormap(self.config, stats, mp_colormap)

        self.leafletmap.add_geolayer(
            self.stations_metadata,
            colormap, 
            self.widgets,
            self.config["station_coordinates"],
        )
        frame = self.create_frame()
        display(frame)
        return