from hat.interactive.explorers import StationsExplorer
from hat.interactive.leaflet import StatsColormap
from hat.interactive.widgets import Widget, WidgetsManager
from liscal import evaluation, subcatchment, objective, products
import xarray as xr
from IPython.display import clear_output, display
from ipywidgets import Layout, Output, VBox
import ipywidgets

from IPython.display import display, HTML
from io import BytesIO
import base64



class CalibPlotWidget(Widget):

    def __init__(self, stations_metadata, lisflood_config):
        self.stations_metadata = stations_metadata
        self.plotter = evaluation.SpeedometerPlot(lisflood_config.plot_params)
        self.config = lisflood_config

        self.figure = Output()

        super().__init__(self.figure)

    def update(self, *args, **kwargs):

        obsid = int(args[0])

        subcatch = subcatchment.SubCatchment(
        self.config, obsid, initialise=False
        )
        
        obj = objective.ObjectiveKGE(self.config, subcatch)

        try:
            figs = products.create_products(self.config, subcatch, obj, False)
            html_content = None
        except:
            html_content = f"Error loading data for obsid {obsid}"

        with self.output:
            clear_output(wait=True)
            
            if html_content is None:
                html_content = """
                <div style="max-height: 500px; overflow-y: scroll;">
                """

                for fig in figs:
                    # Save the figure to a BytesIO buffer instead of a file
                    buf = BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)  # Go to the beginning of the BytesIO buffer
                    img_data = buf.getvalue()
                    
                    # Encode the image data to base64
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    
                    # Embed the image as a base64 string in the HTML
                    html_content += f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/><br>'

                html_content += "</div>"

            display(HTML(html_content))

        

        return



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
            align_items="center",
            spacing="20px",
            width="1500px",
        )
        top_layout = ipywidgets.Layout(
            justify_content="space-around",
            align_items="center",
            spacing="20px",
            width="100%",
            height='750px',
        )
        bottom_layout = ipywidgets.Layout(
            justify_content="space-around",
            align_items="center",
            spacing="20px",
            width="100%",
            overflow="auto",
        )

        # # Frames
        top_frame = self.leafletmap.output(top_layout)
        bottom_frame = ipywidgets.VBox(
            [self.widgets["plot"].figure],
            layout=bottom_layout,
        )

        main_frame = ipywidgets.VBox(
            [self.title_label, top_frame, bottom_frame],
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