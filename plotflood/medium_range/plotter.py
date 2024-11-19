import datetime

import numpy as np
import Magics.macro as mag


COLOURS = dict(
    dwd='rgb(0.2, 0.4, 0.9)',
    eud='rgb(1.0, 0.0, 0.0)',
    low='rgb(0.6, 0.98, 0.6)',
    medium='rgb(1.0, 0.96, 0.7)',
    high='rgb(1.0, 0.72, 0.7)',
    extreme='rgb(0.96, 0.78, 0.96)',
    bg='rgb(0.96, 0.96, 0.96)',
    ege='rgb(0.2, 0.4, 0.9)',
)


def _min(min_y, data_array):
    return float(min(min_y, data_array.min()))


def _max(max_y, data_array):
    return float(max(max_y, data_array.max()))


thresholds_label = dict(
    low='1.5-year',
    medium='2-year',
    high='5-year',
    extreme='20-year',)


class Plotter():
    """ Base class for plotting.

    You should not create instances of this class.
    Derived classes should implement the following methods:
    plot()
    """

    yaxis_type = 'regular'
    # yaxis_tick_positions = np.array()
    subpage_y_axis_type = 'regular'
    axis_label_height = 0.47
    axis_title_height = 0.53
    legend_text_font_size = 0.47
    output_cairo_antialias = 'on'

    # this should be ignored if yaxis_type is 'regular'
    yaxis_tick_positions = [1., 2., 5., 10., 20., 50., 100., 200., 500., 1000.]

    def __init__(self, forecasts, ens_fc, data_attr, output_format='png'):
        """ Constructor
        :param forecasts: object containing forecast (dwd, cos, eud, eue)
        :param forecasts: name of ensemble forecast
        :returns: function for plotting hydrographs.
        """

        self.forecasts = forecasts
        date = forecasts.date

        self.output_format = output_format

        self.ens_fc = ens_fc
        self.data_attr = data_attr

        self.yaxis_title_text = self.data_attr.get('yaxis_label', '')

        # get forecasts date range (24h for temperature)
        self.plot_dt = {}
        self.fc_dates = {}
        for fc_key, fc in self.forecasts.plot_items():
            self.fc_dates[fc_key] = []
            self.plot_dt[fc_key] = fc.dt
            for step in range(fc.n_steps):
                self.fc_dates[fc_key].append(fc.date.ymd_hm(hours=(step+0.5)*fc.dt))

        # get min and max dates
        wb = self.forecasts.wb
        self.min_date = wb.date.ymd_hm()
        self.t0_fillup = wb.date.ymd_hm(t=wb.n_steps)
        self.t0_date = date.ymd_hm()
        self.max_date = date.ymd_hm(days=forecasts.max_n_days)
        self.min_max_dates = [self.min_date, self.max_date]

        # axis
        self.xaxis = mag.maxis(
            axis_orientation='horizontal',
            axis_type='date',
            axis_days_label_height=self.axis_label_height,
            axis_months_label_height=self.axis_label_height,
            axis_years_label_height=self.axis_label_height,
            axis_grid='on',
            axis_grid_colour='grey',
            axis_grid_line_style='dot',
            axis_days_sunday_label_colour='black',
            axis_title='off',
        )

        self.yaxis = mag.maxis(
            axis_orientation='vertical',
            axis_type=self.yaxis_type,
            axis_position='left',
            axis_tick_label_height=self.axis_label_height,
            axis_tick_position_list=self.yaxis_tick_positions,
            axis_grid='on',
            axis_grid_colour='grey',
            axis_grid_line_style='dot',
            axis_title='on',
            axis_title_text=self.yaxis_title_text,
            axis_title_height=self.axis_title_height,
        )

        # vertical line representing nominal time of the forecast
        self.fc_date_graph = mag.mgraph(
            graph_line_colour='rgba(0.0, 0.0, 0.0, 0.2)',
            graph_line_style='dot',
            graph_line_thickness=3,
            legend='off',
        )

        # vertical line representing nominal time of the fillup
        self.fillup_date_graph = mag.mgraph(
            graph_line_colour='rgba(0.0, 0.0, 0.0, 0.2)',
            graph_line_style='dot',
            graph_line_thickness=3,
            legend='off',
        )


class HydrographPlotter(Plotter):
    """ Base class for hydrograph plotting classes.

    You should not create instances of this class.
    Derived classes should implement the following methods:
    _get_alert_threshold()
    _fix_y_range()
    """

    def __init__(self, forecasts, ens_fc, data_attr, output_format='png'):

        super().__init__(forecasts, ens_fc, data_attr, output_format)

        self.symbols = {}

        # water balance
        self.symbols['wb'] = mag.mgraph(
            graph_line_colour='grey',
            graph_line_thickness=2,
            graph_symbol='on',
            graph_symbol_colour='black',
            graph_symbol_height=0.5,
            graph_symbol_marker_index=15,
            legend='on',
            legend_user_text=forecasts.wb.descr,
        )

        # eud and dwd fillup
        for fc_key, fc in self.forecasts.fillup_items():
            self.symbols[fc_key+'-fillup'] = mag.mgraph(
                graph_line_colour='grey',
                graph_line_thickness=2,
                graph_symbol='on',
                graph_symbol_colour=COLOURS[fc_key],
                graph_symbol_height=0.5,
                graph_symbol_marker_index=15,
                legend='on',
                legend_user_text=fc.descr,
            )

        # dwd and eud deterministic forecasts
        for fc_key, fc in self.forecasts.fcs_items():
            if not fc.is_ensemble():
                self.symbols[fc_key] = mag.mgraph(
                    graph_line_colour=COLOURS[fc_key],
                    graph_line_thickness=3,
                    legend='on',
                    legend_user_text=fc.descr,
                )

        # thresholds
        self.bg_graph = mag.mgraph(
            graph_line_colour=COLOURS['bg'],
            graph_shade_colour=COLOURS['bg'],
            graph_line_thickness=0,
            graph_type='area',
            graph_shade='on',
            legend='off',
        )

        self.thresholds_graph = {}
        for ths_key, ths in thresholds_label.items():
            self.thresholds_graph[ths_key] = mag.mgraph(
                graph_line_colour=COLOURS[ths_key],
                graph_shade_colour=COLOURS[ths_key],
                graph_line_thickness=0,
                graph_type='area',
                graph_shade='on',
                legend='on',
                legend_user_text=ths,
            )

        # horizontal line representing maximum simulated value of 20yr run
        self.max_20y_graph = mag.mgraph(
            graph_line_colour='rgba(0., 0., 0., 0.5)',
            graph_line_style='dash',
            graph_line_thickness=3,
            legend='on',
            legend_user_text='max.sim.',
        )

        # legend
        self.legend = mag.mlegend(
            legend='on',
            legend_display_type='disjoint',
            legend_column_count=5,
            legend_text_colour='black',
            legend_text_font_size=self.legend_text_font_size,
            legend_box_mode='positional',
            legend_box_x_position=4.8,
            legend_box_y_position=12.45,
            legend_box_x_length=24.0,
            legend_box_y_length=2.5,
            legend_entry_text_width=80.0,
        )

    def fix_y_range(self, miny, maxy, thresholds):
        return NotImplemented

    def plot(self, data, output_path):

        dataname = self.data_attr['name']

        myoutput = mag.output(
            output_formats=[self.output_format],
            output_name_first_page_number='off',
            output_cairo_antialias=self.output_cairo_antialias,
            output_name=output_path)

        min_y = 10000000000
        max_y = -10000000000
        forecast_plots = []
        # deterministic forecasts + wb + fillups
        for fc_key, fc in self.forecasts.plot_items():
            if not fc.is_ensemble():

                data_val = data[fc_key][dataname]

                min_y = _min(min_y, data_val)
                max_y = _max(max_y, data_val)

                data_inputs = mag.minput(
                    input_x_type='date',
                    input_date_x_values=self.fc_dates[fc_key],
                    input_y_values=data_val.values)
                forecast_plots.extend([data_inputs, self.symbols[fc_key]])

        # ensemble forecast
        fc_key = self.ens_fc
        fc = self.forecasts[fc_key]

        perc_0 = data[fc_key][dataname + '_p0']
        perc_25 = data[fc_key][dataname + '_p25']
        perc_50 = data[fc_key][dataname + '_p50']
        perc_75 = data[fc_key][dataname + '_p75']
        perc_100 = data[fc_key][dataname + '_p100']

        min_y = _min(min_y, perc_0)
        max_y = _max(max_y, perc_100)

        ens_boxplot = mag.mboxplot(
            boxplot_date_positions=self.fc_dates[fc_key],
            boxplot_minimum_values=perc_0.values,
            boxplot_maximum_values=perc_100.values,
            boxplot_box_lower_values=perc_25.values,
            boxplot_box_upper_values=perc_75.values,
            boxplot_median_values=perc_50.values,
            boxplot_box_colour='rgba(0.9, 0.9, 1.0, 0.8)',
            boxplot_box_border_colour='rgb(0.0, 0.0, 0.0)',
            boxplot_box_border_thickness=3,
            boxplot_median_colour='rgb(0.0, 0.0, 0.0)',
            boxplot_whisker_line_colour='rgb(0.0, 0.0, 0.0)',
            boxplot_whisker_line_thickness=3,
            # box plot width hard-ish coded to avoid overpositioning
            # in GLofas 30 day range.
            boxplot_box_width=min(1.0 / fc.freq, 0.5),
            legend='on',
            legend_text_font_size=self.legend_text_font_size,
            legend_user_text=fc.descr,
        )

        # Colour background with thresholds
        data_ths = {}
        bgGraph_levels = ['low', 'medium', 'high', 'extreme']
        extra_right_axis_levels = ['50', '100']  # extra levels to show only as yticks
        extra_levels = ['limit', 'limit_low', 'limit_up']  # to read data but not a background color or yticks
        all_levels = bgGraph_levels + extra_right_axis_levels + extra_levels
        right_axis_levels = bgGraph_levels + extra_right_axis_levels

        for bg_level in all_levels:
            level_value = float(data['static'].get(dataname + '_' + bg_level, 0))
            if level_value:
                data_ths[bg_level] = level_value

        min_y, max_y = self.fix_y_range(min_y, max_y, data_ths)

        right_axis_positions = [data_ths.get(lev) for lev in right_axis_levels if data_ths.get(lev)]
        right_axis_labels = [str(data['static'].get('rp_' + lev).values) for lev in right_axis_levels if data['static'].get('rp_' + lev)]
        # add also return period on right axis
        yaxis_right = mag.maxis(
            axis_orientation='vertical',
            axis_type='position_list',
            axis_position='right',
            axis_tick_label_height=self.axis_label_height,
            axis_tick_position_list=sorted(right_axis_positions),
            axis_tick_label_type="label_list",
            axis_tick_label_list=[label for label, _ in sorted(zip(right_axis_labels, right_axis_positions), key=lambda pair: pair[1])],
            axis_grid='on',
            axis_grid_colour='grey',
            axis_grid_line_style='dot',
            axis_title='on',
            axis_title_position=120,
            axis_title_text='Forecast return period [years]',
            axis_title_height=self.axis_title_height,
        )

        bg_values = []
        available_levels = list()
        for lev in data_ths.keys():
            if lev in bgGraph_levels:
                available_levels.append(lev)
                bg_values.append([min(max_y, data_ths[lev]), min(max_y, data_ths[lev])])
        bg_values.append([max_y, max_y])

        bg_inputs = [
            mag.minput(
                input_x_type='date',
                input_date_x_values=self.min_max_dates,
                input_date_x2_values=self.min_max_dates,
                input_y_values=[min_y, min_y],
                input_y2_values=bg_values[0],
            ),
            self.bg_graph,
        ]
        for i in range(len(available_levels)):
            bg_inputs.extend([
                mag.minput(
                    input_x_type='date',
                    input_date_x_values=self.min_max_dates,
                    input_date_x2_values=self.min_max_dates,
                    input_y_values=bg_values[i],
                    input_y2_values=bg_values[i + 1],
                ),
                self.thresholds_graph[available_levels[i]],
            ])

        # horizontal line: max 20yr simulated discharge
        hlines = []
        data_maxsim = data['static'].get(dataname + '_maxsim', None)
        if data_maxsim is not None:
            data_maxsim = float(data_maxsim)
            max_20y_input = mag.minput(
                input_x_type='date',
                input_date_x_values=self.min_max_dates,
                input_y_values=[data_maxsim, data_maxsim],
            )
            hlines.extend([max_20y_input, self.max_20y_graph])

        # vertical line representing nominal date/time of the forecast
        fc_date_input = mag.minput(
            input_x_type='date',
            input_date_x_values=[self.t0_date, self.t0_date],
            input_y_values=[min_y, max_y],
        )
        hlines.extend([fc_date_input, self.fc_date_graph])

        # vertical line representing nominal date/time of the fillup
        fillup_date_input = mag.minput(
            input_x_type='date',
            input_date_x_values=[self.t0_fillup, self.t0_fillup],
            input_y_values=[min_y, max_y],
        )
        hlines.extend([fillup_date_input, self.fillup_date_graph])

        # Setting the cartesian view
        projection = mag.mmap(
            super_page_x_length=30.0,
            super_page_y_length=14.25,
            subpage_x_position=3.18,
            subpage_y_position=2.50,
            subpage_x_length=30.0 - 3.18 - 2.66,
            subpage_y_length=14.25 - 2.50 - 1.5,
            subpage_map_projection='cartesian',
            subpage_x_axis_type='date',
            subpage_y_axis_type=self.subpage_y_axis_type,
            subpage_x_date_min=self.min_date,
            subpage_x_date_max=self.max_date,
            subpage_y_min=min_y,
            subpage_y_max=max_y,
            page_id_line="off",
        )

        mag.plot(
            myoutput, projection,
            # axis
            self.yaxis, yaxis_right, self.xaxis,
            # background color
            *bg_inputs,
            # dotted lines
            *hlines,
            # forecasts
            ens_boxplot,
            *forecast_plots,
            self.legend,
        )


class DischargeHydrographPlotter(HydrographPlotter):

    subpage_y_axis_type = 'regular'

    def __init__(self, forecasts, ens_fc, data_attr, output_format='png'):

        super().__init__(forecasts, ens_fc, data_attr, output_format)

    def fix_y_range(self, miny, maxy, thresholds):

        min_y = 0.
        max_y = max(maxy, min(thresholds.values())) * 1.1

        return min_y, max_y


class ReturnPeriodHydrographPlotter(HydrographPlotter):

    yaxis_type = 'position_list'
    subpage_y_axis_type = 'logarithmic'

    def __init__(self, forecasts, ens_fc, data_attr, output_format='png'):

        super().__init__(forecasts, ens_fc, data_attr, output_format)

    def fix_y_range(self, miny, maxy, thresholds):
        max_y = max(maxy, thresholds['extreme'] * (1.1))
        max_y = min(max_y, thresholds.get('limit_up', max_y))
        return 0.9, max_y


class LinePlotter(Plotter):

    """ Class for line plotting.
    """
    def __init__(self, forecasts, ens_fc, data_attr, output_format='png'):
        super().__init__(forecasts, ens_fc, data_attr, output_format)

        self.symbols = {}

        # water balance
        self.symbols['wb'] = mag.mgraph(
            graph_line_colour='grey',
            graph_line_thickness=2,
            graph_symbol='on',
            graph_symbol_colour='black',
            graph_symbol_height=0.5,
            graph_symbol_marker_index=15,
            legend='on',
            legend_user_text=forecasts.wb.descr)

        # eud and dwd fillup
        for fc_key, fc in self.forecasts.fillup_items():
            self.symbols[fc_key+'-fillup'] = mag.mgraph(
                graph_line_colour='grey',
                graph_line_thickness=2,
                graph_symbol='on',
                graph_symbol_colour=COLOURS[fc_key],
                graph_symbol_height=0.5,
                graph_symbol_marker_index=15,
                legend='on',
                legend_user_text=fc.descr,
            )

        # dwd and eud deterministic forecasts
        for fc_key, fc in self.forecasts.fcs_items():
            if not fc.is_ensemble():
                self.symbols[fc_key] = mag.mgraph(
                    graph_line_colour=COLOURS[fc_key],
                    graph_line_thickness=2,
                    graph_symbol='on',
                    graph_symbol_colour=COLOURS[fc_key],
                    graph_symbol_height=0.5,
                    graph_symbol_marker_index=15,
                    graph_symbol_outline='on',
                    graph_symbol_outline_thickness=2,
                    legend='on',
                    legend_user_text=fc.descr,
                )

        # ensemble forecast percentile boxplots
        fc_key = self.ens_fc
        fc = self.forecasts[fc_key]

        self.perc_mid_graph = mag.mgraph(
            graph_type='area',
            graph_line_colour='rgb(0.75, 0.75, 0.75)',
            graph_shade_colour='rgb(0.75, 0.75, 0.75)',
            legend='on',
            legend_user_text=fc.descr + " 25%-75%",
        )

        self.perc_min_max_graph = mag.mgraph(
            graph_type='area',
            graph_line_colour='rgb(0.90, 0.90, 0.90)',
            graph_shade_colour='rgb(0.90, 0.90, 0.90)',
            legend='on',
            legend_user_text=fc.descr + "  0%-100%",
        )

        self.perc_mean_graph = mag.mgraph(
            graph_line_colour='black',
            graph_line_thickness=4,
            legend='on',
            legend_user_text=fc.descr + ' 50%',
        )

        self.legend = mag.mlegend(
            legend='on',
            legend_text_colour='black',
            legend_column_count=3,
            legend_text_font_size=self.legend_text_font_size,
            legend_box_mode='positional',
            legend_box_x_position=1.62,
            legend_box_y_position=9.5,
            legend_box_x_length=29.0,
            legend_box_y_length=2.5,
            legend_entry_text_width=80.,
        )

    def plot(self, data, output_path):

        dataname = self.data_attr['name']

        myoutput = mag.output(
            output_formats=[self.output_format],
            output_name_first_page_number='off',
            output_cairo_antialias=self.output_cairo_antialias,
            output_name=output_path)

        min_y = 10000000000
        max_y = -10000000000

        forecast_plots = []

        # deterministic forecasts + water balance + fillup
        det_inputs = {}
        for fc_key, fc in self.forecasts.plot_items():
            if not fc.is_ensemble():
                factor = 1.0
                # will convert from e.g. mm/6hours to mm/hour
                if self.data_attr['scale'] == 'dt':
                    factor = 1.0 / fc.dt
                data_det = data[fc_key][dataname] * factor

                min_y = _min(min_y, data_det)
                max_y = _max(max_y, data_det)

                mean_dates, mean_values = self.transform(self.plot_dt[fc_key], self.fc_dates[fc_key], data_det.values)

                det_inputs[fc_key] = mag.minput(
                    input_x_type='date',
                    input_date_x_values=mean_dates,
                    input_y_values=mean_values)
                forecast_plots.extend([det_inputs[fc_key], self.symbols[fc_key]])

        # ensemble forecast percentile lineplots
        fc_key = self.ens_fc
        fc = self.forecasts[fc_key]

        factor = 1.0
        # will convert from e.g. mm/6hours to mm/hour
        if self.data_attr['scale'] == 'dt':
            factor = 1.0 / fc.dt
        perc_0 = data[fc_key][dataname + '_p0'].values * factor
        perc_25 = data[fc_key][dataname + '_p25'].values * factor
        perc_50 = data[fc_key][dataname + '_p50'].values * factor
        perc_75 = data[fc_key][dataname + '_p75'].values * factor
        perc_100 = data[fc_key][dataname + '_p100'].values * factor

        min_y = _min(min_y, perc_0)
        max_y = _max(max_y, perc_100)

        perc_mid_input = mag.minput(
            input_x_type='date',
            input_date_x_values=self.fc_dates[fc_key],
            input_date_x2_values=self.fc_dates[fc_key],
            input_y_values=perc_25,
            input_y2_values=perc_75,
        )

        perc_min_max_input = mag.minput(
            input_x_type='date',
            input_date_x_values=self.fc_dates[fc_key],
            input_date_x2_values=self.fc_dates[fc_key],
            input_y_values=perc_0,
            input_y2_values=perc_100,
        )

        mean_dates, mean_values = self.transform(self.plot_dt[fc_key], self.fc_dates[fc_key], perc_50)

        perc_mean_input = mag.minput(
            input_x_type='date',
            input_date_x_values=mean_dates,
            input_y_values=mean_values,
        )

        min_y, max_y = self.fix_y_range(min_y, max_y)

        # vertical line representing nominal date/time of the forecast
        fc_date_input = mag.minput(
            input_x_type='date',
            input_date_x_values=[self.t0_date, self.t0_date],
            input_y_values=[min_y, max_y],
        )

        # vertical line representing nominal date/time of the fillup
        fillup_date_input = mag.minput(
            input_x_type='date',
            input_date_x_values=[self.t0_fillup, self.t0_fillup],
            input_y_values=[min_y, max_y],
        )

        # Setting the cartesian view
        projection = mag.mmap(
            super_page_x_length=30.0,
            super_page_y_length=12.0,
            subpage_x_position=3.18,
            subpage_y_position=2.15,
            subpage_x_length=30.0 - 3.18 - 2.66,
            subpage_y_length=12.0 - 2.15 - 2.43,
            subpage_map_projection='cartesian',
            subpage_x_axis_type='date',
            subpage_y_axis_type='regular',
            subpage_x_date_min=self.min_date,
            subpage_x_date_max=self.max_date,
            subpage_y_min=min_y,
            subpage_y_max=max_y,
            page_id_line="off",
        )

        mag.plot(
            myoutput, projection,
            # axis
            self.yaxis, self.xaxis,
            # dotted lines
            fc_date_input, self.fc_date_graph,
            fillup_date_input, self.fillup_date_graph,
            # forecasts
            perc_min_max_input, self.perc_min_max_graph,
            perc_mid_input, self.perc_mid_graph,
            perc_mean_input, self.perc_mean_graph,
            *forecast_plots,
            self.legend,
        )

    def fix_y_range(self, miny, maxy):
        min_y = miny - (maxy - miny) * 0.1
        max_y = maxy + (maxy - miny) * 0.1
        return min_y, max_y

    # nothing to do for line plot
    def transform(self, fc_dt, dates, values):
        return dates, values


class BarPlotter(LinePlotter):

    """ Class for bar plotting.
    """

    def __init__(self, forecasts, ens_fc, data_attr, output_format='png'):
        super().__init__(forecasts, ens_fc, data_attr, output_format)

        self.symbols = {}

        # water balance
        self.symbols['wb'] = mag.mgraph(
            graph_line_colour='black',
            graph_line_thickness=2,
            legend='on',
            legend_user_text=forecasts.wb.descr,
        )

        # eud and dwd fillup
        for fc_key, fc in self.forecasts.fillup_items():
            self.symbols[fc_key+'-fillup'] = mag.mgraph(
                graph_line_colour=COLOURS[fc_key],
                graph_line_thickness=2,
                legend='on',
                legend_user_text=fc.descr,
            )

        # dwd and eud deterministic forecasts
        for fc_key, fc in self.forecasts.fcs_items():
            if not fc.is_ensemble():
                self.symbols[fc_key] = mag.mgraph(
                    graph_line_colour=COLOURS[fc_key],
                    graph_line_thickness=3,
                    legend='on',
                    legend_user_text=fc.descr,
                )

        # ensemble forecast percentile boxplots
        fc_key = self.ens_fc
        fc = self.forecasts[fc_key]
        fc_step = datetime.timedelta(hours=self.plot_dt[fc_key]).total_seconds()

        self.perc_mid_graph = mag.mgraph(
            graph_type='bar',
            graph_bar_line_colour='rgb(0.75, 0.75, 0.75)',
            graph_bar_line_thickness=1,
            graph_bar_width=fc_step,
            graph_shade_colour="rgb(0.75, 0.75, 0.75)",
            legend='on',
            legend_user_text=fc.descr + " 25%-75%",
        )

        self.perc_min_max_graph = mag.mgraph(
            graph_type='bar',
            graph_bar_line_colour='rgb(0.90, 0.90, 0.90)',
            graph_bar_line_thickness=1,
            graph_bar_width=fc_step,
            graph_shade_colour='rgb(0.90, 0.90, 0.90)',
            legend='on',
            legend_user_text=fc.descr + " 0%-100%",
        )

        self.perc_mean_graph = mag.mgraph(
            graph_line_colour='black',
            graph_line_thickness=4,
            legend='on',
            legend_user_text=fc.descr + ' 50%',
        )

        self.legend = mag.mlegend(
            legend='on',
            legend_text_colour='black',
            legend_column_count=3,
            legend_text_font_size=self.legend_text_font_size,
            legend_box_mode='positional',
            legend_box_x_position=1.62,
            legend_box_y_position=9.5,
            legend_box_x_length=29.0,
            legend_box_y_length=2.5,
            legend_entry_text_width=80.,
        )

    def fix_y_range(self, miny, maxy):

        min_y = 0.
        max_y = max(1., maxy + (maxy - miny) * 0.1)

        return min_y, max_y

    def transform(self, fc_dt, dates, values):
        fc_step = datetime.timedelta(hours=fc_dt / 2)
        dates_tr = []
        values_tr = np.zeros(len(values) * 2)
        for i in range(len(dates)):
            date = datetime.datetime.strptime(dates[i], '%Y-%m-%d %H:%M')
            d1 = date - fc_step
            d2 = date + fc_step
            dates_tr.append(d1.strftime('%Y-%m-%d %H:%M'))
            dates_tr.append(d2.strftime('%Y-%m-%d %H:%M'))
            values_tr[i * 2] = values[i]
            values_tr[i * 2 + 1] = values[i] * (1 + 1e-08)  # need to add eps or there's a small bug in the first points
        return dates_tr, values_tr
