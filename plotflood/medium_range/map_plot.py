import os

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import Magics.macro as mag

STYLES_MAPPER = dict()

def register_style(style_name):
    def wrapped(style_cls):
        STYLES_MAPPER.update({style_name: style_cls})
    return wrapped

class MapPlot:
    """
    Base class for map plotter
    """

    output_format = 'png'
    available_styles = STYLES_MAPPER

    def __init__(self, domain, map_style, reversed_scheme=False):

        self.domain = domain
        try:
            contour_style = self.__class__.available_styles[map_style]
        except KeyError:
            raise ValueError('Map type ' + map_style + ' not supported')

        self.contour_style = contour_style

        width = 20.0
        height = width * self.domain.ny / self.domain.nx

        self.projection = mag.mmap(subpage_map_projection="cartesian",
                                   subpage_x_min=self.domain.xulc,
                                   subpage_x_max=self.domain.xbrc,
                                   subpage_y_min=self.domain.ybrc,
                                   subpage_y_max=self.domain.yulc,
                                   subpage_y_axis_type="regular",
                                   subpage_x_axis_type="regular",
                                   subpage_x_position=0.,
                                   subpage_y_position=0.,
                                   subpage_x_length=width,
                                   subpage_y_length=height,
                                   super_page_x_length=width,
                                   super_page_y_length=height,
                                   page_id_line='off',
                                   subpage_frame='off')

        level_list = contour_style.contour_level_list
        colors_list = contour_style.contour_shade_list
        if reversed_scheme:
            colors_list = list(reversed(colors_list))

        self.contour = mag.mcont(
            contour='off',
            contour_label='off',
            contour_shade='on',
            legend='off',
            contour_level_selection_type='level_list',
            contour_level_list=level_list,
            contour_shade_technique='dump_shading',
            contour_shade_colour_method='list',
            contour_shade_colour_list=colors_list,
        )

    def plot(self, data, output_path):

        output = mag.output(
            output_format=self.output_format,
            output_name=output_path,
            output_name_first_page_number='off',
            output_cairo_antialias='on',
            output_width=self.domain.nx,
        )

        mapdata = mag.minput(
            input_type='cartesian',
            input_field_initial_x=self.domain.xulc,
            input_field_final_x=self.domain.xbrc,
            input_field_initial_y=self.domain.yulc,
            input_field_final_y=self.domain.ybrc,
            input_field=data.reshape(self.domain.ny, self.domain.nx)
        )

        mag.plot(output, self.projection, mapdata, self.contour)


# ContourStyle class
class ContourStyle():
    contour_shade_list = list()  # TODO: add support to Magics' predefined color schemes
    contour_level_list = list()  # must have len(shade_list) + 1

    # For colorbar plotting only
    extend = 'neither'  # one of max, min, both or neither
    labels = list()  # same length as shade_list

    def print(self):
        return print(self)

    def __repr__(self):
        return str(list(zip(self.contour_shade_list, self.contour_level_list)))

    def plot_colorbar(
        self, outdir=os.getcwd(), reversed_scheme=False, scale=None, outfmt='png'
    ):

        def to_decimal(x):
            alpha = 1.
            x1 = float(x)
            if x1 > 1.:
                alpha = 1. / 255.
            return x1 * alpha

        clist = []
        colors = reversed(self.contour_shade_list) if reversed_scheme else self.contour_shade_list
        suffix = 'reversed' if reversed_scheme else ''

        outname = '{}_{}colorbar.{}'.format(
            self.__class__.__name__,
            suffix,
            outfmt
        )

        for cc in colors:
            new = cc.replace(")", "")
            if 'RGBA' in cc:
                new = new.replace('RGBA(', '')
                r, g, b, a = tuple(map(to_decimal, new.split(',')))
            else:
                new = new.replace("RGB(", "")
                r, g, b = tuple(map(to_decimal, new.split(',')))
                a = 1.
            clist.append(np.array([r, g, b, a]))

        extend_edges = getattr(self, 'extend', 'neither')
        levels = self.contour_level_list
        if scale == 'percent':
            levels = list(map(lambda x: x * 100., levels))

        labels = getattr(self, 'labels')
        # If labels present will place them between usual ticks.
        # One for each category/color and that should be done before pruning
        # for extension.
        if labels:
            assert len(labels) == len(clist), 'Labels and color list do not match'
            y_ticks = []
            for i in range(len(levels) - 1):
                y_ticks.append(levels[i] + (levels[i + 1] - levels[i]) / 2.)

        # pruning extended edges
        if extend_edges == 'both':
            level_ticks = levels[1:-1]
        elif extend_edges == 'min':
            level_ticks = levels[1:]
        elif extend_edges == 'max':
            level_ticks = levels[:-1]
        else:
            level_ticks = levels

        bounds = mpl.colors.BoundaryNorm(levels, len(levels))
        cmap = mpl.colors.ListedColormap(clist)
        fig = plt.figure()
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=bounds, cmap=cmap),
            orientation='vertical',
            ticks=level_ticks,
            extend=extend_edges,
        )
        fig.get_axes()[0].set_visible(False)

        if labels:
            cbar.set_ticks(y_ticks)
            cbar.set_ticklabels(labels)
            cbar.ax.tick_params(length=0)

        fig.savefig(os.path.join(outdir, outname),
                    bbox_inches='tight',
                    dpi=100)
        plt.close()


# Alert maps
@register_style(style_name='DetAlerts')
class ContourDetAlerts(ContourStyle):
    contour_shade_list = [
        "RGB(255, 0, 0)",
        "RGB(255, 255, 0)",
        "RGB(0, 255, 0)",
        "RGB(204, 0, 255)",
        "RGB(200, 200, 200)"
    ]
    contour_level_list = [-1., 0.5, 1.5, 2.5, 5.5, 6.5]


class ContourEnsAlerts(ContourStyle):
    contour_level_list = [1., 5.5, 15.5, 25.5, 35.5, 45.5, 52.]

@register_style(style_name='EnsHighAlerts')
class ContourEnsHighAlerts(ContourEnsAlerts):
    contour_shade_list = [
        "RGB(255, 204, 204)",
        "RGB(255, 167, 156)",
        "RGB(252, 132, 111)",
        "RGB(245, 97, 71)",
        "RGB(232, 60, 37)",
        "RGB(219, 0, 0)"
    ]

@register_style(style_name='EnsExtremeAlerts')
class ContourEnsExtremeAlerts(ContourEnsAlerts):
    contour_shade_list = [
        "RGB(250, 195, 255)",
        "RGB(219, 137, 223)",
        "RGB(189, 88, 192)",
        "RGB(161, 49, 161)",
        "RGB(130, 20, 129)",
        "RGB(100, 0, 96)"
    ]


# Probability maps
class ContourEnsProb(ContourStyle):
    contour_level_list = [1., 10.01, 30.01, 50.01, 70.01, 90.01, 100.01]

@register_style(style_name='EnsProbLT48h')
class ContourEnsProbLT48h(ContourEnsProb):
    contour_shade_list = [
        "RGB(195, 252, 255)",
        "RGB(131, 204, 220)",
        "RGB(79, 153, 186)",
        "RGB(39, 102, 151)",
        "RGB(10, 56, 117)",
        "RGB(0, 37, 100)"
    ]

@register_style(style_name='EnsProbGT48h')
class ContourEnsProbGT48h(ContourEnsProb):
    contour_shade_list = [
        "RGB(255, 255, 204)",
        "RGB(255, 255, 102)",
        "RGB(255, 204, 0)",
        "RGB(255, 153, 0)",
        "RGB(255, 102, 0)",
        "RGB(255, 51, 0)"
    ]


# Rain maps
@register_style(style_name='DetRain')
class ContourDetRain(ContourStyle):
    contour_shade_list = [
        "RGB(205, 250, 100)",
        "RGB(145, 240, 67)",
        "RGB(88, 230, 32)",
        "RGB(62, 209, 54)",
        "RGB(61, 186, 101)",
        "RGB(49, 163, 144)",
        "RGB(30, 134, 166)",
        "RGB(33, 88, 150)",
        "RGB(28, 52, 135)",
        "RGB(13, 17, 120)",
    ]
    contour_level_list = [0., 14., 28., 42., 56., 70., 84., 98., 112., 126., 1400.]

    extend = 'max'


class ContourEnsRain(ContourStyle):
    contour_level_list = [1., 10., 20., 30., 40., 50., 60., 70., 80., 90., 1000.]
    extend = 'max'

@register_style('EnsRainGt50')
class ContourEnsRainGt50(ContourEnsRain):
    contour_shade_list = [
        "RGB(204, 255, 204)",
        "RGB(186, 250, 182)",
        "RGB(169, 245, 162)",
        "RGB(150, 240, 141)",
        "RGB(131, 235, 120)",
        "RGB(112, 230, 101)",
        "RGB(94, 222, 82)",
        "RGB(73, 217, 63)",
        "RGB(50, 209, 42)",
        "RGB(14, 204, 14)"
    ]

@register_style('EnsRainGt150')
class ContourEnsRainGt150(ContourEnsRain):
    contour_shade_list = [
        "RGB(182, 237, 240)",
        "RGB(152, 210, 237)",
        "RGB(124, 187, 235)",
        "RGB(92, 163, 230)",
        "RGB(54, 141, 227)",
        "RGB(33, 118, 217)",
        "RGB(34, 89, 199)",
        "RGB(29, 62, 181)",
        "RGB(23, 39, 163)",
        "RGB(9, 9, 146)",
    ]


# Anomaly maps
class ContourObsAnomaly(ContourStyle):
    contour_level_list = [-9998.9, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 9998.9]
    extend = 'both'

@register_style('ObsSoilMoistureAnomaly')
class ContourObsSoilMoistureAnomaly(ContourObsAnomaly):
    contour_shade_list = [
        "RGB(255, 0, 0)",
        "RGB(255, 170, 0)",
        "RGB(255, 255, 0)",
        "RGBA(255, 255, 255,0)",
        "RGB(233, 204, 249)",
        "RGB(131, 51, 147)",
        "RGB(0, 0, 255)",
    ]

    labels = [
        'Highly drier than normal (SMA < -2)',
        'Much drier than normal (-2 <= SMA < -1.5)',
        'Drier than normal (-1.5 <= SMA < -1)',
        'Near normal (-1 <= SMA < 1)',
        'Wetter than normal (1 <= SMA < 1.5)',
        'Much wetter than normal (1.5 <= SMA < 2)',
        'Highly wetter than normal (SMA > 2)',
    ]

@register_style('ObsSnowCoverageAnomaly')
class ContourObsSnowCoverageAnomaly(ContourObsAnomaly):
    contour_shade_list = [
        "RGB(255, 0, 0)",
        "RGB(255, 170, 0)",
        "RGB(255, 255, 0)",
        "RGB(240, 240, 240)",
        "RGB(233, 204, 249)",
        "RGB(131, 51, 147)",
        "RGB(0, 0, 255)"
    ]

    labels = [
        'Highly less than normal (SSPI < -2)',
        'Much less than normal (-2 <= SSPI < -1.5)',
        'Less than normal (-1.5 <= SSPI < -1)',
        'Near normal (-1 <= SSPI < 1)',
        'More than normal (1 <= SSPI < 1.5)',
        'Much more than normal (1.5 <= SSPI < 2)',
        'Highly more than normal (SSPI > 2)',
    ]

@register_style('ObsGlobalAnomaly')
class ContourGlobalAnomaly(ContourObsAnomaly):
    """
    For more detailed anomaly fields.
    """

    contour_level_list = [-1000., -5., -3., -2., -1.5, -1., -0.5, 0.5, 1., 1.5, 2., 3., 5., 1000.]

    contour_shade_list = [
        "RGB(0.4716,0.4066,0.2421)",
        "RGB(0.592,0.5254,0.3257)",
        "RGB(0.6841,0.5953,0.3512)",
        "RGB(0.7724,0.6737,0.4433)",
        "RGB(0.8645,0.7679,0.5238)",
        "RGB(0.9469,0.8833,0.6923)",
        "RGB(1., 1., 1.)",
        "RGB(0.6632,0.8898,0.9525)",
        "RGB(0.4071,0.82,0.9576)",
        "RGB(0.003922,0.6846,1)",
        "RGB(0.003922,0.5352,1)",
        "RGB(0.128,0.3748,0.9073)",
        "RGB(0.08557,0.2526,0.7536)"
    ]
    extend = 'both'


# Snow coverage maps
@register_style('ObsSnowCoverage')
class ContourObsSnowCoverage(ContourStyle):
    contour_shade_list = [
        "RGB(0, 0, 102)",
        "RGB(0, 0, 255)",
        "RGB(0, 128, 255)",
        "RGB(115, 163, 255)",
        "RGB(204, 230, 255)",
        "RGB(255, 218, 0)",
        "RGB(255, 128, 0)",
        "RGB(255, 0, 0)",
        "RGB(153, 22, 8)",
        "RGB(82, 3, 0)",
        "RGB(255, 255, 255)",
        "RGB(82, 3, 0)",
    ]
    contour_level_list = [1., 10., 25., 50., 75., 100., 125., 150., 200., 250., 9998.9, 9999.1, 200000.]
    extend = 'max'

@register_style('GlobalSnowMelt')
class ContourGlobalSnowMelt(ContourStyle):

    contour_level_list = [1., 2., 5., 10., 20., 35., 50., 100., 10000]

    contour_shade_list = [
        "RGB(0.5022,0.9644,0.5176)",
        "RGB(0.2534,0.9152,0.2755)",
        "RGB(0.01006,0.8448,0.03789)",
        "RGB(0.02458,0.7127,0.04752)",
        "RGB(0.03197,0.572,0.04997)",
        "RGB(0.04195,0.4443,0.05536)",
        "RGB(0.04945,0.327,0.05872)",
        "RGB(0.05705,0.2175,0.06239)",
    ]
    extend = 'max'

@register_style('GlobalSnowCover')
class ContourGlobalSnowCoverage(ContourStyle):

    contour_level_list = [1., 2., 5., 10., 20., 35., 50., 75., 100., 200., 300., 500., 100000.]

    contour_shade_list = [
        "RGB(0.5064,0.5809,0.4858)",
        "RGB(0.6153,0.6753,0.5953)",
        "RGB(0.7395,0.7658,0.7323)",
        "RGB(0.8118,0.8118,0.8118)",
        "RGB(0.8902,0.8902,0.8902)",
        "RGB(0.949,0.949,0.949)",
        "RGB(0.9961,0.9961,0.9961)",
        "RGB(0.9676,0.8309,0.4206)",
        "RGB(0.9895,0.7421,0.1164)",
        "RGB(0.9961,0.6008,0.007828)",
        "RGB(0.9961,0.4526,0.007828)",
        "RGB(0.919,0.3211,0.02214)",
    ]

    extend = 'max'

@register_style('AccumulatedRain')
class ContourAccRain(ContourStyle):
    """
    For Global TP
    """

    contour_level_list = [5., 10., 20., 35., 50., 75., 100., 200., 300., 500., 100000]

    contour_shade_list = [
        "RGB(0.7804,0.9424,0.9451)",
        "RGB(0.5824,0.9178,0.9235)",
        "RGB(0.4083,0.8454,0.9328)",
        "RGB(0.003922,0.7842,1)",
        "RGB(0.003922,0.6348,1)",
        "RGB(0.003922,0.502,1)",
        "RGB(0.003922,0.3193,1)",
        "RGB(0.02261,0.2396,0.746)",
        "RGB(0.03764,0.06818,0.4957)",
        "RGB(0.05,0.06999,0.3)",
    ]

    extend = 'max'

@register_style('GlobalDailyRain')
class ContourGlobalDailyRain(ContourStyle):

    contour_level_list = [1., 2., 5., 10., 20., 35., 50., 75., 100., 200., 100000.]

    contour_shade_list = [
        "RGB(0.7804,0.9424,0.9451)",
        "RGB(0.5824,0.9178,0.9235)",
        "RGB(0.4078,0.8471,0.9333)",
        "RGB(0.003922,0.7842,1)",
        "RGB(0.003922,0.6348,1)",
        "RGB(0.003922,0.502,1)",
        "RGB(0.003922,0.3193,1)",
        "RGB(0.02261,0.2396,0.746)",
        "RGB(0.03764,0.06818,0.4957)",
        "RGB(0.05,0.06999,0.3)",
    ]

    extend = 'max'


@register_style('GlobalRainbowRain')
class ContourGlobalRainbowRain(ContourStyle):

    contour_level_list = [10., 15., 20., 25., 50., 100., 150., 200., 10000.]

    contour_shade_list = [
        "RGB(0.003922, 0.4522, 1)",
        "RGB(0.003922, 0.834, 1)",
        "RGB(0.003922, 1,0.8008)",
        "RGB(0.7569, 1, 0.02745)",
        "RGB(0.9961, 0.8, 0.01564)",
        "RGB(0.9961, 0.4405, 0.01564)",
        "RGB(0.9765, 0.1822, 0.003845)",
        "RGB(0.5645, 0.1293, 0.03156)"
    ]

    extend = 'max'


# Soil moisture maps
@register_style('ObsSoilMoisture')
class ContourObsSoilMoisture(ContourStyle):
    contour_shade_list = [
        "RGB(82, 3, 0)",
        "RGB(153, 22, 8)",
        "RGB(255, 0, 0)",
        "RGB(255, 128, 0)",
        "RGB(255, 218, 0)",
        "RGB(204, 230, 255)",
        "RGB(115, 163, 255)",
        "RGB(0, 128, 255)",
        "RGB(0, 0, 255)",
        "RGB(0, 0, 102)",
    ]
    contour_level_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 9998.9]
    extend = 'max'

@register_style('ObsGlobalSoilMoisture')
class ContourGlobalSoilMoisture(ContourStyle):
    """
    Another option for SoilMoisture.
    """

    contour_level_list = [0, .10, .20, .30, .40, .50, .60, .70, .80, .90, .99, 9998.9]

    contour_shade_list = [
        "RGB(0.8336,0.9389,0.8372)",
        "RGB(0.7236,0.947,0.731)",
        "RGB(0.5022,0.9644,0.5176)",
        "RGB(0.2539,0.9382,0.2767)",
        "RGB(0.09726,0.8753,0.1232)",
        "RGB(0.01614,0.8074,0.04253)",
        "RGB(0.02435,0.6658,0.04575)",
        "RGB(0.03239,0.5794,0.05061)",
        "RGB(0.04152,0.4291,0.05444)",
        "RGB(0.04724,0.3292,0.05664)",
        "RGB(0.06278,0.1411,0.06538)",
    ]

    extend = 'max'


# Rainfall maps
@register_style('ObsRain')
class ContourObsRain(ContourStyle):
    contour_shade_list = [
        "RGB(237, 248, 251)",
        "RGB(213, 230, 241)",
        "RGB(190, 213, 231)",
        "RGB(171, 194, 221)",
        "RGB(155, 172, 209)",
        "RGB(140, 150, 198)",
        "RGB(138, 124, 185)",
        "RGB(136, 98, 173)",
        "RGB(134, 71, 158)",
        "RGB(131, 43, 141)",
    ]
    contour_level_list = [2., 5., 10., 15., 25., 40., 60., 80., 100., 120., 1000.]
    extend = 'max'


# Temperature maps
@register_style('ObsTemperature')
class ContourObsTemperature(ContourStyle):
    contour_shade_list = [
        "RGB(5, 113, 176)",
        "RGB(67, 150, 196)",
        "RGB(130, 187, 216)",
        "RGB(179, 213, 230)",
        "RGB(224, 235, 241)",
        "RGB(246, 234, 229)",
        "RGB(245, 209, 193)",
        "RGB(244, 183, 157)",
        "RGB(240, 152, 122)",
        "RGB(227, 101, 92)",
        "RGB(214, 50, 62)",
        "RGB(202, 0, 32)",
    ]
    contour_level_list = [-70., -15., -10., -5., -1., 0., 1., 5., 10., 15., 20., 24., 51.]
    extend = 'both'

@register_style('ObsGlobalTemperature')
class ContourGlobalTemperature(ContourStyle):
    """
    For Global range of possible values of surface temperature.
    """

    contour_level_list = [-100., -30., -25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25., 30., 35., 100.]

    contour_shade_list = [
        "RGB(0.4393,0.191,0.5149)",
        "RGB(0.5353,0.3266,0.5988)",
        "RGB(0.6036,0.4016,0.6651)",
        "RGB(0.6784,0.4869,0.7366)",
        "RGB(0.7345,0.5829,0.7896)",
        "RGB(0.8182,0.6869,0.8582)",
        "RGB(0.8753,0.7828,0.9034)",
        "RGB(0.9235,0.8382,0.5824)",
        "RGB(0.8627,0.7414,0.4079)",
        "RGB(0.7422,0.6221,0.2617)",
        "RGB(0.6526,0.5451,0.2494)",
        "RGB(0.5249,0.421,0.1967)",
        "RGB(0.3854,0.3189,0.1636)",
        "RGB(0.2686,0.2039,0.05296)",
        "RGB(0.1487,0.1216,0.06311)",
    ]

    extend = 'both'


@register_style('StaticReservoirImpact')
class ContourReservoirImpact(ContourStyle):
    contour_level_list = [
        0.005, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1., 1000.
    ]

    contour_shade_list = [
        "RGB(220, 220, 220)",
        "RGB(194, 194, 194)",
        "RGB(165, 165, 165)",
        "RGB(137, 137, 137)",
        "RGB(108, 108, 108)",
        "RGB(80, 80, 80)",
        "RGB(30, 30, 30)",
        "RGB(5, 5, 5)",
    ]

    extend = 'max'