import os
import numpy as np
import xarray as xr
import pandas as pd
import calendar
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import gridspec
from matplotlib import patches
from matplotlib import transforms
from matplotlib import ticker

from liscal import binary_scores, hydro_stats


def create_products(cfg, subcatch, obj):

    # create output directory
    os.makedirs(cfg.summary_path, exist_ok=True)

    # Long term run has run_id X
    run_id = 'X'

    # compute statistics (KGE, NSE, etc.)
    Q, stats = obj.compute_statistics(run_id)
    print(Q)
    print(stats)
    print(Q.index)

    # compute monthly discharge data
    sim_monthly, obs_monthly = hydro_stats.split_monthly(Q.index, Q['Sim'].values, Q['Obs'].values, spinup=subcatch.spinup)

    # get return periods at station coordinates
    thresholds = xr.open_dataset(cfg.return_periods).sel(x=subcatch.data['LisfloodX'], y=subcatch.data['LisfloodY'])
    print(thresholds)

    # create speedometer plots
    speedo = SpeedometerPlot(cfg.plot_params)
    speedo.plot(os.path.join(subcatch.path_out, 'speedo'), stats)
    os.system('convert {0}.svg {0}.pdf'.format(os.path.join(subcatch.path_out, 'speedo')))

    # create box plot
    box = MonthlyBoxPlot(cfg.plot_params)
    box.plot(os.path.join(subcatch.path_out, 'boxy'), sim_monthly, obs_monthly)
    os.system('convert {0}.svg {0}.pdf'.format(os.path.join(subcatch.path_out, 'boxy')))

    # create time series plot
    ts = TimeSeriesPlot(cfg.plot_params)
    ts.plot(os.path.join(subcatch.path_out, 'timmy'), Q.index, Q['Sim'].values, Q['Obs'].values, thresholds)
    os.system('convert {0}.svg {0}.pdf'.format(os.path.join(subcatch.path_out, 'timmy')))

    # compute contingency table and export
    # contingency_values = binary_scores.contingency_table(thresholds, Q)
    # contingency_df = pd.DataFrame(data=contingency_values, index=subcatch.obsid)
    # print(contingency_df)
    # contingency_df.to_csv(path.join(cfg.summary_path, 'contingency_table_{}.csv'.format(subcatch.obsid)))


# Plotting class for speedometer gauges, modified to match Louise Arnal's EFAS4.0 designs
# Original code from https://nicolasfauchereau.github.io/climatecode/posts/drawing-a-gauge-with-matplotlib/
class SpeedometerPlot():
    
    darkestblue = [i / 255. for i in (57, 16, 139)]
    darkerblue = [i / 255. for i in (58, 68, 214)]
    lighterblue = [i / 255. for i in (86, 148, 254)]
    lighestblue = [i / 255. for i in (160, 201, 254)]
    grey = [i / 255. for i in (196, 198, 201)]
    
    def __init__(self, plot_params):
        self.plot_params = plot_params

        # KGE and correlations plots: non symmetric speedometers
        self.colors_nonsym = [self.grey, self.lighestblue, self.lighterblue, self.darkerblue, self.darkestblue]
        self.labels_nonsym = ['$<$ 0.2', '0.2\n-\n0.4', '0.4\n-\n0.6', '0.6\n-\n0.8', '0.8\n-\n1.0']

        # Bias and variability ratios: symmetric speedometers
        self.colors_sym = [self.grey, self.lighterblue, self.darkerblue, self.darkestblue, self.darkerblue, self.lighterblue, self.grey]
        self.labels_sym = ['$<$ 0.5', '0.5\n-\n0.7', '0.7\n-\n0.9', '0.9\n-\n1.1', '1.1\n-\n1.3', '1.3\n-\n1.5', '$>$ 1.5']

        self.title_size_big = plot_params.title_size_big
        self.title_size_small = plot_params.title_size_small

    def rot_text(self, ang):
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation

    def degree_range(self, n):
        start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
        end = np.linspace(0,180,n+1, endpoint=True)[1::]
        mid_points = start + ((end-start)/2.)
        return np.c_[start, end], mid_points

    def gauge(self, ax, labels=['', '', '', '', ''], colors='jet_r', arrow=1, title='', score=None, fontsize=None):
        
        # some sanity checks first
        if arrow > 1:
            raise Exception("\n\nThe arrow position can't be larger than 100%.\n\n")
        
        # if colors is a string, we assume it's a matplotlib colormap and we discretize in n_lables discrete colors
        n_lables = len(labels)
        if isinstance(colors, str):
            cmap = cm.get_cmap(colors, n_lables)
            cmap = cmap(np.arange(n_lables))
            colors = cmap[::-1, :].tolist()
        if isinstance(colors, list):
            if len(colors) == n_lables:
                colors = colors[::-1]
            else:
                raise Exception("\n\nnumber of colors {} not equal to number of categories{}\n".format(len(colors), n_lables))
        
        # begins the plotting
        ang_range, mid_points = self.degree_range(n_lables)
        labels = labels[::-1]
        
        # plots the sectors and the arcs
        for ang, c in zip(ang_range, colors):
            # sectors
            ax.add_patch(patches.Wedge((0., 0.), .4, *ang, facecolor='white', lw=2))
            # arcs
            ax.add_patch(patches.Wedge((0., 0.), .4, *ang, width=0.20, edgecolor='white', facecolor=c, lw=2, alpha=1.0))

        # set the labels
        for i, (mid, lab) in enumerate(zip(mid_points, labels)):
            if colors[i] == self.darkerblue or colors[i] == self.darkestblue:
                text_color = 'w'
            else:
                text_color = 'k'
            ax.text(0.3 * np.cos(np.radians(mid)), 0.3 * np.sin(np.radians(mid)), lab, 
                horizontalalignment='center', verticalalignment='center',
                fontsize=fontsize, fontweight='bold', 
                rotation=self.rot_text(mid), color=text_color)

        # set the title
        ax.text(0, 0.01, title, horizontalalignment='center', verticalalignment='center', fontsize=fontsize, fontweight='bold')
        
        # Calculate arrow angle
        pos = 180 - arrow * 180
        # normal arrow
        ax.arrow(
            0, 0, 0.3 * np.cos(np.radians(pos)), 0.3 * np.sin(np.radians(pos)),
            width=0.01, head_width=0.01, head_length=0.2, facecolor='black', edgecolor='white',
            head_starts_at_zero=True, length_includes_head=True
        )
        # inverted arrow
        # ax.arrow(
        #     0.499 * np.cos(np.radians(pos)), 0.499 * np.sin(np.radians(pos)),
        #     -0.3 * np.cos(np.radians(pos)), -0.3 * np.sin(np.radians(pos)),
        #     width=0.01, head_width=0.01, head_length=0.2, facecolor='black', edgecolor='white',
        #     head_starts_at_zero=True, length_includes_head=True
        # )

        # Value label
        if score is not None:
            ax.text(0.45 * np.cos(np.radians(pos)), 0.45 * np.sin(np.radians(pos)), "{0:.2f}".format(score),
                horizontalalignment='center', verticalalignment='center',
                fontsize=2.5*fontsize, fontweight='bold',
                rotation=self.rot_text(pos), color='k')

        # removes frame and ticks, and makes axis equal and tight
        ax.set_frame_on(False)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axis('equal')

    def plot(self, path_out, stats):

        kge = stats['kge']
        corr = stats['corr']

        # Calculate relative position in %
        mini = 0.3;
        maxi = 1.7
        bias = stats['bias']
        bias_prop = max(min((bias - mini) / (maxi - mini), 1), 0)
        # Extract values for spread
        spread = stats['spread']
        spread_prop = max(min((spread - mini) / (maxi - mini), 1), 0)

        plot_params = self.plot_params

        # Update the font before creating any plot objects
        plt.rc('font', **self.plot_params.text['font'])
        plt.rc('text', **self.plot_params.text['text'])
        plt.rc('figure', **self.plot_params.text['figure'])

        # Start figure
        fig = plt.figure()
        # Define a subplot positioning grid
        gs = gridspec.GridSpec(3, 3, figure=fig)
        # KGE
        ax1 = fig.add_subplot(gs[:2, :])
        self.gauge(ax=ax1,
            labels=self.labels_nonsym,
            colors=self.colors_nonsym,
            arrow=kge, title='KGE',
            score=kge, fontsize=self.title_size_big)
        # Correlation
        ax2 = fig.add_subplot(gs[2, 0])
        self.gauge(ax=ax2, 
            labels=self.labels_nonsym,
            colors=self.colors_nonsym, 
            arrow=corr, title='Correlation', 
            score=corr, fontsize=self.title_size_small)
        # Bias ratio
        ax3 = fig.add_subplot(gs[2, 1])
        self.gauge(ax=ax3,
            labels=self.labels_sym,
            colors=self.colors_sym,
            arrow=bias_prop, title='Bias ratio',
            score=bias, fontsize=self.title_size_small)
        # Spread ratio
        ax4 = fig.add_subplot(gs[2, 2])
        self.gauge(ax=ax4,
            labels=self.labels_sym,
            colors=self.colors_sym,
            arrow=spread_prop, title='Variability ratio',
            score=spread, fontsize=self.title_size_small)

        fig.set_size_inches(16.5, 11.7) # A3 size
        fig.subplots_adjust(left=0.1, bottom=0, right=1, top=1, wspace=-0.2, hspace=0.0)

        # Save the figure
        plt.savefig(path_out+'.'+self.plot_params.file_format, format=self.plot_params.file_format)


class MonthlyBoxPlot():

    pink = [i / 255. for i in (236, 118, 218)]
    lightpurple = [i / 255. for i in (114, 119, 223)]
    purple = [i / 255. for i in (91, 63, 159)]
    darkestblue = [i / 255. for i in (23, 125, 245)]
    darkerblue = [i / 255. for i in (91, 155, 213)]
    lighterblue = [i / 255. for i in (131, 179, 223)]
    lightestblue = [i / 255. for i in (189, 215, 238)]
    grey = [i / 255. for i in (196, 198, 201)]

    def __init__(self, plot_params):
        self.plot_params = plot_params

        self.label_size = plot_params.label_size
        self.axes_size = plot_params.axes_size
        self.legend_size_small = plot_params.legend_size_small

    def wing_plot(self, boxplot, ax, filler=None):
        
        n = len(boxplot['medians'])
        
        # Calculate median
        median = []
        for i in boxplot['medians']:
            median += [i.get_ydata()[0]]
        
        # Wings of 3 stdevs (+/- Q3+1.5*IQR)
        p5 = []
        p95 = []
        for ii, i in enumerate(boxplot['whiskers']):
            ydata = i.get_ydata()
            if len(ydata) == 0:
                p5 += [np.nan]
                p95 += [np.nan]
            else:
                if ydata[0] <= median[int(ii/2)]:
                    p5 += [ydata[1]]
                elif ydata[0] >= median[int(ii/2)]:
                    p95 += [ydata[1]]
                else:
                    print("Invalid value for data: " + str(ydata[0]))
        
        # Wings of 1 stdev (+/- Q1 and Q3)
        p25 = []
        p75 = []
        for i in boxplot['boxes']:
            ydata = i.get_ydata()
            if len(ydata) == 0:
                p25 += [np.nan]
                p75 += [np.nan]
            else:
                p25 += [min(ydata)]
                p75 += [max(ydata)]
        
        # Wings covering outliers
        p1 = []
        p99 = []
        for ii, i in enumerate(boxplot['fliers']):
            p1 += [p75[ii]]
            ydata = i.get_ydata()
            if len(ydata) == 0:
                p99 += [1.01*p1[ii]]
            else:
                p99 += [max(ydata)]
        
        # Layer it all on the plot in the right order
        if filler is None:
            try:
                ax.fill_between(np.arange(1,n+3), [p1[-3]] + p1 + [p1[2]], [p99[-3]] + p99 + [p99[2]],
                    facecolor=self.lightestblue, alpha=1.0, edgecolor=None)
            except ValueError:
                print("OUCH1")
            try:
                ax.fill_between(np.arange(1,n+3), [p5[-3]] + p5 + [p5[2]], [p95[-3]] + p95 + [p95[2]],
                    facecolor=self.lighterblue, alpha=1.0, edgecolor=None)
            except ValueError:
                print("OUCH2")
            try:
                ax.fill_between(np.arange(1,n+3), [p25[-3]] + p25 + [p25[2]], [p75[-3]] + p75 + [p75[2]],
                    facecolor=self.darkerblue, alpha=1.0, edgecolor=None)
            except ValueError:
                print("OUCH3")
            # median line
            try:
                plt.plot(np.arange(1, n + 3), [median[-3]] + median + [median[2]], color=self.darkestblue, linewidth=3)
            except ValueError:
                print("OUCH4")
        else:
            ax.fill_between(np.arange(1, n + 3), [filler] + p1 + [filler], [filler] + p99 + [filler],
                facecolor=self.lightestblue, alpha=1.0, edgecolor=None)
            ax.fill_between(np.arange(1, n + 3), [filler] + p5 + [filler], [filler] + p95 + [filler],
                facecolor=self.lighterblue, alpha=1.0, edgecolor=None)
            ax.fill_between(np.arange(1, n + 3), [filler] + p25 + [filler], [filler] + p75 + [filler],
                facecolor=self.darkerblue, alpha=1.0, edgecolor=None)
            plt.plot(np.arange(1, n + 3), [np.nan] + median + [np.nan], color=self.darkestblue, linewidth=3)
        
        return (p1, p5, p25, median, p75, p95, p99)

    def apply_boxplot_theme(self, boxplot):
        for i in boxplot['boxes']:
            i.set_color(self.purple)
            i.set_facecolor(self.lightpurple)
            i.set_edgecolor(self.purple)
            i.set_linewidth(1.5)
        for i in boxplot['whiskers'] + boxplot['caps'] + boxplot['fliers']:
            i.set_color(self.purple)
            i.set_fillstyle('full')
            i.set_markerfacecolor(self.lightpurple)
            i.set_markeredgecolor(self.purple)
            i.set_linewidth(2)
        for i in boxplot['medians']:
            i.set_color(self.purple)
            i.set_fillstyle('full')
            i.set_linewidth(2)
        for i in boxplot['means']:
            i.set_color(self.pink)
            i.set_fillstyle('full')
            i.set_linewidth(1.5)

    def gen_legend(self, fontsize):
        
        prettydata = [
            0,
            1, 1, 1, 1, 1,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7,
            5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            7, 7, 7, 7, 7, 7, 7,
            8, 8 ,8, 8, 8,
            9, 9, 9, 9,
            10, 10,
            14,
            16
        ]
        
        # add it to the figure
        fig = plt.gcf()
        
        # temprorary axes to get boxplot values
        axd = fig.add_subplot(566)
        boxplot = axd.boxplot([prettydata, prettydata], notch=True, bootstrap=10000)
        plt.delaxes(axd)
        
        # new axes to plot legend
        axl = fig.add_subplot(555)
        
        # boxplot
        legendbox = axl.boxplot(prettydata, notch=True, sym='.', bootstrap=100, showmeans=False, meanline=True, patch_artist=True, widths=0.5)
        self.apply_boxplot_theme(legendbox)
        
        # corresponding wingplot
        (p1, p5, p25, median, p75, p95, p99) = self.wing_plot(boxplot, axl, filler=np.nan)
        axl.set_xlim(-0.5, 8)
        axl.set_ylim(-3, 20)
        plt.axis('off')
        
        # annotations
        axl.text(x=3.5, y=(p99[0]+p95[0])/2, s='outliers', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
        axl.text(x=3.5, y=p95[0], s='Q3 + 1.5 IQR', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
        axl.text(x=3.5, y=p75[0], s='Q3: 75th perc.', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
        axl.text(x=3.5, y=median[0], s='median', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
        axl.text(x=3.5, y=p25[0], s='Q1: 25th perc.', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
        axl.text(x=3.5, y=p5[0], s='Q1 - 1.5 IQR', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
        axl.text(x=1, y=-1.5, s='Qsim', color='black', verticalalignment='center', horizontalalignment='center', fontweight='bold', family='fantasy', fontsize=fontsize+2, weight='bold')
        axl.text(x=2.5, y=-1.5, s='Qobs', color='black', verticalalignment='center', horizontalalignment='center', fontweight='bold', family='fantasy', fontsize=fontsize+2, weight='bold')
        
        # Pretty box around it
        bb = transforms.Bbox([[0.5, -2.5], [6.5, 17]])
        p_bbox = patches.FancyBboxPatch((bb.xmin, bb.ymin), abs(bb.width), abs(bb.height), 
            boxstyle="round", edgecolor="k", facecolor='white', zorder=0)
        axl.add_patch(p_bbox)

    def gen_legend2(self, fig, gs, fontsize):
        
        prettydata = [
            0,
            1, 1, 1, 1, 1,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7, 4.7,
            5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5,
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            7, 7, 7, 7, 7, 7, 7,
            8, 8 ,8, 8, 8,
            9, 9, 9, 9,
            10, 10,
            14,
            16
        ]
        
        # temprorary axes to get boxplot values
        axd = fig.add_subplot(gs[1:4, 6:])
        boxplot = axd.boxplot([prettydata, prettydata], notch=True, bootstrap=10000)
        plt.delaxes(axd)
        
        # new axes to plot legend
        axl = fig.add_subplot(gs[1:4, 6:])
        
        # boxplot
        legendbox = axl.boxplot(prettydata, notch=True, sym='.', bootstrap=100, showmeans=False, meanline=True, patch_artist=True, widths=1)
        self.apply_boxplot_theme(legendbox)
        
        # corresponding wingplot
        (p1, p5, p25, median, p75, p95, p99) = self.wing_plot(boxplot, axl, filler=np.nan)
        axl.set_xlim(-0.5, 8)
        axl.set_ylim(-3, 20)
        plt.axis('off')
        
        # annotations
        axl.text(x=3.5, y=(p99[0]+p95[0])/2, s='outliers', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
        axl.text(x=3.5, y=p95[0], s='Q3 + 1.5 IQR', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
        axl.text(x=3.5, y=p75[0], s='Q3: 75th perc.', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
        axl.text(x=3.5, y=median[0], s='median', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
        axl.text(x=3.5, y=p25[0], s='Q1: 25th perc.', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
        axl.text(x=3.5, y=p5[0], s='Q1 - 1.5 IQR', color='black', verticalalignment='center', family='fantasy', fontsize=fontsize)
        axl.text(x=1, y=-1.5, s='Qsim', color='black', verticalalignment='center', horizontalalignment='center', fontweight='bold', family='fantasy', fontsize=fontsize+2, weight='bold')
        axl.text(x=2.5, y=-1.5, s='Qobs', color='black', verticalalignment='center', horizontalalignment='center', fontweight='bold', family='fantasy', fontsize=fontsize+2, weight='bold')
        
        # Pretty box around it
        bb = transforms.Bbox([[0.5, -2.5], [6.5, 17]])
        p_bbox = patches.FancyBboxPatch((bb.xmin, bb.ymin), abs(bb.width), abs(bb.height), 
            boxstyle="round", edgecolor="k", facecolor='white', zorder=0)
        axl.add_patch(p_bbox)

    def month2string(self, m):
        if m==1:
            return 'Jan'
        elif m==2:
            return 'Feb'
        elif m==3:
            return 'Mar'
        elif m==4:
            return 'Apr'
        elif m==5:
            return 'May'
        elif m==6:
            return 'Jun'
        elif m==7:
            return 'Jul'
        elif m==8:
            return 'Aug'
        elif m==9:
            return 'Sep'
        elif m==10:
            return 'Oct'
        elif m==11:
            return 'Nov'
        elif m==12:
            return 'Dec'
        else:
            raise Exception('Invalid month digit given. Should be 1 ~ 12')

    def plot(self, path_out, sim_monthly, obs_monthly):

        # FIGURE OF MONTHLY CLIMATOLOGY FOR CALIBRATION PERIOD
        # Update the font before creating any plot objects
        plt.rc('font', **self.plot_params.text['font'])
        plt.rc('text', **self.plot_params.text['text'])
        plt.rc('figure', **self.plot_params.text['figure'])
        plt.rc('axes', **self.plot_params.text['axes'])
        plt.rcParams["font.size"] = 14
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"

        months = np.array([9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # water year
        fig = plt.figure()
        gs = gridspec.GridSpec(5, 8, figure=fig)
        
        # Get obs data from boxplot and delete it
        axd = plt.axes()
        obs_box = axd.boxplot([obs_monthly[i - 1] for i in months])
        plt.delaxes(axd)
        
        # create axes
        ax = fig.add_subplot(gs[:, :6]) #plt.axes()
        
        # Make the obs wings plot
        (p1, p5, p25, median, p75, p95, p99) = self.wing_plot(obs_box, ax)
        
        # Plot the sim as boxplots
        sim_box = ax.boxplot([np.ones((len(months))) * np.nan] + [sim_monthly[i - 1] for i in months],
            notch=True, sym='.', bootstrap=10000, showmeans=False, meanline=True, patch_artist=True,
            widths=0.5)
        
        self.apply_boxplot_theme(sim_box)

        # Esthetics
        # ax.set_title('Monthly discharge climatology in calibration period', fontsize=titleFontSize)
        ax.grid(b=True, axis='y')

        # horizontal axis
        plt.xlabel(r'Month', fontsize=self.label_size)
        plt.xticks(range(1, 16), [""] + [self.month2string(m) for m in months])
        plt.xlim([1.5, 15.5])

        # vertical axis
        plt.ylabel(r'Discharge [m3/s]', fontsize=self.label_size)
        ax.tick_params(labelsize=self.axes_size, size=8, width=2, which='major')
        ax.tick_params(size=4, width=1.5, which='minor')

        # Add manually-made legend
        fontsize = fig.get_axes()[0].get_xticklabels()[0].get_fontsize() / 2.0 * 0.8
        # self.gen_legend(fontsize=self.legend_size_small)
        self.gen_legend2(fig, gs, fontsize=self.legend_size_small)

        # Restore the correct active axes
        ax = fig.get_axes()[0]
        plt.sca(ax)

        # Maximize the window for optimal view
        fig.set_size_inches(16.5, 11.7) # A3 size
        fig.subplots_adjust(left=0.1, bottom=0, right=1, top=1, wspace=-0.2, hspace=0.0)
        
        # linear scale
        max_value = 0
        for obs in obs_monthly:
            max_obs = np.nanmax(obs)
            max_value = np.nanmax([max_value, max_obs])
        for sim in sim_monthly:
            max_sim = np.nanmax(sim)
            max_value = np.nanmax([max_value, max_sim])
        logscale = False
        if logscale:
          plt.yscale(r'log')
          ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000])
          ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
          plt.ylim([1, 2 * max_value])
        else:
          plt.yscale(r'linear')
          plt.ylim([1, 1.05 * max_value])
        
        # Save the linear scale figure
        plt.savefig(path_out+'.'+self.plot_params.file_format, format=self.plot_params.file_format)


class TimeSeriesPlot():

    magenta = [i / 255. for i in (191, 81, 225)]
    red = [i / 255. for i in (255, 29, 29)]
    orange = [i / 255. for i in (250, 167, 63)]
    green = [i / 255. for i in (112, 173, 71)]
    purple = [i / 255. for i in (91, 63, 159)]
    lighterblue = [i / 255. for i in (131, 179, 223)]
    darkestblue = [i / 255. for i in (23, 125, 245)]

    def __init__(self, plot_params):
        self.plot_params = plot_params

        self.axes_size = plot_params.axes_size
        self.label_size = plot_params.label_size
        self.threshold_size = plot_params.threshold_size

    def plot_threshold(self, ax, mindate, maxdate, max_value, threshold, color, label):
        if max_value > threshold:
            time_period = maxdate-mindate
            plt.hlines(threshold, color=color, linewidth=2, linestyle=':', xmin=mindate, xmax=maxdate)
            ax.text(x=maxdate+time_period*0.01, y=threshold, s=label, color=color, verticalalignment='center', fontsize=self.threshold_size)
            ax.set_xlim(mindate-time_period*0.05, maxdate+time_period*0.1)


    def plot(self, path_out, index, sim, obs, thresholds):

        # FIGURE OF CALIBRATION PERIOD TIME SERIES
        # Update the font before creating any plot objects
        plt.rc('font', **self.plot_params.text['font'])

        fig = plt.figure()
        ax = plt.axes()
        dates = [i.value * 1e-09 for i in index]
        # Qsim
        plt.plot(dates, sim, color=self.purple, linewidth=1)
        # Qobs
        ax.fill_between(dates, np.zeros(len(obs)), obs, facecolor=self.lighterblue,
                        alpha=1.0, edgecolor=self.darkestblue, linewidth=0.5)
        # Return period
        mindate = index[0].value * 1e-09
        maxdate = index[-1].value * 1e-09
        max_value = np.nanmax([sim, obs])
        self.plot_threshold(ax, mindate, maxdate, max_value, thresholds['rl1.5'], self.green, label='1.5-year')
        self.plot_threshold(ax, mindate, maxdate, max_value, thresholds['rl2'], self.orange, label='2-year')
        self.plot_threshold(ax, mindate, maxdate, max_value, thresholds['rl5'], self.red, label='5-year')
        self.plot_threshold(ax, mindate, maxdate, max_value, thresholds['rl20'], self.magenta, label='20-year')

        plt.ylabel(r'Discharge [m3/s]', fontsize=self.label_size)  # ³
        
        # Activate major ticks at the beginning of each boreal season
        period = (index[-1] - index[0]).days/365.25
        if period > 10:
            majorticks = [
                calendar.timegm(datetime(k, j, 1).timetuple())
                for k in np.arange(index[0].year, index[-1].year + 1, 1)
                for j in [9]
                if datetime(k, j, 1) >= index[0] and datetime(k, j, 1) <= index[-1]
            ]
        else:
            majorticks = [
                calendar.timegm(datetime(k, j, 1).timetuple())
                for k in np.arange(index[0].year, index[-1].year + 1, 1)
                for j in [3, 6, 9, 12]
                if datetime(k, j, 1) >= index[0] and datetime(k, j, 1) <= index[-1]
            ]
        ax.set_xticks(majorticks)
        
        # Rewrite labels
        locs, intlabels = plt.xticks()
        plt.setp(intlabels, rotation=70)
        labels = [datetime.strftime(datetime.fromtimestamp(i), "%b %Y") for i in majorticks]
        ax.set_xticklabels(labels)
        
        # Activate minor ticks every month
        minorticks = [
            calendar.timegm(datetime(k, j + 1, 1).timetuple())
            for k in np.arange(index[0].year, index[-1].year + 1, 1)
            for j in range(12)
            if datetime(k, j + 1, 1) >= index[0] and datetime(k, j + 1, 1) <= index[-1]
        ]
        
        # For the minor ticks, use no labels; default NullFormatter.
        ax.xaxis.set_minor_locator(ticker.FixedLocator(minorticks))
        ax.tick_params(labelsize=self.axes_size, size=8, width=2, which='major')
        ax.tick_params(size=4, width=1.5, which='minor')

        # Maximize the window for optimal view
        fig.set_size_inches(16.5, 11.7) # A3 size
        fig.subplots_adjust(left=0.1, bottom=0, right=1, top=1, wspace=-0.2, hspace=0.0)

        # DD better to always place the legend box to avoid random placement and risking not being able to read KGE NSE etc.
        leg = ax.legend(['Qsim', 'Qobs'], fancybox=True, framealpha=0.8, prop={'size': self.threshold_size*.75}, labelspacing=0.1,
                        loc='center', bbox_to_anchor=(0.5, -0.3))
        leg.get_frame().set_edgecolor('white')
        
        # linear scale
        logscale = False
        if logscale:
          plt.yscale(r'log')
          ax.set_yticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000])
          ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
          plt.ylim([1, 2 * max_value])
        else:
          plt.yscale(r'linear')
          plt.ylim([1, 1.05 * max_value])

        # Save the linear scale figure
        plt.savefig(path_out+'.'+self.plot_params.file_format, format=self.plot_params.file_format)
