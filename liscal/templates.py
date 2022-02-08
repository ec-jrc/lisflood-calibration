import os
import copy

class LisfloodSettingsTemplate():

    def __init__(self, cfg, subcatch, nthreads='1'):

        self.obsid = subcatch.obsid
        settings_dir = os.path.join(subcatch.path, 'settings')
        os.makedirs(settings_dir, exist_ok=True)

        self.outfix = os.path.join(settings_dir, os.path.basename(cfg.lisflood_template[:-4]))
        self.lisflood_template = cfg.lisflood_template
        with open(os.path.join('templates', cfg.lisflood_template), "r") as f:
            template_xml = f.read()
    
        template_xml = template_xml.replace('%nthreads', nthreads)
        template_xml = template_xml.replace('%gaugeloc', subcatch.gaugeloc) # Gauge location
        template_xml = template_xml.replace('%inflowflag', subcatch.inflowflag)
        template_xml = template_xml.replace('%ForcingStart', cfg.forcing_start.strftime('%d/%m/%Y %H:%M')) # Date of forcing start
        template_xml = template_xml.replace('%SubCatchmentPath', subcatch.path)

        self.template_xml = template_xml

    def settings_path(self, suffix, run_id):
        return self.outfix+suffix+run_id+'.xml'

    def write_init(self, run_id, prerun_start, prerun_end, run_start, run_end, param_ranges, parameters):

        prerun_file = self.settings_path('PreRun', run_id)
        run_file = self.settings_path('Run', run_id)
        
        out_xml = copy.deepcopy(self.template_xml)

        out_xml = out_xml.replace('%run_rand_id', run_id)

        for ii in range(len(param_ranges)):
            ## DD Special Rule for the SAVA
            if self.obsid == '851' and (param_ranges.index[ii] == "adjust_Normal_Flood" or param_ranges.index[ii] == "ReservoirRnormqMult"):
                out_xml = out_xml.replace('%adjust_Normal_Flood',"0.8")
                out_xml = out_xml.replace('%ReservoirRnormqMult',"1.0")
            out_xml = out_xml.replace("%"+param_ranges.index[ii],str(parameters[ii]))

        out_xml_prerun = out_xml
        out_xml_prerun = out_xml_prerun.replace('%InitLisflood', "1")
        out_xml_prerun = out_xml_prerun.replace('%CalStart', prerun_start)
        out_xml_prerun = out_xml_prerun.replace('%CalEnd', prerun_end)
        with open(prerun_file, "w") as f:
            f.write(out_xml_prerun)

        out_xml_run = out_xml
        out_xml_run = out_xml_run.replace('%InitLisflood', "1")
        out_xml_run = out_xml_run.replace('%CalStart', run_start)
        out_xml_run = out_xml_run.replace('%CalEnd', run_end)
        with open(run_file, "w") as f:
            f.write(out_xml_run)

        return prerun_file, run_file

    def write_template(self, run_id, cal_start_local, cal_end_local, param_ranges, parameters):

        out_xml = copy.deepcopy(self.template_xml)

        out_xml = out_xml.replace('%run_rand_id', run_id)
        out_xml = out_xml.replace('%CalStart', cal_start_local) # Date of Cal starting
        out_xml = out_xml.replace('%CalEnd', cal_end_local)  # Time step of forcing at which to end simulation
        print(cal_start_local)
        print(cal_end_local)

        for ii in range(len(param_ranges)):
            ## DD Special Rule for the SAVA
            if self.obsid == '851' and (param_ranges.index[ii] == "adjust_Normal_Flood" or param_ranges.index[ii] == "ReservoirRnormqMult"):
                out_xml = out_xml.replace('%adjust_Normal_Flood',"0.8")
                out_xml = out_xml.replace('%ReservoirRnormqMult',"1.0")
            out_xml = out_xml.replace("%"+param_ranges.index[ii],str(parameters[ii]))

        out_xml_prerun = out_xml
        out_xml_prerun = out_xml_prerun.replace('%InitLisflood',"1")
        with open(self.outfix+'-PreRun'+run_id+'.xml', "w") as f:
            f.write(out_xml_prerun)

        out_xml_run = out_xml
        out_xml_run = out_xml_run.replace('%InitLisflood',"0")
        with open(self.outfix+'-Run'+run_id+'.xml', "w") as f:
            f.write(out_xml_run)
