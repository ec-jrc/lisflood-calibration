import os


class LisfloodSettingsTemplate():

    def __init__(self, cfg, subcatch):

        self.obsid = subcatch.obsid
        self.outfix = os.path.join(subcatch.path, os.path.basename(cfg.lisflood_template[:-4]))
        self.lisflood_template = cfg.lisflood_template
        with open(os.path.join('templates', cfg.lisflood_template), "r") as f:
            template_xml = f.read()
    
        template_xml = template_xml.replace('%gaugeloc', subcatch.gaugeloc) # Gauge location
        template_xml = template_xml.replace('%inflowflag', subcatch.inflowflag)
        template_xml = template_xml.replace('%ForcingStart', cfg.forcing_start.strftime('%d/%m/%Y %H:%M')) # Date of forcing start
        template_xml = template_xml.replace('%SubCatchmentPath', subcatch.path)

        self.template_xml = template_xml

    def settings_path(self, suffix, run_rand_id):
        return self.outfix+suffix+run_rand_id+'.xml'

    def write_template(self, run_rand_id, cal_start_local, cal_end_local, param_ranges, parameters):

        out_xml = self.template_xml

        out_xml = out_xml.replace('%run_rand_id', run_rand_id)
        out_xml = out_xml.replace('%CalStart', cal_start_local) # Date of Cal starting
        out_xml = out_xml.replace('%CalEnd', cal_end_local)  # Time step of forcing at which to end simulation

        for ii in range(len(param_ranges)):
            ## DD Special Rule for the SAVA
            if self.obsid == '851' and (param_ranges.index[ii] == "adjust_Normal_Flood" or param_ranges.index[ii] == "ReservoirRnormqMult"):
                out_xml = out_xml.replace('%adjust_Normal_Flood',"0.8")
                out_xml = out_xml.replace('%ReservoirRnormqMult',"1.0")
            out_xml = out_xml.replace("%"+param_ranges.index[ii],str(parameters[ii]))

        out_xml_prerun = out_xml
        out_xml_prerun = out_xml_prerun.replace('%InitLisflood',"1")
        with open(self.outfix+'-PreRun'+run_rand_id+'.xml', "w") as f:
            f.write(out_xml_prerun)

        out_xml_run = out_xml
        out_xml_run = out_xml_run.replace('%InitLisflood',"0")
        with open(self.outfix+'-Run'+run_rand_id+'.xml', "w") as f:
            f.write(out_xml_run)