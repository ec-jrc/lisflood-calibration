import os


class LisfloodSettingsTemplate():

    def __init__(self, cfg, subcatch):

        self.timestep = cfg.timestep
        self.prerun_timestep = cfg.prerun_timestep
        self.obsid = subcatch.obsid
        settings_dir = os.path.join(subcatch.path, 'settings')
        os.makedirs(settings_dir, exist_ok=True)

        self.outfix = os.path.join(settings_dir, os.path.basename(cfg.lisflood_template[:-4]))
        self.lisflood_template = cfg.lisflood_template
        with open(os.path.join('templates', cfg.lisflood_template), "r") as f:
            template_xml = f.read()
    
        template_xml = template_xml.replace('%gaugeloc', subcatch.gaugeloc) # Gauge location
        template_xml = template_xml.replace('%inflowflag', subcatch.inflowflag)
        template_xml = template_xml.replace('%ForcingStart', cfg.forcing_start.strftime('%d/%m/%Y %H:%M')) # Date of forcing start
        template_xml = template_xml.replace('%SubCatchmentPath', subcatch.path)

        self.template_xml = template_xml

    def settings_path(self, suffix, run_id):
        return self.outfix+suffix+run_id+'.xml'

    def write_template(self, run_id, prerun_start, prerun_end, run_start, run_end, param_ranges, parameters, write_states=False):

        prerun_file = self.settings_path('PreRun', run_id)
        run_file = self.settings_path('Run', run_id)

        out_xml = self.template_xml

        for ii in range(len(param_ranges)):
            ## DD Special Rule for the SAVA --> SAVA belongs to EFAS, these lines must be commented when running the GloFAS calibration
            if self.timestep == 360 and self.obsid == '851' and (param_ranges.index[ii] == "adjust_Normal_Flood" or param_ranges.index[ii] == "ReservoirRnormqMult"):
                out_xml = out_xml.replace('%adjust_Normal_Flood',"0.8")
                out_xml = out_xml.replace('%ReservoirRnormqMult',"1.0")
            out_xml = out_xml.replace("%"+param_ranges.index[ii],str(parameters[ii]))

        # Prerun file
        out_xml_prerun = out_xml
        out_xml_prerun = out_xml_prerun.replace('%InitLisflood',"1")
        out_xml_prerun = out_xml_prerun.replace('%ColdStart', "0")
        out_xml_prerun = out_xml_prerun.replace('%EndMaps', "1")
        out_xml_prerun = out_xml_prerun.replace('%CalStart', prerun_start)
        out_xml_prerun = out_xml_prerun.replace('%CalEnd', prerun_end)
        # do not write tss files of the states during the calibration
        out_xml_prerun = out_xml_prerun.replace('%repDischargeTs', "0")
        out_xml_prerun = out_xml_prerun.replace('%repStateGauges', "0")
        out_xml_prerun = out_xml_prerun.replace('%repRateGauges', "0")
        out_xml_prerun = out_xml_prerun.replace('%repMeteoGauges', "0")
        out_xml_prerun = out_xml_prerun.replace('%repMBTs', "0")
        for data in ['uz', 'uzf', 'uzi']:
            out_xml_prerun = out_xml_prerun.replace(f'%{data}_init', '0')
            out_xml_prerun = out_xml_prerun.replace(f'%{data}_prerun_init', '0')
        for data in ['lz', 'tha', 'thb', 'thc', 'thfa', 'thfb', 'thfc', 'thia', 'thib', 'thic']:
            out_xml_prerun = out_xml_prerun.replace(f'%{data}_init', '-9999')
            out_xml_prerun = out_xml_prerun.replace(f'%{data}_prerun_init', '-9999')
        out_xml_prerun = out_xml_prerun.replace('%run_rand_id', run_id)
        out_xml_prerun = out_xml_prerun.replace('%initialize', '_prerun')
        if self.timestep == 360:  # 6-hourly, this is EFAS
            dt_sec = self.prerun_timestep*60  # daily step for prerun
            out_xml_prerun = out_xml_prerun.replace('%dtsec', f'{dt_sec}')
            out_xml_prerun = out_xml_prerun.replace('%timestep', 'daily')
        
        with open(prerun_file, "w") as f:
            f.write(out_xml_prerun)

        # Run file
        out_xml_run = out_xml
        out_xml_run = out_xml_run.replace('%InitLisflood',"0")
        out_xml_run = out_xml_run.replace('%ColdStart', "1")
        out_xml_run = out_xml_run.replace('%EndMaps', "0")
        out_xml_run = out_xml_run.replace('%CalStart', run_start)
        out_xml_run = out_xml_run.replace('%CalEnd', run_end)
        if write_states:
            out_xml_run = out_xml_run.replace('%repDischargeTs', "1")
            out_xml_run = out_xml_run.replace('%repStateGauges', "1")
            out_xml_run = out_xml_run.replace('%repRateGauges', "1")
            out_xml_run = out_xml_run.replace('%repMeteoGauges', "1")
            out_xml_run = out_xml_run.replace('%repMBTs', "1")
        else:     
            out_xml_run = out_xml_run.replace('%repDischargeTs', "0")
            out_xml_run = out_xml_run.replace('%repStateGauges', "0")
            out_xml_run = out_xml_run.replace('%repRateGauges', "0")
            out_xml_run = out_xml_run.replace('%repMeteoGauges', "0")
            out_xml_run = out_xml_run.replace('%repMBTs', "0")
        init_data = ['uz', 'uzf', 'uzi', 'tha', 'thb', 'thc', 'thfa', 'thfb', 'thfc', 'thia', 'thib', 'thic']
        for data in init_data:
            out_xml_run = out_xml_run.replace(f'%{data}_init', f'$(PathOut)/{data}.end.nc')
            # %{data}_prerun_init added to allow use of %initialize variable in output wrinting of the prerun
            # when using two distinct output variables for prerun and run as in GloFAS calibration (see settings_GloFAS.xml)
            out_xml_run = out_xml_run.replace(f'%{data}_prerun_init', f'$(PathOut)/{data}.end_prerun.nc')
        out_xml_run = out_xml_run.replace('%run_rand_id', run_id)
        out_xml_run = out_xml_run.replace('%initialize', '_run')
        if self.timestep == 360:  # 6-hourly, this is EFAS
            dt_sec = self.timestep*60
            out_xml_run = out_xml_run.replace('%dtsec', f'{dt_sec}')
            out_xml_run = out_xml_run.replace('%timestep', 'hourly')
    
        with open(run_file, "w") as f:
            f.write(out_xml_run)

        return prerun_file, run_file     

    def write_init(self, run_id, prerun_start, prerun_end, run_start, run_end, param_ranges, parameters):

        prerun_file = self.settings_path('PreRun', run_id)
        run_file = self.settings_path('Run', run_id)
 
        out_xml = self.template_xml
        
        # Common parameters
        for ii in range(len(param_ranges)):
            ## DD Special Rule for the SAVA --> SAVA belongs to EFAS, these lines must be commented when running the GloFAS calibration
            if self.timestep == 360 and self.obsid == '851' and (param_ranges.index[ii] == "adjust_Normal_Flood" or param_ranges.index[ii] == "ReservoirRnormqMult"):
                out_xml = out_xml.replace('%adjust_Normal_Flood',"0.8")
                out_xml = out_xml.replace('%ReservoirRnormqMult',"1.0")
            out_xml = out_xml.replace("%"+param_ranges.index[ii],str(parameters[ii]))
        out_xml = out_xml.replace('%InitLisflood', "1")
        out_xml = out_xml.replace('%ColdStart', "0")
        # do not write tss files of the states during the calibration
        out_xml = out_xml.replace('%repDischargeTs', "0")
        out_xml = out_xml.replace('%repStateGauges', "0")
        out_xml = out_xml.replace('%repRateGauges', "0")
        out_xml = out_xml.replace('%repMeteoGauges', "0")
        out_xml = out_xml.replace('%repMBTs', "0")
        for data in ['uz', 'uzf', 'uzi']:
            out_xml = out_xml.replace(f'%{data}_init', '0')
            out_xml = out_xml.replace(f'%{data}_prerun_init', '0')
        for data in ['lz', 'tha', 'thb', 'thc', 'thfa', 'thfb', 'thfc', 'thia', 'thib', 'thic']:
            out_xml = out_xml.replace(f'%{data}_init', '-9999')
            out_xml = out_xml.replace(f'%{data}_prerun_init', '-9999')

        # Prerun file
        out_xml_prerun = out_xml
        out_xml_prerun = out_xml_prerun.replace('%InitLisflood', "1")
        out_xml_prerun = out_xml_prerun.replace('%ColdStart', "0")
        out_xml_prerun = out_xml_prerun.replace('%CalStart', prerun_start)
        out_xml_prerun = out_xml_prerun.replace('%CalEnd', prerun_end)
        out_xml_prerun = out_xml_prerun.replace('%EndMaps', "1")
        out_xml_prerun = out_xml_prerun.replace('%run_rand_id', run_id)
        out_xml_prerun = out_xml_prerun.replace('%initialize', '_prerun')      
        if self.timestep == 360:  # 6-hourly, this is EFAS
            dt_sec = self.prerun_timestep*60
            out_xml_prerun = out_xml_prerun.replace('%dtsec', f'{dt_sec}')
            out_xml_prerun = out_xml_prerun.replace('%timestep', 'daily')    
        
        with open(prerun_file, "w") as f:
            f.write(out_xml_prerun)
 
        # Run file
        out_xml_run = out_xml
        out_xml_run = out_xml_run.replace('%InitLisflood', "0")
        out_xml_run = out_xml_run.replace('%ColdStart', "1")
        out_xml_run = out_xml_run.replace('%CalStart', run_start)
        out_xml_run = out_xml_run.replace('%CalEnd', run_end)
        out_xml_run = out_xml_run.replace('%EndMaps', "0")
        out_xml_run = out_xml_run.replace('%run_rand_id', run_id)
        out_xml_run = out_xml_run.replace('%initialize', '_run')
        if self.timestep == 360:  # 6-hourly, this is EFAS
            dt_sec = self.timestep*60
            out_xml_run = out_xml_run.replace('%dtsec', f'{dt_sec}')
            out_xml_run = out_xml_run.replace('%timestep', 'hourly')

        with open(run_file, "w") as f:
            f.write(out_xml_run)
 
        return prerun_file, run_file            
