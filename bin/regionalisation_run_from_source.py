import os
import shutil
import lisf1


if __name__ == '__main__':

    print("=================== START ===================")
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', help='Source catchments')
    parser.add_argument('--target', help='Target catchments')
    parser.add_argument('--regionalisation', help='Regionalisation type')
    args = parser.parse_args()

    print('----------------------------------------------------------------------------------------------')

    print(f"Source catchment = {args.source}")
    print(f"Target catchment = {args.target}")

    source_id = os.path.basename(args.source)
    target_id = os.path.basename(args.target)

    output_dir = os.path.join(args.target, "out", args.regionalisation)
    if os.path.exists(output_dir) == False:   
        os. makedirs(output_dir, exist_ok=False)

    for run_type in ['PreRun', 'Run']:
        print('***************************************************************************************************')
        print(f"Set up and compute the LISFLOOD {run_type}")

        source_xml = os.path.join(args.source, "settings", f'settings_lisflood{run_type}long_term_run.xml')
        target_xml = os.path.join(args.target, "settings", f'settings_lisflood{run_type}long_term_run.xml')
        with open(source_xml, "r") as f:
            source_template = f.read()
            if '<textvar name="Gauges" value=' in source_template:  
                source_start = source_template.index('<textvar name="Gauges" value=')
                source_end = source_template.index('-7.85 41.15 -7.75 41.15 -7.35 41.15 -7.25 41.15 -7.15 41.15 -7.05 41.15 -5.85 41.75 # Duoro')
                
        with open(target_xml, "r") as f:
            target_template = f.read()
            if '<textvar name="Gauges" value=' in target_template:  
                target_start = target_template.index('<textvar name="Gauges" value=')
                target_end = target_template.index('-7.85 41.15 -7.75 41.15 -7.35 41.15 -7.25 41.15 -7.15 41.15 -7.05 41.15 -5.85 41.75 # Duoro')         

        target_template = target_template.replace(target_template[target_start:target_end], source_template[source_start:source_end])     
        target_template = target_template.replace(f'{args.source}', f'{args.target}')      
        target_template = target_template.replace("out/long_term_run", "out/args.regionalisation")

        regionalisation_xml = os.path.join(args.target, "settings", f'settings_lisflood{run_type}_{args.regionalisation}.xml')
        with open(regionalisation_xml, "w") as f:
            f.write(target_template)

        # lisf1.main(regionalisation_xml)
