

if __name__ == '__main__':

    cfg = config.Config(sys.argv[1])

    with open(sys.argv[2], "r") as catchmentFile:
        obsid = int(catchmentFile.readline().replace("\n", ""))

    print(">> Reading Qmeta2.csv file...")
    stations = pandas.read_csv(os.path.join(cfg.path_result, "Qmeta2.csv"), sep=",", index_col=0)

    try:
        station_data = stations.loc[obsid]
    except KeyError as e:
        raise Exception('Station {} not found in stations file'.format(obsid))

    print("=================== "+str(obsid)+" ====================")
    subcatch = subcatchment.SubCatchment(cfg, obsid, station_data)

    lis_template = templates.LisfloodSettingsTemplate(cfg, subcatch)

    hydro_model.generate_benchmark(cfg, subcatch, lis_template)
