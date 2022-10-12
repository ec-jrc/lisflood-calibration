import os
import subprocess

if __name__ == "__main__":
    # folder where the .o and .e jobs output files are stored
    # files should be named as LF_cal_XX_<CathcmentID>.e<jobNumber_6digits> and LF_cal_XX_<CathcmentID>.o<jobNumber_6digits>
    # example for catchment 568: LF_cal_01_568.e123456 and LF_cal_01_568.o123456 
    dirname = "/home/username/CatchmentsDone"
    # catchments folder (where "maps" "out" "inflow" and "settings" folders are stored) 
    catchment_folder = "/home/username/catchments/"
    #outputfile = os.path.join(dirname, "results.txt")
    outputfile = "results.csv"
    with open(outputfile, "w") as fout:
        fout.write("Catchment, JobNumber, calibTimeSec, longrunTimeSec, DrainingArea.km2.LDD, inflows, startGenCurrCalibration, elapsedTimeSec, numGensTotal, Termination, KGE, outFolderSizeKB, settingsFolderSizeKB, mapsFolderSizeKB, inflowFolderSizeKB")
        fout.write("\n")
        for file in os.listdir(dirname):
            bAreaFound = 0
            bCalibTimeFound = 0
            bLongRunTimeFound = 0
            bGenFound = 0
            bFirstGenFound = 0
            nNumFirstGen = -1
            bStartGenFound = 0
            bInflowsFound = 0
            filename = os.fsdecode(file)
            if filename[-7]=="e" and filename[-8]==".": 
                catchment = filename[len("LF_cal_XX_"):-8]
                fout.write(catchment + ", ")
                jobNumber = filename[-6:]
                fout.write(jobNumber + ", ")
                fullpath_e = os.path.join(dirname, filename)
                fullpath_o = os.path.join(dirname, filename.replace(".e", ".o", 1))
                with open(fullpath_e, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("real\t") and bCalibTimeFound==1 and bLongRunTimeFound==0:
                            bLongRunTimeFound=1
                            minT=line[5:line.find("m")]
                            secT=line[5+len(minT)+1:line.find(".")]
                            fout.write(str(int(minT)*60+int(secT))+ ", ")
                        if line.startswith("real\t") and bCalibTimeFound==0:
                            bCalibTimeFound=1
                            minT=line[5:line.find("m")]
                            secT=line[5+len(minT)+1:line.find(".")]
                            fout.write(str(int(minT)*60+int(secT))+ ", ")
                if bCalibTimeFound==0:
                    fout.write(", ")    
                if bLongRunTimeFound==0:
                    fout.write(", ")    
                with open(fullpath_o, "r") as f:
                    lines = f.readlines()
                    currLine=-1
                    for line in lines:
                        currLine+=1
                        if line=="Starting calibration\n" and bFirstGenFound==0:
                            bFirstGenFound = 1
                        if line.startswith("Generation ") and bFirstGenFound==1:
                            bFirstGenFound = 2
                            nNumFirstGen = int(line[len("Generation "):len("Generation ")+3].replace(",",""))
                        if line=="Restoring previous calibration state\n" and bGenFound==0:
                            bStartGenFound=1
                        if line.startswith(">> Time elapsed:") and bGenFound==0:
                            if bStartGenFound==0:
                                fout.write(str(nNumFirstGen) + ", ")  
                                # it should be actually zero: fout.write("0, ")
                            else:
                                fout.write(str(nNumFirstGen) + ", ")  
                            fout.write(line[len(">> Time elapsed: "):-3] + ", ")
                            fout.write(lines[currLine-1][len("Done generation "):-1] + ", ")
                            lineKGE="error"
                            if lines[currLine-2]==">> Termination criterion no-improvement KGE fulfilled.\n":
                                if lines[currLine-3]==">> Termination criterion maxGen fulfilled.\n":
                                    lineKGE=lines[currLine-4]
                                    fout.write("maxGen AND no-improvement" + ", ")
                                else:
                                    lineKGE=lines[currLine-3]
                                    fout.write("no-improvement" + ", ")
                            if lines[currLine-2]==">> Termination criterion maxGen fulfilled.\n":
                                if lines[currLine-3]==">> Termination criterion no-improvement KGE fulfilled.\n":
                                    lineKGE=lines[currLine-4]
                                    fout.write("maxGen AND no-improvement" + ", ")
                                else:
                                    lineKGE=lines[currLine-3]
                                    fout.write("maxGen" + ", ")                            
                            generationStr=lines[currLine-1][len("Done generation "):-1]
                            if (lineKGE!="error") and (lineKGE[:len(">> gen: "+ generationStr)]==">> gen: "+ generationStr):
                                fout.write(lineKGE[len(">> gen: "+ generationStr + " effmax_KGE: "):-1] + ", ")
                            else:
                                fout.write("error" + ", ")
                            bGenFound = 1

                        if line.startswith("DrainingArea.km2.LDD          ") and bAreaFound == 0:
                            fout.write(line[len("DrainingArea.km2.LDD          "):-1] + ", ")
                            bAreaFound = 1
                        if line.startswith("Found ") and line.endswith("inflows\n") and bInflowsFound == 0:
                            fout.write(line[len("Found "):-len("inflows\n")-1] + ", ")
                            bInflowsFound = 1
                if bAreaFound==0:
                    fout.write(", ")    
                if bInflowsFound==0:
                    fout.write(", ")
                if bGenFound==0:
                    fout.write(", , , , , ")   
                foldersize=str(subprocess.Popen('du -d 1 ' + os.path.join(catchment_folder, catchment, "out"),shell=True,stdout=subprocess.PIPE).stdout.read())
                foldersize=foldersize[2:foldersize.find("\\t")]
                fout.write(foldersize + ", ")
                foldersize=str(subprocess.Popen('du -d 1 ' + os.path.join(catchment_folder, catchment, "settings"),shell=True,stdout=subprocess.PIPE).stdout.read())
                foldersize=foldersize[2:foldersize.find("\\t")]
                fout.write(foldersize + ", ")
                foldersize=str(subprocess.Popen('du -d 1 ' + os.path.join(catchment_folder, catchment, "maps"),shell=True,stdout=subprocess.PIPE).stdout.read())
                foldersize=foldersize[2:foldersize.find("\\t")]
                fout.write(foldersize + ", ")
                foldersize=str(subprocess.Popen('du -d 1 ' + os.path.join(catchment_folder, catchment, "inflow"),shell=True,stdout=subprocess.PIPE).stdout.read())
                foldersize=foldersize[2:foldersize.find("\\t")]
                fout.write(foldersize)
                fout.write("\n")
