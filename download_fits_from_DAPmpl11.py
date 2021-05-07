#!/usr/bin/env python

import os
import pandas as pd


pw_file = "/Users/mmckay/Desktop/sdss_pw.txt"  # SDSS Collab password file

# Downloading data DAP Maps and Pipe3D Maps from MaNGA MPL11

# DAP Maps for BBRD sample
# Read in csv with BreakBRD plateifu
bbrd_table = pd.read_csv(
    "/Users/mmckay/Desktop/research/FMR_MZR/final_MaNGAdr16_bbrd_crossmatch.csv"
)

# Interate through the plateifu and download MPL11 DAP Maps
plateifu_list = bbrd_table["plateifu"]
for plateifu in plateifu_list:
    plate, ifu = plateifu.split("-")

    # Retrieves DAP Maps
    os.system(
        "rsync -avz --password-file {} rsync://sdss@dtn01.sdss.utah.edu/sas/mangawork/manga/spectro/analysis/MPL-11/HYB10-MILESHC-MASTARSSP/{}/{}/*manga-{}-{}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz /Volumes/lil_oynx/bbrd_dapfits".format(
            pw_file, plate, ifu, plate, ifu
        )
    )

    # Retrieve Pipe3D maps
    os.system(
        "rsync -avz --password-file {} rsync://sdss@dtn01.sdss.utah.edu/sas/mangawork/manga/sandbox/pipe3d/v3_1_1/3.1.1/{}/*manga-{}-{}.Pipe3D.cube.fits.gz /Volumes/lil_oynx/bbrd_Pipe3Dfits".format(
            pw_file, plate, plate, ifu
        )
    )

# DAP Maps for LG12 sample
# # Read in csv with LG12 plateifu

lg12_table = pd.read_csv(
    "/Users/mmckay/Desktop/research/FMR_MZR/final_MaNGAdr16_lg12_crossmatch.csv"
)

# Interate through the plateifu and download MPL11 DAP Maps
plateifu_list = lg12_table["plateifu"]
for plateifu in plateifu_list:
    plate, ifu = plateifu.split("-")
    os.system(
        "rsync -avz --password-file {} rsync://sdss@dtn01.sdss.utah.edu/sas/mangawork/manga/spectro/analysis/MPL-11/HYB10-MILESHC-MASTARSSP/{}/{}/*manga-{}-{}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz /Volumes/lil_oynx/lg12_dapfits".format(
            pw_file, plate, ifu, plate, ifu
        )
    )

    # Retrieve Pipe3D maps
    os.system(
        "rsync -avz --password-file {} rsync://sdss@dtn01.sdss.utah.edu/sas/mangawork/manga/sandbox/pipe3d/v3_1_1/3.1.1/{}/*manga-{}-{}.Pipe3D.cube.fits.gz /Volumes/lil_oynx/lg12_Pipe3Dfits".format(
            pw_file, plate, plate, ifu
        )
    )

