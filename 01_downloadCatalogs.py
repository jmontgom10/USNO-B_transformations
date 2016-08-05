# This script will retrieve the Landolt (1992) stars and corresponding USNO-B1.0
# data from Vizier (or some other online service).

import numpy as np
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.table import Table
import pdb

#  Define the name of the output Landolt/USNOB catalog
outputFile = 'landoltStars.csv'

# Reset ROW_LIMIT property to retrieve FULL catalog
Vizier.ROW_LIMIT = -1

# # Retrieve the Landolt (1992) data.
# landoltStars = (Vizier.get_catalogs('II/183A/table2'))[0]
#
# # Construct the output table
# outputNames  = ['_RAJ2000',
#                 '_DEJ2000',
#                 'Star',
#                 'RAJ2000',
#                 'DEJ2000',
#                 'o_Vmag',
#                 'Vmag',
#                 'e_Vmag',
#                 'B-V',
#                 'e_B-V',
#                 'U-B',
#                 'e_U-B',
#                 'V-R',
#                 'e_V-R',
#                 'R-I',
#                 'e_R-I',
#                 'V-I',
#                 'e_V-I',
#                 'Nd',
#                 'Omag',
#                 'Emag',
#                 'Jmag',
#                 'Fmag',
#                 'Nmag']
#
# outputDtypes = ['<f8',
#                 '<f8',
#                 'S11',
#                 'S8',
#                 'S9',
#                 '<i2',
#                 '<f4',
#                 '<f4',
#                 '<f4',
#                 '<f4',
#                 '<f4',
#                 '<f4',
#                 '<f4',
#                 '<f4',
#                 '<f4',
#                 '<f4',
#                 '<f4',
#                 '<f4',
#                 '<i2',
#                 '<f4',
#                 '<f4',
#                 '<f4',
#                 '<f4',
#                 '<f4']

# Try the *UPDATED* catalog (2009AJ....137.4186L)
# but update the "outputNames" to match

###########
# It should be possible to significantly increase the sample by using the more
# recent data at +50 deg. declination found in "2013AJ....146..131L"
###########
landoltStars = (Vizier.get_catalogs('J/AJ/137/4186'))[0]

outputNames  = ['_RAJ2000',
                '_DEJ2000',
                'Name',
                'Vmag',
                'e_Vmag',
                'B-V',
                'e_B-V',
                'U-B',
                'e_U-B',
                'V-R',
                'e_V-R',
                'R-I',
                'e_R-I',
                'V-I',
                'e_V-I',
                'Nobs',
                'Omag',
                'Emag',
                'Jmag',
                'Fmag',
                'Nmag']

outputDtypes = ['<f8',
                '<f8',
                'S11',
                '<f4',
                '<f4',
                '<f4',
                '<f4',
                '<f4',
                '<f4',
                '<f4',
                '<f4',
                '<f4',
                '<f4',
                '<f4',
                '<f4',
                '<i2',
                '<f4',
                '<f4',
                '<f4',
                '<f4',
                '<f4']

# Generate the initial blank table
outputTable = Table(masked = True, names = outputNames, dtype = outputDtypes)

# Initalize lists to store the declinations of the E and F emulsions
F2DecRange = []
E1DecRange = []
# Loop through each Landolt star and retrieve the USNO-B1 data for that object.
for iStar, star in enumerate(landoltStars):
    # Parse the star's name in SimbadName
    simbadName = star['SimbadName'].decode('utf-8')

    # Parse this star's pointing
    RA, Dec  = star['_RAJ2000'], star['_DEJ2000']

    # Skip over Landolt stars more than 10 degrees from the equator.
    if np.abs(Dec) > 20.0:
        print('Star {0} is far from the equator... skipping'.format(simbadName))
        continue

    # Query the USNO-B1.0 data for this object
    USNOB = Vizier.query_object(simbadName,
        radius = 5.0*u.arcsec, catalog='USNO-B1')

    if len(USNOB) > 0:
        # Some match was found, so grab that 'catalog'
        USNOB = USNOB[0]
    else:
        print('Star {0} not matched in USNO-B1.0... skipping'.format(simbadName))
        continue

    # If there is more than one object returned,
    if len(USNOB) > 1:
        # Then find the object closest to the query point and just use that.
        matchInd = np.where(USNOB['_r'].data == USNOB['_r'].data.min())
        USNOB    = USNOB[matchInd]

    # Test if we know where this data came from
    B1Sbool  = (not USNOB['B1S'].data.mask[0])
    R1Sbool  = (not USNOB['R1S'].data.mask[0])
    B2Sbool  = (not USNOB['B2S'].data.mask[0])
    R2Sbool  = (not USNOB['R2S'].data.mask[0])
    IsrcBool = (not USNOB['IS'].data.mask[0])
    surveySourceBool = (B1Sbool or R1Sbool or
                        B2Sbool or R2Sbool or
                        IsrcBool)

    # Parse the USNO-B1 "color" into...
    # O, J, E, F, or N using the following data

    # Surveys used for USNO-B1.0:
    # ----------------------------------------------------------
    # #   Name    Emuls  B/R Wavelen.   Zones  Fld  Dates    Epoch
    #                         (nm)     (Dec)         Obs.
    # ----------------------------------------------------------
    # 0 = POSS-I  103a-O (B) 350-500 -30..+90 936 1949-1965 (1st)
    # 1 = POSS-I  103a-E (R) 620-670 -30..+90 936 1949-1965 (1st)
    # 2 = POSS-II IIIa-J (B) 385-540 +00..+87 897 1985-2000 (2nd)
    # 3 = POSS-II IIIa-F (R) 610-690 +00..+87 897 1985-1999 (2nd)
    # 4 = SERC-J  IIIa-J (B) 385-540 -90..-05 606 1978-1990 (2nd)
    # 5 = ESO-R   IIIa-F (R) 630-690 -90..-05 624 1974-1994 (1st)
    # 6 = AAO-R   IIIa-F (R) 590-690 -90..-20 606 1985-1998 (2nd)
    # 7 = POSS-II IV-N   (I) 730-900 +05..+87 800 1989-2000 (N/A)
    # 8 = SERC-I  IV-N   (I) 715-900 -90..+00 892 1978-2002 (N/A)
    # 9 = SERC-I* IV-N   (I) 715-900 +05..+20  25 1981-2002 (N/A)
    # --------------------------------------------------

    if surveySourceBool == True:
        print('Parsing star {0}'.format(simbadName), end=' ')

        # If there is at least some source information, then add a new row
        outputTable.add_row([0]*len(outputNames), mask = [True]*len(outputNames))

        # Determine the row index of the added null row
        iRow = len(outputTable) - 1

        # Copy the star data into the outputTable
        for col in star.columns:
            if col in outputNames:
                outputTable[col][iRow] = star[col]

        # Extract the sources for the B1/2 and R1/2 magnitudes
        B1S  = USNOB['B1S'].data.data[0]
        R1S  = USNOB['R1S'].data.data[0]
        B2S  = USNOB['B2S'].data.data[0]
        R2S  = USNOB['R2S'].data.data[0]
        Isrc = USNOB['IS'].data.data[0]

        # Treat each magnitude separately
        ############################ B1 magnitudes ############################
        if B1Sbool == True:
            if B1S == 0:
                outputTable['Omag'][iRow] = USNOB['B1mag'].data.data[0]
            else:
                print('B2 source is definitely screwed up', end = '')
                pdb.set_trace()

        ############################ R1 magnitudes ############################
        if R1Sbool == True:
            if R1S == 1:
                outputTable['Emag'][iRow] = USNOB['R1mag'].data.data[0]
            elif R1S == 3:
                print('R1 is from 2nd epoch, and R1 == Fmag', end = '')
                pdb.set_trace()
                outputTable['Fmag'][iRow] = USNOB['R1mag'].data.data[0]
            elif R1S == 5:
                print('R1 is from 1nd epoch, but R1 == Fmag', end = '')
                pdb.set_trace()
                outputTable['Fmag'][iRow] = USNOB['R1mag'].data.data[0]
            elif R1S == 6:
                print('R1 is from 2nd epoch, and R1 == Fmag', end = '')
                pdb.set_trace()
                outputTable['Fmag'][iRow] = USNOB['R1mag'].data.data[0]
            else:
                print('R1 source is definitely screwed up', end = '')
                pdb.set_trace()

        ############################ B2 magnitudes ############################
        if B2Sbool == True:
            if B2S == 2:
                outputTable['Jmag'][iRow] = USNOB['B2mag'].data.data[0]
            elif B2S == 4:
                outputTable['Jmag'][iRow] = USNOB['B2mag'].data.data[0]
            else:
                print('B2 source is definitely screwed up', end = '')
                pdb.set_trace()

        ############################ R2 magnitudes ############################
        if R2Sbool == True:
            if R2S == 1:
                print('R2 is from 1st epoch, and R2 == Emag', end = '')
                pdb.set_trace()
                outputTable['Emag'][iRow] = USNOB['R1mag'].data.data[0]
            elif R2S == 3:
                outputTable['Fmag'][iRow] = USNOB['R1mag'].data.data[0]
            elif R2S == 5:
                print('R2 is from 1st epoch, but R2 == Fmag (*OK*)', end = '')
                outputTable['Fmag'][iRow] = USNOB['R1mag'].data.data[0]
            elif R2S == 6:
                outputTable['Fmag'][iRow] = USNOB['R1mag'].data.data[0]
            else:
                print('R2 source is definitely screwed up', end = '')
                pdb.set_trace()

        ############################# I magnitudes #############################
        if IsrcBool == True:
            if Isrc == 7:
                outputTable['Nmag'][iRow] = USNOB['Imag'].data.data[0]
            elif Isrc == 8:
                outputTable['Nmag'][iRow] = USNOB['Imag'].data.data[0]
            elif Isrc == 9:
                outputTable['Nmag'][iRow] = USNOB['Imag'].data.data[0]
            else:
                print('I magnitude source is definitely screwed up', end = '')
                pdb.set_trace()

        # Now that everything has been parsed, print a newline character
        print('\n', end = '')

    else:
        print('Star {0} has no info on survey source'.format(simbadName))
        print('Assuming emulsions based on band and epoch')
        # If there is at least some source information, then add a new row
        outputTable.add_row([0]*len(outputNames), mask = [True]*len(outputNames))

        # Determine the row index of the added null row
        iRow = len(outputTable) - 1

        # Copy the star data into the outputTable
        for col in star.columns:
            if col in outputNames:
                outputTable[col][iRow] = star[col]

        # Within the Landolt catalog, no stars BREAK the following expected
        # pattern. Thus, we will assume that...
        # Mag --> Emuls
        # B1  --> O
        # B2  --> J
        # R1  --> E
        # R2  --> F
        # I   --> N
        #
        if (not USNOB['B1mag'].data.mask[0]):
            outputTable['Omag'][iRow] = USNOB['B1mag'].data.data[0]
        if (not USNOB['B2mag'].data.mask[0]):
            outputTable['Jmag'][iRow] = USNOB['B2mag'].data.data[0]
        if (not USNOB['R1mag'].data.mask[0]):
            outputTable['Emag'][iRow] = USNOB['R1mag'].data.data[0]
        if (not USNOB['B1mag'].data.mask[0]):
            outputTable['Fmag'][iRow] = USNOB['R2mag'].data.data[0]
        if (not USNOB['Imag'].data.mask[0]):
            outputTable['Nmag'][iRow] = USNOB['Imag'].data.data[0]

# Now that the entire Landolt catalog has been parsed, save it to disk
percentageKept = float(len(outputTable))/float(len(landoltStars))
print('\n{0:4.4g} percent of Landolt Stars parsed'.format(percentageKept))

# Perform the save
outputTable.write(outputFile, format='ascii.csv')

print('Done!')
