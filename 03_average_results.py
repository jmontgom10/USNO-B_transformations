# This is a change for github.

# This script will simply average and print the results of the MCMC fits
import numpy as np
import pdb

# 1) V = O + a1 + a2*(O - E)
a1   = [0.034139, 0.03385, 0.03478, 0.03468, 0.03377]
a2   = [-0.618092, -0.6183, -0.6181, -0.6192, -0.618]
Smat = [[[ 0.96810586, -1.23796727],
         [-1.23796727,  2.46283635]],
        [[ 0.94789571, -1.20076032],
         [-1.20076032,  2.43242864]],
        [[ 0.9446266,  -1.18629838],
         [-1.18629838,  2.40507423]],
        [[ 0.93350045, -1.18628011],
         [-1.18628011,  2.41975797]],
        [[ 0.9483828,  -1.1966002],
         [-1.1966002,  2.42379822]]]

# Convert these entries into arrays
a1   = np.array(a1)
a2   = np.array(a2)
Smat = np.array(Smat)

# Test that the off-diagonal elements of Smat are identical
for iMat, Smat1 in enumerate(Smat):
    if Smat1[0][1] == Smat1[1][0]:
        pass
    else:
        print('# {0}'.format(iMat))
        print(Smat1)
        pdb.set_trace()

# Now that everything is fixed, print the results
print('# 1) V = O + a1 + a2*(O - E)')
print('\n\n a1 = {0:.6}\n a2 = {1:.6}'.format(np.mean(a1), np.mean(a2)))
print('...Covariance Matrix...')
print(np.mean(Smat, axis = 0))

# 2) V = J + a1 + a2*(J - F)
a1   = [-0.02787, -0.02781, -0.02655, -0.02762, -0.02756]
a2   = [-0.5119, -0.5119, -0.5126, -0.5118, -0.5116]
Smat = [
    [[1.01438582, -1.19911867],
     [-1.19911867, -2.20284977]],
    [[1.00210773, -1.18490328],
     [-1.18490328, 2.18516379]],
    [[1.01073856, -1.19291062],
     [-1.19291062, 2.31932918]],
    [[0.99185906, -1.17576594],
     [-1.17576594, 2.19732064]],
    [[1.01203599, -1.19837535],
     [-1.19837535, 2.19840259]]
     ]

# Convert these entries into arrays
a1   = np.array(a1)
a2   = np.array(a2)
Smat = np.array(Smat)

# Test that the off-diagonal elements of Smat are identical
for iMat, Smat1 in enumerate(Smat):
    if Smat1[0][1] == Smat1[1][0]:
        pass
    else:
        print('# {0}'.format(iMat))
        print(Smat1)
        pdb.set_trace()

# Now that everything is fixed, print the results
print('# 2) V = J + a1 + a2*(J - F)')
print('\n\n a1 = {0:.6}\n a2 = {1:.6}'.format(np.mean(a1), np.mean(a2)))
print('...Covariance Matrix...')
print(np.mean(Smat, axis = 0))

###############################################################################
# 3) R = O + a1 + a2*(O - E)
a1   = [-0.04502, -0.04553, -0.04444, -0.04452, -0.04417]
a2   = [-0.5574,  -0.5569,  -0.5580,  -0.5577,  -0.5578 ]
Smat = [
    [[1.01438582, -1.19911867],
     [-1.19911867, -2.20284977]],
    [[1.00210773, -1.18490328],
     [-1.18490328, 2.18516379]],
    [[1.01073856, -1.19291062],
     [-1.19291062, 2.31932918]],
    [[0.99185906, -1.17576594],
     [-1.17576594, 2.19732064]],
    [[1.01203599, -1.19837535],
     [-1.19837535, 2.19840259]]
     ]

# Convert these entries into arrays
a1   = np.array(a1)
a2   = np.array(a2)
Smat = np.array(Smat)

# Test that the off-diagonal elements of Smat are identical
for iMat, Smat1 in enumerate(Smat):
    if Smat1[0][1] == Smat1[1][0]:
        pass
    else:
        print('# {0}'.format(iMat))
        print(Smat1)
        pdb.set_trace()

# Now that everything is fixed, print the results
print('# 3) R = O + a1 + a2*(O - E)')
print('\n\n a1 = {0:.6}\n a2 = {1:.6}'.format(np.mean(a1), np.mean(a2)))
print('...Covariance Matrix...')
print(np.mean(Smat, axis = 0))

# 4) R = J + a1 + a2*(J - F)
########### CRAP... I NEED TO RE-RUN THIS GROUP TWICE!... ##########
a1   = [-0.02745, -0.02745, -0.027, -0.0283, -0.02818]
a2   = [-0.512, -0.5119, -0.5125, -0.5114, -0.5118]
Smat = [
    [[ 1.0001231,  -1.19245798],
     [-1.19245798,  2.21911291]],
    [[ 0.98734931, -1.16663933],
     [-1.16663933,  2.15207279]],
    [[ 0.99165833, -1.17468559],
     [-1.17468559,  2.16779318]],
    [[ 1.01290988, -1.19459225],
     [-1.19459225,  2.19759042]],
    [[ 0.98555027, -1.15843432],
     [-1.15843432,  2.15327946]]]

# Convert these entries into arrays
a1   = np.array(a1)
a2   = np.array(a2)
Smat = np.array(Smat)

# Test that the off-diagonal elements of Smat are identical
for iMat, Smat1 in enumerate(Smat):
    if Smat1[0][1] == Smat1[1][0]:
        pass
    else:
        print('# {0}'.format(iMat))
        print(Smat1)
        pdb.set_trace()

# Now that everything is fixed, print the results
print('# 4) R = J + a1 + a2*(J - F)')
print('\n\n a1 = {0:.6}\n a2 = {1:.6}'.format(np.mean(a1), np.mean(a2)))
print('...Covariance Matrix...')
print(np.mean(Smat, axis = 0))

# 5) (V - R) = a1 + a2*(O - E)
a1   = [0.1413, 0.141238, 0.141156, 0.141537, 0.141311]
a2   = [0.2480, 0.247985, 0.248016, 0.247924, 0.247985]
Smat = [
    [[ 0.0435239,  -0.0539323],
     [-0.0539323,   0.13777418]],
    [[ 0.04354154, -0.05390392],
     [-0.05390392,  0.13775161]],
    [[ 0.0431072,  -0.05314877],
     [-0.05314877,  0.13727533]],
    [[ 0.04327704, -0.05423253],
     [-0.05423253,  0.13890106]],
    [[ 0.04277972, -0.05330   ], # Truncated: disagreement in written records.
     [-0.05330   ,  0.13765856]]
]

# Convert these entries into arrays
a1   = np.array(a1)
a2   = np.array(a2)
Smat = np.array(Smat)

# Test that the off-diagonal elements of Smat are identical
for iMat, Smat1 in enumerate(Smat):
    if Smat1[0][1] == Smat1[1][0]:
        pass
    else:
        print('# {0}'.format(iMat))
        print(Smat1)
        pdb.set_trace()

# Now that everything is fixed, print the results
print('# 5) (V - R) = a1 + a2*(O - E)')
print('\n\n a1 = {0:.6}\n a2 = {1:.6}'.format(np.mean(a1), np.mean(a2)))
print('...Covariance Matrix...')
print(np.mean(Smat, axis = 0))

# 6) (V - R) = a1 + a2*(J - F)
########### CRAP... I NEED TO RE-RUN THIS GROUP TWICE!... ##########
a1   = [0.153327, 0.153074, 0.153294, 0.153166, 0.153301]
a2   = [0.273445, 0.273707, 0.273393, 0.273535, 0.273528]
Smat = [
    [[ 0.05991129, -0.06416461],
     [-0.06416461,  0.15449429]],
    [[ 0.06029676, -0.06478078],
     [-0.06478078,  0.15737808]],
    [[ 0.06018929, -0.06380653],
     [-0.06380653,  0.15601718]],
    [[ 0.06097673, -0.06612431],
     [-0.06612431,  0.15995959]],
    [[ 0.05967482, -0.0635612],
     [-0.0635612,   0.1553452]]
]

# Convert these entries into arrays
a1   = np.array(a1)
a2   = np.array(a2)
Smat = np.array(Smat)

# Test that the off-diagonal elements of Smat are identical
for iMat, Smat1 in enumerate(Smat):
    if Smat1[0][1] == Smat1[1][0]:
        pass
    else:
        print('# {0}'.format(iMat))
        print(Smat1)
        pdb.set_trace()

# Now that everything is fixed, print the results
print('# 6) (V - R) = a1 + a2*(J - F)')
print('\n\n a1 = {0:.6}\n a2 = {1:.6}'.format(np.mean(a1), np.mean(a2)))
print('...Covariance Matrix...')
print(np.mean(Smat, axis = 0))
