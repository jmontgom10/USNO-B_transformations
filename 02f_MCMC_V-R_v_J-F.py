# This script will run a test on the MCMCline function

from MCMCfunc import MCMCfunc
import numpy as np
from scipy import stats
from astropy.table import Table
import matplotlib.pyplot as plt
import corner
import pdb

################################################################################
# DEFINE FUNCTIONS FOR DEALING WITH COVARIANCE MATRICES
################################################################################
def build_cov_matrix(sx, sy, rhoxy):
    # build the covariance matrix from sx, sy, and rhoxy
    cov_matrix = np.matrix([[sx**2,       rhoxy*sx*sy],
                            [rhoxy*sx*sy, sy**2      ]])

    return cov_matrix

# Define two functions for swapping between (sigma_x, sigma_y, theta) and
# (sigma_x, sigma_y, rhoxy)
def convert_angle_to_covariance(sx, sy, theta):
    # Define the rotation matrix using theta
    rotation_matrix = np.matrix([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta),  np.cos(theta)]])

    # Build the eigen value matrix
    lamda_matrix = np.matrix(np.diag([sx, sy]))

    # Construct the covariance matrix
    cov_matrix = rotation_matrix*lamda_matrix*lamda_matrix*rotation_matrix.I

    # Extract the variance and covariances
    sx1, sy1 = np.sqrt(cov_matrix.diagonal().A1)
    rhoxy    = cov_matrix[0,1]/(sx1*sy1)

    return sx1, sy1, rhoxy

def convert_covariance_to_angle(sx, sy, rhoxy):
    # build the covariance matrix from sx, sy, and rhoxy
    cov_matrix = build_cov_matrix(sx, sy, rhoxy)

    # Extract the eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    x_stddev, y_stddev = np.sqrt(eig_vals)

    # Convert the eigen vector into an angle
    y_vec = eig_vecs[:, 0]
    theta = np.arctan2(y_vec[1,0], y_vec[0,0])

    # Make sure the sign of rhoxy is the same as the sign on theta
    if np.sign(rhoxy) != np.sign(theta):
        if np.sign(theta) < 0:
            theta += np.pi
        if np.sign(theta) > 0:
            theta -= np.pi

    return x_stddev, y_stddev, theta

################################################################################
# READ IN THE DATA AND COMPUTE COLORS
################################################################################
# Landolt/USNO-B1.0 catalog file
landoltFile  = 'landoltStars.csv'
landoltStars = Table.read(landoltFile)


# Color relations and plots
################ Relation 1 ################
# U = O + C1*(O - E) + C2
############################################
# Cull any bad values from the table
goodStars = np.logical_not(landoltStars['Jmag'].data.mask)
goodStars = np.logical_and(
    goodStars,
    np.logical_not(landoltStars['Fmag'].data.mask))
goodStars = np.logical_and(
    goodStars,
    np.isfinite(landoltStars['Jmag'].data))
goodStars = np.logical_and(
    goodStars,
    np.isfinite(landoltStars['Fmag'].data))
goodStars = np.logical_and(
    goodStars,
    landoltStars['e_V-R'].data > 0)


# Check that there are plenty of stars for producing a relation
numGood = np.sum(goodStars)
if numGood > 100:
    goodInds     = np.where(goodStars)
    landoltStars = landoltStars[goodInds]
else:
    print('Fewer than 100 stars left... something seems wrong.')
    pdb.set_trace()


# Extract the magnitudes needed for this color-color plot
Jmags    = landoltStars['Jmag'].data.data
Fmags    = landoltStars['Fmag'].data.data

# Construct the colors to plot...
J_F  = Jmags - Fmags
V_R  = landoltStars['V-R'].data.data

# ... and their uncertainties
sig_JF = np.sqrt(2*0.25)
sig_VR = landoltStars['e_V-R'].data.data

# Establish the boundaries of acceptable parameters for the prior
bounds = [(0.23, 0.305),    # Theta (angle of the line slope)
          (0.08, 0.20),     # b_perp (min-dist(line-origin))
          (0.0, 1.0),       # Pb (Probability of sampling an outliers)
          (-8, +8),         # Mx (<x> of outlier distribution)
          (-2.0, 5.0),      # lnVx (log-x-variance of outlier distribution)
          (-8, +8),         # My (<y> of outlier distribution)
          (-2.0, 5.0)]      # lnVy (log-y-variance of outlier distribution)

# Define a prior function for the parameters
def ln_prior(params):
    # We'll just put reasonable uniform priors on all the parameters.
    if not all(b[0] < v < b[1] for v, b in zip(params, bounds)):
        return -np.inf
    return 0

# The "foreground" linear likelihood:
def ln_like_fg(params, x, y, sx, sy):
    theta, b_perp, _, _, _, _, _ = params

    # The following terms allow us to skip over the matrix algebra in the Loop
    # below. These are the absolute value of the components of a unit vector
    # pointed pependicular to the line.
    sinT = np.sin(theta)
    cosT = np.cos(theta)

    # This code can be worked out from some matrix algebra, but it is much
    # faster to avoid going back and forth between matrices, so the explicit
    # form is used here

    # Comput the projected distance of this point from the line
    Di  = cosT*y - sinT*x - b_perp

    # Compute the projected variance of the data
    Si2 = (sinT*sx)**2 + (cosT*sy)**2

    # Compute the final likelihood
    out = -0.5*(((Di**2)/Si2) + np.log(Si2))

    # Return the sum of the ln_likelihoods (product in linear space)
    return out

# The "background" outlier likelihood:
# (For a complete, generalized, covariance matrix treatment, see the file
# "HBL_exer_14_with_outlier_covariance.py")
def ln_like_bg(params, x, y, sx, sy):
    theta, b_perp, Mx, Pb, lnVx, My, lnVy = params

    sinT = np.sin(theta)
    cosT = np.cos(theta)

    # Build the orthogonal-to-line unit vector (and its transpose)
    v_vecT = np.matrix([-sinT, cosT])
    v_vec  = v_vecT.T

    # In parameter terms, "Vx" is the actual variance along the x-axis
    # *NOT* the variance along the eigen-vector. Using this assumption, and
    # forcing the outlier distribution tilt-angle to be equal to the main
    # distribution tilt angle, we get...
    sxOut, syOut, rhoxy = convert_angle_to_covariance(
        np.exp(lnVx), np.exp(lnVy), theta)

    # Compute the elements of the convolved covariance matrix
    # # The old way...
    # out_cov_matrix = build_cov_matrix(sxOut, syOut, rhoxy)
    # data_cov_matrix = np.array([build_cov_matrix(np.sqrt(2*0.25), syi, 0) for syi in sy])
    # testList = []
    # for data_cov_mat1 in data_cov_matrix:
    #     conv_cov_matrix = out_cov_matrix + data_cov_mat1
    #     testList.append((v_vecT.dot(conv_cov_matrix).dot(v_vec))[0,0])
    # Si2test = np.array(testList)

    # The fast way...
    conv_cov_matrix11 = sxOut**2 + sx**2
    conv_cov_matrix12 = rhoxy*sxOut*syOut
    conv_cov_matrix22 = syOut**2 + sy**2

    # Now get the projection of the covariance matrix along the direction
    # orthogonal to the line
    Si2 = (-sinT*(-sinT*conv_cov_matrix11 + cosT*conv_cov_matrix12) +
           cosT*(-sinT*conv_cov_matrix12 + cosT*conv_cov_matrix22))

    # Compute the distances between the data and the line can do it all at once.
    Di = cosT*y - sinT*x - b_perp

    out = -0.5*(((Di**2)/Si2) + np.log(Si2))

    if np.sum(np.isnan(out)) > 0: pdb.set_trace()

    # Return the ln_likelihood of the background model given these-data points
    # Equally likely for ALL data-points
    return out

# Define a likelihood function for the parameters and data
def ln_likelihood(params, x, y, sx, sy):
    # Unpack the parameters
    _, _, Pb, Mx, lnVx, My, lnVy = params

    # Compute the vector of foreground likelihoods and include the Pb prior.
    natLogLike_fg = ln_like_fg(params, x, y, sx, sy)
    arg1 = natLogLike_fg + np.log(1.0 - Pb)

    # Compute the vector of background likelihoods and include the Pb prior.
    natLogLike_bg = ln_like_bg(params, x, y, sx, sy)
    arg2 = natLogLike_bg + np.log(Pb)

    # Combine these using log-add-exp for numerical stability.
    natLogLike = np.sum(np.logaddexp(arg1, arg2))

    # Include a second output as "blobs"
    return natLogLike, (arg1, arg2)

# Now we need to actually run the MCMCfunc using these pre-defined functions
# Simply start the parameters in the center of the bounds
params_guess    = np.array([np.mean(b) for b in bounds])
data            = (J_F, V_R, sig_JF, sig_VR)
n_walkers       = 500
n_burn_in_steps = 200
n_steps         = 2000
sampler = MCMCfunc(ln_prior, ln_likelihood, params_guess, data,
	n_walkers=n_walkers, n_burn_in_steps=n_burn_in_steps, n_steps=n_steps)

# Print out the mean acceptance fraction. In general, acceptance_fraction has an
# entry for each walker so, in this case, it is a 250-dimensional vector.
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

# Estimate the integrated autocorrelation time for the time series in each
# parameter.
print("Autocorrelation time:", sampler.get_autocorr_time())

plt.ion()
################################################################################
# USE THIS CODE TO CHECK IF THE BURNING WAS LONG ENOUGH TO START THE WALKERS AT
# WELL DISTRIBUTED POSITIONS
################################################################################
# figure1, axes = plt.subplots(len(params_guess), sharex=True, figsize=(8,6))
#
# #Y-axis labels and other pleasantries
# chain_shape = np.shape(sampler.chain)
# axes[0].set_title('Walker Paths', fontsize='large')
# axes[5].set_xlabel('Step Number', fontsize='large')
# axes[5].set_xlim(0,chain_shape[1])
#
# labels = [
#     r"$\theta$",
#     r"$b_p$",
#     r"$P_b$",
#     r"$M_x$",
#     r"$\ln V_x$",
#     r"$M_y$",
#     r"$\ln V_y$"]
#
# for i, lab in enumerate(labels):
#     axes[i].set_ylabel(lab, fontsize='large')
#
# #Plot the walkers
# for walkers in range(chain_shape[0]):
#     for params in range(chain_shape[2]):
#         axes[params].plot(sampler.chain[walkers,:,params], linewidth=0.5,
#             alpha=0.5, color='k')
# pdb.set_trace()
################################################################################

# pdb.set_trace()
# Compute the range of acceptable values from the marginalized posterior
# distributions for each parameter. These will use report median +# or -#
truthRanges = [(v[1], v[2]-v[1], v[1]-v[0]) for v in
    zip(*np.percentile(sampler.flatchain, [16, 50, 84], axis=0))]
truths = [t[0] for t in truthRanges]

# Provide labels for each of the parameters
labels = [
    r"$\theta$",
    r"$b_p$",
    r"$P_b$",
    r"$M_x$",
    r"$\ln V_x$",
    r"$M_y$",
    r"$\ln V_y$"]

# Use the "corner" class to produce a posterior distribution plot
corner.corner(sampler.flatchain, bins=50, range=bounds, labels=labels,
    truths=truths)

################################################################################
# COMPUTE THE POSTERIOR PROBABILITY THAT EACH DATA-POINT IS "GOOD"
################################################################################
norm = 0.0
post_prob = np.zeros(len(data[0]))
for i in range(sampler.chain.shape[1]):
    for j in range(sampler.chain.shape[0]):
        ll_fg, ll_bg = sampler.blobs[i][j]
        post_prob += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
        norm += 1
post_prob /= norm

# print('Probability that each point is good')
# for xpt, ypt, xyGoodProb in zip(data[0], data[1], post_prob):
#     print('({0}, {1}) -- {2}'.format(xpt, ypt, xyGoodProb))

################################################################################
# PLOT THE ACTUAL DATA AND THE BEST FIT (WITH A SAMPLE OF ACCEPTABLE FITS DRAWN
# FROM THE MARINALIZED POSTERIEOR DISTRIBUTION).
################################################################################
plt.figure()
x1, y1 = data[0:2]

# Plot down the basic data-points and freeze the axes
plt.scatter(x1, y1, marker='+', color='k')

# Mark the outliers (all points with post_prob < 0.99)
outBools = (post_prob < 0.99)
if np.sum(outBools) > 0:
    outInds = np.where(outBools)
    xOut, yOut =  x1[outInds], y1[outInds]
    outProb    = post_prob[outInds]

    plt.scatter(xOut, yOut, marker='s',
        c=outProb, cmap="gray_r", vmin=0, vmax=1)

# Set the blot range, and turn-off autoscaling
plt.axis((-2.0, 5.0, -0.6, 1.6))
plt.autoscale(False)

# Plot the slope +/- sigma_intrinsic lines
# Start by including a "typical errorbar"
plt.errorbar(-1.0, 1.25, xerr=data[2], yerr=np.median(data[3]), color='k')

# Plot 100 sampled lines
# Grab the x-axes extent for the current plot.
xl = np.array(plt.xlim())
# Grab JUST the samples for the slope and intercept
samples = sampler.flatchain[:,0:2]
for theta, b_perp in samples[np.random.randint(len(samples), size=100)]:
    m, b = np.tan(theta), b_perp/np.cos(theta)
    plt.plot(xl, m*xl + b, color="k", alpha=0.1)

# Plot the best fitting line line
mBest, bBest = np.tan(truths[0]), truths[1]/np.cos(truths[0])
plt.plot(xl, mBest*xl + bBest, color="k", lw=1.5)

plt.xlabel("$J - F$")
plt.ylabel("$V - R$")

# Figure out the linear fit uncertainties
# Start by transforming the samples into slope / intercept space
transformedSamples = np.array(
    (np.tan(sampler.flatchain[:,0]),
     sampler.flatchain[:,1]/np.cos(sampler.flatchain[:,0])))
transformedSamples = transformedSamples.T

# Grab the quartiles of the transformed samples
slopeIntRanges = [(v[1], v[2]-v[1], v[1]-v[0]) for v in
    zip(*np.percentile(transformedSamples, [16, 50, 84], axis=0))]

# Now estimate the best fit using the transformed sample
a2,  a1  = slopeIntRanges[0][0], slopeIntRanges[1][0]
sa2, sa1 = np.mean(slopeIntRanges[0][1:3]), np.mean(slopeIntRanges[1][1:3])

print('\nCongratulations, you are now master of the USNO-B1.0 data!\n')
print('(V - R) = a1 + a2*(O - E)')
print('with \n a1 = {0:.6}\t+/- {1:.6} \n a2 = {2:.6}\t+/- {3:.6}'.format(
    a1, sa1, a2, sa2))

# Compute the covariance matrix of the linear fit parameters (slope, intercept)
# First collecet the relevant samples and transform the monte-carlo
# (theta, b_perp) values into (slope, intercept) values.
data  = np.array([np.tan(samples[:,0]), samples[:,1]/np.cos(samples[:,0])])
# # Count the nuber of data points
# npts  = len(samples)
# # This covariance matrix has only two degrees of freedom
# ddof  = 2
# # Each data-point should be repeated only one time (not using binned data)
# freq  = np.ones(npts)
# # Each data-point has equal weigting
# awts  = np.ones(npts)
#
# # The following steps were taken from the examples in the numpy.cov() function
# # help file.
# wts   = freq*awts
# V1    = np.sum(wts)
# V2    = np.sum(wts*awts)
# data1 = data - np.sum(data*wts, axis = 1, keepdims=True)/V1
# cov   = np.dot(data1 * wts, data1.T) * V1 / (V1**2 - ddof * V2)

# The above code is good for reference, but here is teh actual numpy method
cov = np.cov(data)

print('The co-variance matrix for the fitted parameters is')
print('1000x')
print('[[     Sm**2,  rho*Sm*Sb ]')
print(' [ rho*Sm*Sb,  Sb**2     ]]\n')
print(1000.0*cov)

pdb.set_trace()
print('done')
