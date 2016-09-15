# This script will run a test on the MCMCline function

from MCMCfunc import MCMCfunc
import os
import numpy as np
from scipy import stats
from astropy.table import Table
import pdb
import matplotlib as mpl

# FOR EPS PLOTS
mpl.use('PS')
figName = 'V-O_v_O-E'
figDir  = os.path.join('figs', figName)
figFmt  = 'eps'

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import corner

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
# V = O + C1*(O - E) + C2
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

# Check that there are plenty of stars for producing a relation
numGood = np.sum(goodStars)
if numGood > 100:
    goodInds     = np.where(goodStars)
    landoltStars = landoltStars[goodInds]
else:
    print('Fewer than 100 stars left... something seems wrong.')
    pdb.set_trace()


# Extract the magnitudes needed for this color-color plot
Omags    = landoltStars['Omag'].data.data
Emags    = landoltStars['Emag'].data.data
Vmags    = landoltStars['Vmag'].data.data
sVmags   = landoltStars['e_Vmag'].data.data

# Construct the colors to plot
O_E  = Omags - Emags
V_O  = Vmags - Omags
sig_OE = np.sqrt(2*0.25)
sig_VO = np.sqrt(0.25)

# Provide labels for each of the parameters
labels = [
    r"$\theta$",
    r"$b_p$",
    r"$P_b$",
    r"$M_x$",
    r"$\ln V_x$",
    r"$M_y$",
    r"$\ln V_y$"]

plotLabels = [
    r"$\theta$",
    r"$b_p$"]

# Establish the boundaries of acceptable parameters for the prior
bounds = [(-0.65, -0.45),     # Theta (angle of the line slope)
          (-0.2, 0.2),    # b_perp (min-dist(line-origin))
          (0.0, 1.0),       # Pb (Probability of sampling an outliers)
          (-8, +8),         # Mx (<x> of outlier distribution)
          (0, 5),           # lnVx (log-x-variance of outlier distribution)
          (-8, +8),         # My (<y> of outlier distribution)
          (0, 5)]           # lnVy (log-y-variance of outlier distribution)

plotBounds = [(-0.65, -0.45),     # Theta (angle of the line slope)
              (-0.2, 0.2)]    # b_perp (min-dist(line-origin))

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

    # Construct the covariance matrix of a TILTED-GAUSSIAN
    out_cov_matrix = build_cov_matrix(sxOut, syOut, rhoxy)

    # Construct the covariance matrix of the data-point (identical for all data)
    data_cov_matrix = build_cov_matrix(sx, sy, 0)

    # Convolve the two gaussians (simply add their covairance matrices)
    conv_cov_matrix = out_cov_matrix + data_cov_matrix

    # Now get the projection of the covariance matrix along the direction
    # orthogonal to the line
    Si2 = (v_vecT.dot(conv_cov_matrix).dot(v_vec))[0,0]

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

# Setup the plotting parameters###############################################################################
# Prepare the plotting parameters for this figure
###############################################################################
aspect  = 1.0 # For corner plots, make things square
# emulateapj.cls uses these values
#
#\textwidth=7.1in
#\columnsep=0.3125in
#\parindent=0.125in
#\voffset=-20mm
#\hoffset=-7.5mm
#
# ==> column width = (7.1in - 0.3125in)/2 = 3.39375in

# Prepare the x-axis lengths
xsize   = 3.39375                 # Set the figure width to full text width
lmargin = 0.45/xsize              # Set the left margin size (l-inches/width)
rmargin = 0.12/xsize              # Set the right margin fraction (r-inches/width)
# Preapre the y-axis lengths
bmargin = 0.45/xsize              # Set the bottom margin fraction
tmargin = 0.12/xsize              # Set the top margin fraction (colorbar labels)

b     = (1 - (lmargin + rmargin)) #Calculate the xsize-normalized plot region width
a     = (b*aspect)                #Calculate the xsize-normalized plot region height
ysize = (bmargin + a + tmargin)*xsize #Calculate the figure height

# Compute rectangle corners (x-axis)
x1 = lmargin                      #Calculate the normalized plot positions
x2 = x1 + b
xc = x2 + rmargin
# Compute rectangle corners (y-axis)
y1 = bmargin*(xsize/ysize)
y2 = y1 + a*(xsize/ysize)
yc = y2 + tmargin*(xsize/ysize)

#Test that everything adds to one
print("(x_corner, y_corner) = ({0}, {1})".format(
    np.round(xc, 6), np.round(yc, 6)))
print('Should be (1.0, 1.0)')

# Now build a table to stor fitted parameters and covariance matrix values
fitValTable = Table(
    names=('m', 'b', 's2_m', 's2_b', 'rho_sm_sb'),
    dtype=('f8', 'f8', 'f8', 'f8', 'f8'))

# Now we need to actually run the MCMCfunc using these pre-defined functions
# Simply start the parameters in the center of the bounds
params_guess    = np.array([np.mean(b) for b in bounds])
data            = (O_E, V_O, sig_OE, sig_VO)

# Sampler values for the actual run...
# n_walkers       = 500
# n_burn_in_steps = 150
# n_steps         = 2000

# Sampler values for debugging...
n_walkers       = 50
n_burn_in_steps = 150
n_steps         = 1000

# Number of times to repeat sampling process.
n_loops = 1

for iLoop in range(n_loops):
    print('Starting MCMC loop {0:d}'.format(iLoop + 1))

    # Execute the MCMC sampler
    sampler = MCMCfunc(ln_prior, ln_likelihood, params_guess, data,
    	n_walkers=n_walkers, n_burn_in_steps=n_burn_in_steps, n_steps=n_steps)

    # Print out the mean acceptance fraction. In general, acceptance_fraction has an
    # entry for each walker so, in this case, it is a 250-dimensional vector.
    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

    # Estimate the integrated autocorrelation time for the time series in each
    # parameter.
    print("Autocorrelation time:", sampler.get_autocorr_time())

    samplerData = sampler.flatchain[:, 0:2]

    # Compute the range of acceptable values from the marginalized posterior
    # distributions for each parameter. These will use report median +# or -#
    truthRanges = [(v[1], v[2]-v[1], v[1]-v[0]) for v in
        zip(*np.percentile(samplerData, [16, 50, 84], axis=0))]
    truths = [t[0] for t in truthRanges]

    ###############################################
    #### TREAT COVARIANCE OF FITTED PARAMETERS ####
    ###############################################
    # Figure out the linear fit uncertainties
    # Start by transforming the samples into slope / intercept space
    transformedSamples = np.array(
        [np.tan(sampler.flatchain[:,0]),
         sampler.flatchain[:,1]/np.cos(sampler.flatchain[:,0])])

    # Grab the quartiles of the transformed samples
    slopeIntRanges = [(v[1], v[2]-v[1], v[1]-v[0]) for v in
        zip(*np.percentile(transformedSamples, [16, 50, 84], axis=1))]

    # Now estimate the best fit using the transformed sample
    a2,  a1  = slopeIntRanges[0][0], slopeIntRanges[1][0]
    sa2, sa1 = np.mean(slopeIntRanges[0][1:3]), np.mean(slopeIntRanges[1][1:3])

    print('\nCongratulations, you are now master of the USNO-B1.0 data!\n')
    print('V = O + a1 + a2*(O - E)')
    print('with \n a1 = {0:.4}\t+/- {1:.4} \n a2 = {2:.4}\t+/- {3:.4}'.format(
        a1, sa1, a2, sa2))

    # Compute the covariance matrix of the linear fit parameters (slope, intercept)
    # First collecet the relevant samples and transform the monte-carlo
    # (theta, b_perp) values into (slope, intercept) values.

    # The above code is good for reference, but here is teh actual numpy method
    cov = np.cov(transformedSamples.T)

    print('The co-variance matrix for the fitted parameters is')
    print('1000x')
    print('[[     Sm**2,  rho*Sm*Sb ]')
    print(' [ rho*Sm*Sb,  Sb**2     ]]\n')
    print(1000.0*cov)

    # Add a row to the table...
    fitValTable.add_row((a2, a1, cov[0,0], cov[1,1], cov[0,1]))

# Save the table
fitValTable.write(figName + '_fitVals.csv', format='ascii.csv')

################################################################################
# Generate figures
################################################################################
print('Generating example figures from final sampling run')

# Initalize the figure frame
dataDim    = samplerData.shape[1]
fig1, axes = plt.subplots(dataDim, dataDim, figsize=(xsize, ysize))

# Define the key word arguments for the corner plot 2D histograms
hist2d_kwargs = hist2d_kwargs = {"plot_contours":True,
    "no_fill_contours":True,"fill_contours":False,"plot_density":False,
    "plot_datapoints":False}

# Use the "corner" class to produce a posterior distribution plot
fig1 = corner.corner(samplerData, fig = fig1,
    bins=50, range=plotBounds, truths=truths,
    hist2d_kwargs = hist2d_kwargs,
    labels=plotLabels, label_kwargs={'fontsize':8})

pdb.set_trace()
# Reset the figure margins to conservatively use space...
fig1.subplots_adjust(left=lmargin ,right=rmargin, top=tmargin ,bottom=bmargin)

# Loop through all the remaining axes in the figure
for ax in fig1.get_axes():
    # Loop through all the x-tick labels on those axes and set fontsize
    [l.set_fontsize(8) for l in ax.get_xticklabels()]

    # Now do the same for the y-axis...
    # Loop through all the y-tick labels on those axes and set fontsize
    [l.set_fontsize(8) for l in ax.get_yticklabels()]

# Construct the figure name
fig1Name = figName + '_post{0:d}.eps'.format(iLoop+1)
fig1Path = os.path.join(figDir, fig1Name)
fig1.savefig(fig1Path, format = figFmt)
pdb.set_trace()

# Get rid of the figure
plt.clf()
del fig1

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

################################################################################
# PLOT THE ACTUAL DATA AND THE BEST FIT (WITH A SAMPLE OF ACCEPTABLE FITS DRAWN
# FROM THE MARINALIZED POSTERIEOR DISTRIBUTION).
################################################################################
fig2 = plt.figure(figsize=(xsize, ysize))

# Initalize the axes to fill the specified rectangle shape
# rect is in the form [left, bottom, width, height] (normalized coords)
rect2 = (x1, y1, (x2-x1), (y2-y1))
ax2   = fig2.add_axes(rect2)

# Grab just the (x, y) photometry values for scatter plotting
xData, yData = data[0:2]

# Create a new colormap for the scatter plot markers.
# Color will indicate the marginalized posterior probability that a given
# data point is an outlier.
new_cmap = truncate_colormap(plt.get_cmap("gray_r"), 0.5, 1.0, n=100)

# Plot down the basic data-points and freeze the axes
ax2.scatter(xData, yData, marker='.', s=22, c=post_prob, cmap=new_cmap,
    edgecolors='none', zorder = 1000)

# Set the plot range, and turn-off autoscaling
ax2.axis((-2.0, 5.0, -4.3, 2.1))
ax2.autoscale(False)

# Plot the slope +/- sigma_intrinsic lines
# Start by including a "typical errorbar"
ax2.errorbar(-1.0, -3.0, xerr=data[2], yerr=data[3], color='k')

# Create a vector spaning the x-range of the plot
xl = np.array(ax2.get_xlim())

# Perform the linear algebra to compute the fill region of "acceptable fits"
xSample   = np.linspace(xl.min(), xl.max(), 200)
Amatrix   = np.vander(xSample, 2)
fitVals   = np.array(
    [np.tan(sampler.flatchain[:,0]),
    sampler.flatchain[:,1]/np.cos(sampler.flatchain[:,0])]).T
lines     = np.dot(fitVals, Amatrix.T)
quantiles = np.percentile(lines, [16, 84], axis=0)

# Fill in 1-sigma region with a red polygon
ax2.fill_between(xSample, quantiles[0], quantiles[1], color="r", zorder = 999)

# Plot the best fitting line line
mBest, bBest = np.tan(truths[0]), truths[1]/np.cos(truths[0])
ax2.plot(xl, mBest*xl + bBest, color="k", lw=1.5, zorder=1001)

ax2.set_xlabel("$O - E$")
ax2.set_ylabel("$V - O$")

# Construct the figure name
fig2Name = figName + '_MCMC{0:d}.eps'.format(iLoop+1)
fig2Path = os.path.join(figDir, fig2Name)
fig2.savefig(fig2Path, format = figFmt)

# Get rid of the figure
plt.clf()
del fig2


print('done')
