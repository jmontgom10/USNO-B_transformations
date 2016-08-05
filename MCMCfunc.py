import numpy as np
import emcee
import pdb

def MCMCfunc(ln_prior, ln_like, params_guess, data,
    n_walkers=100, n_burn_in_steps=250, n_steps=1000):

    '''This function takes provided log-likelihood and log-prior functions and
    performs an MCMC sampling of the log-posterior-probability distribution.

    parameters:

    ln_prior -- This is a function object hich takes a set of parameter
                hypotheses and returns the prior (assumed) probability of those
                hypotheses.

                E.g. ln_prior(params)

                Since the user defines the "ln_prior" function outside the
                current function, it can operate however they like so long as it
                takes the "params" dictionary as defined above and returns a
                single, float value of the log of the prior probability.

    ln_like  -- This is a function object which takes a set of hypotheses and
                data as arguments and returns the likelihood of those hypothesis
                given the data.

                E.g. ln_like(params, data)

    params   -- This must be a dictionary with string keys corresponding to the
                names of the parameters to be sampled. The value associated with
                each key of the params dictionary should be a float-point value.

    data     -- This must be a dictionary with string keys corresponding to the
                names of the parameters to be sampled.

                The exact nature of the "params" and "data" arguments does not
                really matter, but to allow this procedure to be generalized to
                a variety of models, the user must find a way to write the
                "ln_prior" and "ln_like" function to depend *only* on these two
                arguments. If this critera is met, then the "ln_probability"
                function defined within the current function will properly
                work, and the MCMC sampler should succeed.


    '''
    # To begin with, we need to actually APPLY Baye's rule to construct the
    # log-proability function. This is simply the log of the product of the
    # prior and likelihood functions, or the sum of the log of the prior and the
    # log of the liklihood function. This function will only take the two
    # dictionaries provided

    # First, test if the ln_like function is returning blobs and set the boolean
    # value to signal the ln_probability function to also use these blobs
    global blobsBool, iteration, lastPercent, burn_in
    test_lnLike = ln_like(params_guess, *data)
    usingBlobs  = isinstance(test_lnLike, tuple)

    def ln_probability(params, *args):
        global blobsBool

        ########################################################################
        # USE THIS CODE TO PRINT PROGRESS UPDATES
        ########################################################################
        global iteration, lastPercent, burn_in

        # Increment the iteration each time the function is called
        iteration  +=1

        if burn_in:
            # If this is being called by a burn-in run, then compute the
            # percentage of the burn in completed
            thisPercent = np.int(100.0*iteration/
                (n_walkers*n_burn_in_steps + n_walkers -1))
        else:
            # If this is being called by a production run, then compute the
            # percentage of the production completed
            thisPercent = np.int(100.0*iteration/
                (n_walkers*n_steps + n_walkers -1))

        # Updating every call dramatically slows down the run, so test if new
        # on-screen text is even required
        if thisPercent > lastPercent:
            print("Sampler progress: {0:3d}%".format(thisPercent), end='\r')
            lastPercent = thisPercent
        ########################################################################

        # Call the prior function, check that it doesn't return -inf then create
        # the log-probability function
        natLogPrior = ln_prior(params)
        if not np.isfinite(natLogPrior):
            # If there is not a finite probability of the sampled parameter,
            # then simple return the most negative possible value

            if usingBlobs:
                return -np.inf, None
            else:
                return -np.inf
        else:
            # Otherwise return Ln(posterior) = Ln(prior) + Ln(Likelihood) + C
            # (We can ignore the "C" value because it's just a scaling constant)
            natLogLikelihood = ln_like(params, *args)

            # Test for any "blobs" output from the ln_like function
            if usingBlobs:
                natLogLikelihood, blob = natLogLikelihood

                # Combine the ln_like and ln_prior outputs to get ln_posterior.
                natLogPostProb = natLogPrior + natLogLikelihood

                return natLogPostProb, blob
            else:
                # Combine the ln_like and ln_prior outputs to get ln_posterior.
                natLogPostProb = natLogPrior + natLogLikelihood

                return natLogPostProb

    # Almost there! Now we must initialize our walkers. Remember that emcee uses
    # a bunch of walkers, and we define their starting distribution. If you have
    # an idea of where your best-fit parameters will be, you can start the
    # walkers in a small Gaussian bundle around that value (as I am doing).
    # Otherwise, you can start them evenly across your parameter space (that is
    # limited by the priors). This will require more walkers and more steps.
    n_dim = len(params_guess)

    # Setup the initial positions of the random walkers for the MCMC sampling
    p0 = np.array(params_guess)
    p0 = [p0 + 1e-5*np.random.randn(n_dim) for k in range(n_walkers)]

    #Finally, you're ready to set up and run the emcee sampler
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_probability,
        args=data)

    # Run the burn-in
    print('Running burn-in...')
    burn_in = True
    iteration = -1
    lastPercent = -1
    output = sampler.run_mcmc(p0, n_burn_in_steps)
    if len(output) > 3:
        pos, prob, state, blobs = output
    else:
        pos, prob, state = output
    print('')

    print("Running production...")
    sampler.reset()
    burn_in = False
    iteration = -1
    lastPercent = -1
    output = sampler.run_mcmc(pos, n_steps, rstate0=state)
    if len(output) > 3:
        pos, prob, state, blobs = output
    else:
        pos, prob, state = output
    print('')

    return sampler
