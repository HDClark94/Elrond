# Rejection sampling
#    n      Sample size
#    cell   Numeric vector that samples the density over a range of quantiles
#    k      Scale factor for the proposal density (calculated internally by default)
# Example
#    load("spatial_firing.Rdata")
#    dat = spatial_firing$hd_spike_histogram
#    cell = dat[[46]]
#    n = 10000
#    x = rejection.sampling(n, cell)
#    plot(cell, type="l")         # Compare with original density
#    plot(density(x))
rejection.sampling = function(n, cell, seed, k=NULL) {
    # install.packages("movMF")
    if(missing(seed)){
        set.seed(225)
    } else{
        set.seed(seed)
    }
    P = function(x) { cell[x] }                     # Target density
    Q = function(x) { dunif(x, min=1, max=360) }    # Proposal density
    rQ = function(n) { runif(n, min=1, max=360) }   # Proposal random sample
    if ( is.null(k) ) {
        x = seq(1, 360, length=360)
        k = max(P(x) / Q(x))          # Scaling factor k so that k*Q(x) will envelope P(x) "just enough"
    }
    samp = vector(mode="numeric", length=n)
    for ( i in 1:n ) {
        repeat {
            z = rQ(1)
            u = runif(1, 0, k*Q(z))
            if ( u <= P(z) ) break
        }
        samp[i] = z
    }
    return( samp )
}


# Check rejection sampling by plotting a random sample and indicating which points lie in the sample and rejection
# Example
#    load("spatial_firing.Rdata")
#    dat = spatial_firing$hd_spike_histogram
#    cell = dat[[46]]
#    plot.rejection.sampling(1000, cell)
plot.rejection.sampling = function(n, cell, k=NULL) {
    P = function(x) { cell[x] }                     # Target density
    Q = function(x) { dunif(x, min=1, max=360) }    # Proposal density
    rQ = function(n) { runif(n, min=1, max=360) }   # Proposal random sample
    if ( is.null(k) ) {
        x = seq(1, 360, length=360)
        k = max(P(x) / Q(x))          # Scaling factor k so that k*Q(x) will envelope P(x) "just enough"
    }
    x = seq(1, 360, length=360)
    plot(k*Q(x) ~ x, ylim=c(0,max(k*Q(x))), type="l", lty=2, lwd=2)     # Plot proposal envelope (dashed) and target density
    lines(P(x) ~ x, lwd=2)
    for ( i in 1:n ) {
        z = rQ(1)
        u = runif(1, 0, k*Q(z))
        col = ifelse( u <= P(z), "green3", "red")
        points(z,u, pch=16, col=col)
    }
}


# Rescale numeric vector x into range 0:1
rescale = function(x) {
    (x-min(x))/(max(x)-min(x))
}


# Degrees to radians
torad = function(x) {
    x*pi/180
}


# Polar to cartesian coodinates
#     theta    Angle in radians (relative to the positive x-axis)
#     r        Radius
tocartesian = function(theta, r=1) {
    return( cbind(x=r*cos(theta), y=r*sin(theta)) )
}


# Magnitude of vector "a".
magnitude = function(a) {
    as.numeric(sqrt(a%*%a))
}


# Return vector "a" scaled for magnitude=1 without changing direction
unit.vector = function(a) {
    a / magnitude(a)
}


# Draw a circle of radius "r" centered at coordinate given by "centre"
circle = function(r=1, centre=c(0,0), steps=100, ...) {
    angles = (0:steps)*2*pi/steps
    lines(cbind(r*cos(angles)+centre[1], r*sin(angles)+centre[2]), ...)
}


# Return a list of vMF mixtures of 1:k components.
#    x   Numeric vector with range 1:360 (degrees)
#    k   Maximum number of components
# The von-Mises Fisher distribution has two parameters: a mean direction and a concentration parameter.
# Larger values of the concentration parameter generates points that are more tightly clustered about the given mean direction.
# A random sample of an unknown mixture of k 2-dimensional vMF distributions is expected to have k concentrations of points
# around a unit circle. The input vector is a sample of a density representing concentrations in certain directions.
# Points in the sample represent angles, (in range 1:360).
# Each fitted mixture is returned as a matrix "theta" and vector "alpha".
# Matrix "theta" contains the fitted parameters of the mixture components, one row per component.
# The direction and magnitude of each row vector give the direction and concentration of the corresponding component of the mixture.
# Vector "alpha" contains the probabilities of each component, scaled so they sum to 1.
# Example
#    n = 1000
#    cell = dat[[46]]
#    x = rejection.sampling(n, cell)
#    fit = vMFmixture(x)
#    vMFmin(fit)
vMFmixture = function(x, k=5) {
    require(movMF)
    m = tocartesian(torad(x))
    lapply(1:k, function(k) movMF(m,k))
}

# Return the vMF mixture with the min BIC
#    fit   List of vMF mixtures (as returned by vMFmixture)
vMFmin = function(fit) {
    bic = sapply(fit, BIC)
    fit[bic==min(bic)][[1]]
}


# Plot fit on a unit circle, showing probabilities and concentrations of each component.
# Note a single component with low concentration could indicate uniformity (no modes).
# Example
#    n = 1000
#    cell = dat[[46]]
#    x = rejection.sampling(n, cell)
#    fit = vMFmin(vMFmixture(x))
#    plot.vMF(fit)
plot.vMF = function(fit) {
    p = round(fit$alpha, 2)
    conc = round(apply(fit$theta, 1, magnitude), 2)
    plot(NULL, ylim=c(-1.2,1.2), xlim=c(-1.2,1.2), ylab="", xlab="", yaxt="n", xaxt="n")
    axis(side=4, at=0, labels=0, las=1)
    circle(col="grey")
    for ( i in 1:nrow(fit$theta) ) {
        v = unit.vector(as.numeric(fit$theta[i,]))
        arrows(0,0, v[1],v[2], length=0.05, col="red")
        text(rbind(as.numeric(v)*1.1), labels=paste(p[i],"\n",conc[i], sep=""), cex=0.7)
        # text(rbind(as.numeric(v)*1.1), labels=paste(round(fit$theta[i,], 2), sep=""), cex=0.7)
    }
}


get_model_fit_alpha = function(fit) {
    return( fit$alpha )
}


get_model_fit_theta = function(fit) {
    return( fit$theta )
}


get_estimated_density = function(fit){
    angs_cart <- tocartesian(torad(seq(0.1, 360, 0.1)))
    estimated_density <- dmovMF(angs_cart, fit$theta)
    return(estimated_density)
}


plot_modes = function(fit){
    plot.vMF(fit)

}






