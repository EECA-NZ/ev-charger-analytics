# Charger Demand Estimation


### Research Question

Suppose the only information we had about an EV charging site with a single charger was when it is plugged into and unplugged from an EV, over (say) one year. Could we infer the number of EVs that missed out on getting charged because it was busy? There is an obvious problem: the blocked arrivals go unobserved! And yet they are the very ones we want to know about! Proposed modelling solution: Treat EV arrivals as independent events with a time-varying density, i.e. as an inhomogeneous Poisson process.

* Assume the density is periodic (e.g. in time of day or of week). In this project, due to lack of data, we assume periodicity only in time of day.

* An EV arriving at the charger can find it either "Available" or "Occupied".

* In the simplest toy model, we can assume the EV drives on if the charger is "Occupied" (i.e. assume no queuing)

We will develop the simple toy model mentioned above, then extend it as follows:

* More realistic model: assume some EVs wait if charger is "Occupied" (i.e. assume queuing)

* Still more realistic model: allow multiple chargers at a site, and assume site-level queuing.


### Theory

In scenarios where we have a system with a 'dead time' after each event during which no other events can be recorded, we can apply a statistical correction to estimate the true rate of events. This is analogous to the correction of count rate for a Geiger counter.

A Geiger counter measures ionizing radiation by detecting and counting particles. Each particle detected is called an 'event'. After each event, there's a brief period where the detector is unable to record another event, known as the 'dead time'. The observed count rate (O) is less than the true count rate (R) due to this dead time. The relationship between the true rate, observed rate, and dead time is described by the formula:

```
R = O / (1 - O*d)
```

Where:

- `R` is the true count rate (events per unit time)

- `O` is the observed count rate (events per unit time)

- `d` is the dead time of the detector.

This is derived in the section titled "CORRECTION OF COUNT RATE TO INCLUDE DEAD TIME:" of [Caltech Physics 77, Experiment 2, September 1994](http://www.cco.caltech.edu/~derose/labs/exp2.html):

> If n' counts are recorded in a time interval t with a detector of dead time d, it is necessary to compute the true number n that would have been observed with a counter of zero dead time. Since n'd is the total dead time, and n/t is the true counting rate, (n/t)n′d is the total number of counts that would have occurred during the total dead time interval. Therefore (n/t)n′d = n - n'. In terms of the counting rate, R = n/t and O = n'/t, we have R = (O / (1 - O*d))


### Application to EV Charger Usage

We can adapt this principle to estimate the demand for EV charging stations. Here, the 'event' is an EV plugging in for a charge, and the 'dead time' is the duration the charger is occupied by the EV. Unlike a Geiger counter, an EV charging station has variable 'dead times' as the charging duration can vary, and the arrival rates may vary, e.g. with the hour of the day. As a starting point we nevertheless propose to apply the simple model, using a fixed or average charging time as the dead time.

### Example

Consider an EV charging station with the following observed data at a particular time of day:

- Charging time (dead time, `d`): 1 hour
- Observed rate of plugging in during peak hours (O): 0.25 plug-ins per hour

Using the Geiger counter formula, we can estimate the true rate (R):

```
R = 0.25 / (1 - 0.25*1) = 0.25 / 0.75 = 1/3
```
plug-ins per hour. This suggests that the true underlying demand is 1/3 plugs per hour. To determine the number of EVs missing out on a charge, we subtract the observed rate from the true rate. Let L denote the rate of lost charges (EVs that leave without a charge per hour). Then:
```
L = R - O = 1/3 - 1/4 = 1/12 EVs per hour
```

On average, at this time of day, every 12 days an EV misses out on a charge during this hour of the day.

As mentioned above, this model assumes a constant average charging time and does not account for variations in arrival rates or charging times, which can vary throughout the day. Nevertheless, it provides a simple starting point for estimating the extent of un-met demand for EV charging stations.

### Prototype implementation in Julia

The script `single-charger-collision-model.jl` provides a prototype implementation of the model. The script generates a synthetic dataset of EV charging events and applies the correction formula to estimate the true demand and the number of missed charges. The script also includes a simple visualization of the results. See the [`Getting Started With The Julia Script`](README-getting-started.md) README for instructions on how to run the script.

### Refinements for Variable Charging Durations and Queuing Behavior

We can accomodate variable charging durations straightforwardly. Instead of using a single average charging time, we can simply use the charging station state timeseries to build up a probability distribution of charging station state by time of day. This can be used to correct the observed plug-in rate. In the simple non-queuing model, the correction formula becomes
```
R = O / (1 - f)
```
where `f` is the historical probability of the charger being unavailable at that time of the day.

To further refine our model for estimating demand at EV charging stations, we need to consider that some drivers may queue for the charger rather than drive to another location. This behavior will impact the observed rate of charger usage and the actual experience of the drivers.

#### Incorporating Queuing Into the Model

In the non-queuing model, there were two arrivals types, namely plugging in, \(rate `O` \) and vehicles that leave \(rate `L` \). In the queuing model, There are three arrival types:

* EVs who arrive and plug in \(`N`\)
* EVs who arrive and queue \(`Q`\)
* EVs who arrive and leave \(`L`\)

Two of these can be observed. Here we have introduced a new variable, `Q`, to represent the rate of vehicles (per hour) plugging in from a queued state. To identify when a vehicle had been queuing, we detect quick turnaround events, defined as the charger being unplugged and then plugged back in within a short timeframe, such as a couple of minutes. These events are not counted as new arrivals but rather as the commencement of charging for a queued vehicle. We see the vehicles who arrive and plug in and the ones who arrived and queued. We don't see the ones who arrive and leave. Schematically, our model with queuing is as follows:
```

            => available   => plug in      N
 
 R arrive
                           => queue        Q
            => unavailable
                           => leave        L
```

We observe O = N + Q arrivals per hour. By subtracting `Q` from `O`, we can account for queued vehicles and obtain the arrival rate at the charger. The rate of new arrivals plugging in, \( `N` \), is the difference between the observed rate of plug-ins, `O`,and the rate of EVs plugging in from a queued state, \( `Q` \):
```
N = O - Q
```
Our original correction formula applies to `N`, and we can estimate the true rate of arrivals as follows:
```
R = N/(1 - f)
```
where `f` is the fraction of time the charger is occupied, at the time of day under consideration. We know f by examining charger/site state information. Assuming that the rate of EVs entering the queue is the same as the rate of EVs leaving the queue, we can estimate `L`, the rate of missed charges, from `R`, `N` and `Q`:
```
L = R - N - Q
```

To summarize:
* We have direct measurements of `Q`, the vehicles who have been queuing.
* We have direct measurement of `O`, the total observed plug-in rate.
* We can estimate `f`, the fraction of time the charger is unavailable, from historic charger state timeseries.
* From `O` and `Q`, we estimate `N`, the observed rate of new arrivals.
* From `N` and `f`, we estimate `R`, the true rate of new arrivals.
* From `R`, `N` and `Q`, we estimate `L`, the rate of EVs leaving without a charge.

The fraction of arrivals that miss out on a charge is then approximately `L/R` and the fraction that have to queue is approximately `Q/R`.

### Refinements for Multiple Chargers at a Site

EVRoam schema distinguishes between three availability states at the charger level: "Available", "Unavailable", and "Occupied". It does not generally reveal which head was inserted (CHAdeMO vs Type 2 CCS). 

We model EVRoam "Sites" as non-interacting groups of interchangeable chargers. We can treat the site as a single charger, and apply the single-charger model to the site. The site's state is determined by the states of its chargers.

Multi-charger-site rules:
* if any charger is "Available", the site is "Available".
* if all chargers are "Unavailable", the site is "Unavailable".
* otherwise, the site is "Occupied".

Queueing detection: A plug-in event soon after a site became available is assumed to have come from a queued EV. As before, combining non-queuing arrival density and site availability percentage, we can infer the "hidden demand" that is obscured due to the site being unavailable.