
### Metrics for Cost of Queuing

Two key metrics will help us understand the "cost" of queuing:

1. **Number of EVs Missing Out on a Charge:**
   This is `M`, as derived above.

2. **Total Wait Time for Queued Vehicles:**
   To estimate the average wait time for those who queue, we can average half of the observed charging time for the session proceeding the queued session:

```
Wait Time/hour = Q * (Average charging duration) / 2
```
where the `Average charging duration` is over sessions immediately preceeding queued sessions.

### Assumptions and Considerations

By integrating queuing behavior into our model, we aim to provide a more comprehensive understanding of the demand for EV charging stations, which can inform better management and planning strategies. However we note the following assumptions and considerations:

- External factors, such as nearby chargers or drivers' time constraints, are not considered.
- Queued vehicles are willing to wait for a charger, and the queue does not exceed the space available.
- The time it takes for a vehicle to start charging after the previous vehicle departs is included to distinguish between a new arrival and a vehicle beginning to charge from the queue.


### Integration with EV Charger Network Gap Analysis

We can use the above metrics to estimate the fraction of EVs missing out on a charge and the fraction of EVs waiting for a charge at each charger. This can be integrated with the EV Charger Network Gap Analysis to identify the most critical gaps in the network.


### National Benefit Calculation for Public EV Chargers

This section outlines a simple approach for estimating the national benefit provided by public electric vehicle (EV) chargers. The formula considers several key factors, including the lifetime cost savings of using an EV over an internal combustion engine (ICE) vehicle, the charging capacity of public chargers, the proportion of an EV's charging that is done publicly, and the average frequency of charging required by an EV.

### Key Variables and derivation

- **B**: Benefit per EV (lifetime cost saving of using an EV compared to an ICE vehicle).
- **Cs**: Public charging sessions supported per day by a public charger.
- **F**: Fraction of an EV's total charging that is done publicly.
- **Ce**: Average charging sessions required by an EV per day.

1. **EVs Supported by One Charger per Day (E)**:
   - Given that `F` fraction of an EV's charging is done publicly, and `Cs` is the total number of public charging sessions available per day, the number of EVs supported per day by one charger is `Cs / F`.
   - However, since each EV doesn't need to charge every day, we adjust this by dividing by the average charging sessions required by an EV per day (`Ce`). So the total fleet of EVs supported by one charger is `E = Cs / F / Ce`.

2. **Total National Benefit (N)**:
   - The national benefit is calculated by multiplying the benefit per EV (`B`) by the number of EVs supported (`E`).
   - Therefore, the formula for the total national benefit is
```
N = B * (Cs / F) * (1 / Ce)
```
We can use this formula to estimate the national benefit of public EV chargers. We can estimate the key variables in the formula as follows.

- **B**: according to the linked [report](https://www.oriongroup.co.nz/assets/Value-of-EVs-to-NZ-Concept-Consulting-August-2019.pdf) by Concept Consulting, the lifetime cost saving of using an EV compared to an ICE vehicle purchased in 2025 is estimated at about $15,000. This is the national benefit per EV.
- **Cs**: this is obtained directly from the EVRoam data for each charger.
- **F**: as a ballpark estimate we assume that 15% of charging occurs at public chargers.
- **Ce**: as a ballpark estimate, we use 0.4 on the basis that an average EV requires charging every two or three days.

With these estimates, we get a benefit of about $250,000 per observed charging session per day. For instance, if a charger supports 10 charging sessions per day on average, the benefit is about $2.5 million per day.

3. **Cost of missed charges (C)**:
   - The cost of missed charges is calculated by multiplying the number of EVs that we model as missing out on a charge per day (`M`) by the benefit per observed charging session per day:
```
C = B * (M / F) * (1 / Ce)
```

