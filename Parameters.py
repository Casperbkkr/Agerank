# Main input parameters of the simulation, you may want to vary these.
N = 100000  # number of persons in the network,
# a trade-off between accuracy and speed


# For the following parameters choose sensible values, as realistic as possible.
ENCOUNTERS = 10

# the probability of getting infected when encountering someone with whom the person has no real interaction.
P_ENCOUNTER = 0.012
# https://doi.org/10.1093/cid/ciab100
#
# The partition of agegroup. STARTGROUP[i] being the first age in the group and STARTGROUP[i+1]-1 being the last age
# in the group.
STARTGROUP = [0, 4, 12, 18, 25, 35, 45, 55, 65, 75]
# Willingness to take the vaccine if offered. Based on current dutch vaccination rate(19-11-2021)
# source: Coronadashboard
Vacc_readiness = 0.847
# Probability parameters (0 <= P <= 1 must hold)
P0 = 0.003  # probability of infection at time 0
P_MEETING = 0.004  # probability of meeting a contact on a given day
# and becoming infected.
# A base value in a non lockdown situation would be 0.02,
# assuming that 10% of daily meetings results in an infection
# and that the same person is met on average 6 times per month.
# For the lockdown situation, this number has been reduced.
# Here, an 80%-effective lockdown multiplies
# P_MEETING by a further factor 0.2.
P_QUARANTINE = 0.9  # probability of a person with symptoms going into
# quarantine where the alternative is being symptomatic
# and still infecting others. This takes into account
# that some persons will quarantine only partially.
P_TRANSMIT0 = 0.2  # probability of becoming a transmitter of the disease
# (with an asymptomatic infection) after having been vaccinated,
# when meeting an infected person, see the CDC brief
# https://www.cdc.gov/coronavirus/2019-ncov/science/science-briefs/fully-vaccinated-people.html

# The probability of infecting someone that lives in the same household.
P_COHAB = 0.211
# https://doi.org/10.1093/cid/ciab100


P_TRANSMIT1 = 0.25  # probability of getting infected by a transmitter
# when meeting him/her, see Levine-Tiefenbrun  et al.,
# Decreased SARS-CoV-2 viral load following vaccination.
# https://www.medrxiv.org/content/10.1101/2021.02.06.21251283v1.full.pdf

# Time parameters based on characteristics of the disease.
# This is a simplification, since the actual development of the disease
# shows a spread around these values.
# It must hold that: DAY_SYMPTOMS < DAY_RECOVERY < DAY_RELEASE
NDAYS_VACC = 28  # number of days to wait after recovery before vaccinating
NDAYS_TRANSMIT = 5  # number of days a vaccinated person can transmit the disease
# assumed only short period, being an asymptomatic infection
DAY_SYMPTOMS = 6  # first day of showing symptoms, and decision day
# of going into quarantine
DAY_RECOVERY = 13  # day of possible recovery, or hospitalisation
DAY_RELEASE = 20  # day of release from hospital, or death

# Vaccination parameters
VACC0 = 0.155  # fraction of vaccination at time 0
# based on 2.2 million first doses in NL for an adult
# population of 14.1 million.
BETA0 = 0.0  # fraction of young among the vaccinated persons at time 0
# These might be young care workers and hospital staff
# For now, neglected. The others are assumed to be the oldest.
VACC = 0.005  # fraction of the population vaccinated per day.
# The vaccination is assumed to have immediate effect, modelling
# receiving the shot two weeks earlier.
# Only susceptible persons are vaccinated.
# The order is by increasing index (young to old)
# for the fraction BETA, and old to young for the fraction 1-BETA.
# The value is based on 100000 first doses per day.
STARTAGE = 18  # starting age of the vaccination, country dependent (NL 18, IL 16)
# Other parameters.
PERIOD = 6  # number of days for which the contacts are the same group.
# It must be between 1 (all monthly contacts are with different
# persons, # and 30 (all monthly contacts are with the same person).
# A period of 6 seems a good compromise.
RATIO_HF = 3  # ratio between number of admissions to the hospital
# and the number of fatalities (ratio must be >=1)
# this does not influence the simulation, as the age-dependence
# of hospitalisation has been modelled through the fatality rate.


def parameter():
    return {"N": N,
            "ENCOUNTERS": ENCOUNTERS,
            "P_ENCOUNTER": P_ENCOUNTER,
            "STARTGROUP": STARTGROUP,
            "Vacc_readiness": Vacc_readiness,
            "P0": P0,
            "P_MEETING": P_MEETING,
            "P_QUARANTINE": P_QUARANTINE,
            "P_TRANSMIT0": P_TRANSMIT0,
            "P_COHAB": P_COHAB,
            "P_TRANSMIT1": P_TRANSMIT1,
            "NDAYS_VACC": NDAYS_VACC,
            "NDAYS_TRANSMIT": NDAYS_TRANSMIT,
            "DAY_SYMPTOMS": DAY_SYMPTOMS,
            "DAY_RECOVERY": DAY_RECOVERY,
            "DAY_RELEASE": DAY_RELEASE,
            "VACC0": VACC0,
            "BETA0": BETA0,
            "VACC": VACC,
            "STARTAGE": STARTAGE,
            "PERIOD": PERIOD,
            "RATIO_HF": RATIO_HF,
            "SUSCEPTIBLE": 0,
            "INFECTIOUS": 1,  # but no symptoms yet
            "SYMPTOMATIC": 2,  # but not quarantined
            "QUARANTINED": 3,
            "HOSPITALISED": 4,
            "RECOVERED": 5,
            "VACCINATED": 6,
            "TRANSMITTER": 7,  # infectious after being vaccinated or having recovered
            "DECEASED": 8
            }
