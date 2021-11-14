import numpy as np
import pandas as pd
import math
import random as rd
import sys

# Main input parameters of the simulation, you may want to vary these.
N = 10000  # number of persons in the network,
# a trade-off between accuracy and speed
BETA = 0.0  # fraction of young among the daily vaccinated persons
NDAYS = 90  # number of days of the simulation

# For the following parameters choose sensible values, as realistic as possible.

# Probability parameters (0 <= P <= 1 must hold)
P0 = 0.003  # probability of infection at time 0
P_MEETING = 0.004  # probability of meeting a contact on a given day
# and becoming infected.
# A base value in a nonlockdown situation would be 0.02,
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
VACC = 0.007  # fraction of the population vaccinated per day.
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

# Possible status of infection
SUSCEPTIBLE = 0
INFECTIOUS = 1  # but no symptoms yet
SYMPTOMATIC = 2  # but not quarantined
QUARANTINED = 3
HOSPITALISED = 4
RECOVERED = 5
VACCINATED = 6
TRANSMITTER = 7  # infectious after being vaccinated or having recovered
# (tests positive but has no symptoms)
DECEASED = 8  # must be the highest numbered status

class track_statistics(object):
    # todo make dataframe
    def __init__(self, tracker_id=1):
        self.tracker_id = tracker_id
        self.data = pd.DataFrame(columns=["susceptible",
                                          "total infected",
                                          "currently infected",
                                          "symptomatic",
                                          "quarantined",
                                          "hospitalized",
                                          "recovered",
                                          "vaccinated",
                                          "transmitter",
                                          "deceased"])

    def update_statistics(self, tracker_changes):
        self.data.index.name = "timestep"
        self.data = self.data.append(tracker_changes, ignore_index=True)

    def init_empty_changes(self):
        dictionary = {"susceptible": 0,
                      "total infected": 0,
                      "currently infected": 0,
                      "symptomatic": 0,
                      "quarantined": 0,
                      "hospitalized": 0,
                      "recovered": 0,
                      "vaccinated": 0,
                      "transmitter": 0,
                      "deceased": 0}
        return dictionary

    def empty_changes(self):
        dictionary = {"susceptible": self.data["susceptible"].iloc[-1],
                      "total infected": self.data["total infected"].iloc[-1],
                      "currently infected": self.data["currently infected"].iloc[-1],
                      "symptomatic": self.data["symptomatic"].iloc[-1],
                      "quarantined": self.data["quarantined"].iloc[-1],
                      "hospitalized": self.data["hospitalized"].iloc[-1],
                      "recovered": self.data["recovered"].iloc[-1],
                      "vaccinated": self.data["vaccinated"].iloc[-1],
                      "transmitter": self.data["transmitter"].iloc[-1],
                      "deceased": self.data["deceased"].iloc[-1]}
        return dictionary

    def output(self, filename='model_output.csv'):
        self.data.to_csv(filename)

    def read_data(self, filename):
        return pd.read_csv(filename)


class person(object):
    def __init__(self, person_id, age, status=SUSCEPTIBLE, vaccination_readiness=True, days_since_infection=0):
        self.person_id = person_id  # corresponds to the index within the adjacency matrix
        self.age = age  # age of the person
        self.status = status  #
        self.days_since_infection = days_since_infection
        self.vaccination_readiness = vaccination_readiness
        self.household = 0

    def update_household(self, household):
        self.household = household

    def update_status(self, new_status):
        self.status = new_status

    def update_days_since_infection(self, new_days):
        self.days_since_infection = new_days

    def how_many_days(self):
        return self.days_since_infection


class group_class(object):
    def __init__(self, group_id):
        self.id = group_id
        self.members = []

    def add_member(self, person):
        self.members.append(person)

    def size_group(self):
        return len(self.members)


class household_class(group_class):
    def __init__(self, household_id, number_of_members):
        self.number_of_members = number_of_members
        super().__init__(household_id)


class age_group_class(group_class):
    def ages_in_group(self, age1, age2):
        return [i for i in range(age1, age2)]

    def __init__(self, age_group_id, from_age, to_age):
        self.ages = self.ages_in_group(from_age, to_age)
        self.age_group_id = age_group_id
        super().__init__(age_group_id)

