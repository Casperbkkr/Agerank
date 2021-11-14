import numpy as np
import pandas as pd
import math
import random as rd
import sys

from Network import *
from Read_data import *
from Classes import *


def initialise_infection(parameters, people, tracker_changes):
    """ This function creates an array of n vertices (persons)
    and randomly initialises a fraction P0 of the persons
    as infected, denoted by status[i]=INFECTIOUS.
    Otherwise, status[i]=SUSCEPTIBLE.
    """

    # infect a fraction P0 of the population
    for person in people:
        if rd.random() < P0:
            person.update_status(INFECTIOUS)
            tracker_changes["currently infected"] += 1
            tracker_changes["total infected"] += 1
        else:
            person.update_status(SUSCEPTIBLE)
            tracker_changes["susceptible"] += 1

    return tracker_changes


def initialise_vaccination(parameters, people, tracker_changes):
    # Initializes a fraction of to population with a vaccination

    # vacinate a fraction VACC0 of the population
    max_young1 = math.floor(parameters["BETA0"] * parameters["VACC0"] * N)
    max_young = min(max_young1, parameters["N"] - 1)  # just a precaution
    min_old1 = math.floor(parameters["N"] - (1 - parameters["BETA0"]) * parameters["VACC0"] * parameters["N"])
    min_old = max(min_old1, 0)

    new_vaccinations = 0
    # todo find out why this starts at age 0 with vaccinating
    for i in range(max_young + 1):  # todo is this oversimplified?
        if people[i].status == parameters["SUSCEPTIBLE"]:
            tracker_changes["susceptible"] += -1
        if people[i].status == parameters["INFECTIOUS"]:
            tracker_changes["currently infected"] += -1
            tracker_changes['total infected'] += -1
        people[i].update_status(parameters["VACCINATED"])
        tracker_changes["vaccinated"] += 1

    # todo hier gaat iets mis
    for i in range(min_old, parameters["N"] - 1):
        if people[i].status == parameters["SUSCEPTIBLE"]:
            tracker_changes["susceptible"] += -1
        if people[i].status == parameters["INFECTIOUS"]:
            tracker_changes["currently infected"] += -1
            tracker_changes['total infected'] += -1
        people[i].update_status(parameters["VACCINATED"])
        tracker_changes["vaccinated"] += 1

    return tracker_changes


def initialize_model(parameters, files, tracker_changes):
    # Read age distribution and add to dataframe
    print("Reading age distribution from: " + 'CBS_NL_population_20200101.txt')
    data = read_age_distribution('CBS_NL_population_20200101.txt')

    # Add fatality rate to dataframe
    print("Reading infection fatality rates from: " + 'fatality_rates_Levin2020.txt')
    data["IFR"] = read_fatality_distribution('fatality_rates_Levin2020.txt', len(data.index))

    # Add corresponding age groups to dataframe
    # These age groups are all ages in the range(start_group[i],start_group[i+1])
    start_group = [0, 4, 12, 18, 25, 35, 45, 55, 65, 75]
    print("Creating age group class objects.")
    data["Age group class object"] = make_age_groups(data, start_group, len(data.index))

    # Determine how many people of each age there are
    print("Determening age distribution for population of " + str(parameters["N"]) + "people.")
    data["Start of age group"] = determine_age_distribution(parameters, data)

    # Read the file containing data about contacts longer then 15 minutes
    print("Creating age group contact distribution.")
    contact_data = read_contact_data(data, 'participants_polymod_NL.txt', 'contacts_polymod_NL.txt',
                                     parameters["PERIOD"])

    # Create people
    print("Creating people.")
    people, people_age_dict = create_people(parameters["N"], data, 0.85)


    print("Creating households")
    household_dict = make_households(parameters["N"], "a", "Huishoudens__samenstelling__regio_11102021_154809.csv",
                                     "Personen_in_huishoudens_naar_leeftijd_en_geslacht.csv", people_age_dict)

    # Create contact network
    print("Generating network.")
    contact_matrix = create_network(data, people, contact_data)

    # Initialize infection
    tracker_changes = initialise_infection(parameters, people, tracker_changes)

    # Initialize vaccination if necessary
    tracker_changes = initialise_vaccination(parameters, people, tracker_changes)

    return data, people, household_dict, contact_matrix, tracker_changes, people_age_dict


def infect_cohabitants(parameters, people, house_dict, tracker_changes):
    infected = []

    for j in people:
        if j.household != 0:
            if j.status == parameters["INFECTIOUS"] or j.status == parameters["SYMPTOMATIC"]:
                infected.append(j)
            elif j.status == parameters["TRANSMITTER"] and rd.random() < parameters["P_TRANSMIT0"]:
                infected.append(j)


    for j in infected:
        members = j.household.members
        cohabitants = [members[i] for i in range(len(members)) if members[i] != j]
        for cohab in cohabitants:
            if rd.random() < parameters["P_COHAB"]:
                if j.status == parameters["SUSCEPTIBLE"]:
                    j.update_status(parameters["INFECTIOUS"])
                    tracker_changes["currently infected"] += 1
                    tracker_changes["total infected"] += 1
                    tracker_changes["susceptible"] += -1
                elif j.status == parameters["VACCINATED"]:
                    if rd.random() < parameters["P_TRANSMIT1"]:
                        j.update_status(parameters["TRANSMITTER"])
                        # todo hier is ook nog een probleempje
                        tracker_changes['currently infected'] += 1
                        tracker_changes['total infected'] += 1
                        tracker_changes["transmitter"] += 1
                        tracker_changes["vaccinated"] += -1
                        person.update_days_since_infection(1)

    return tracker_changes


def infect_standard(parameters, network, people, tracker_changes):
    """This function performs one time step (day) of the infections
    a is the n by n adjacency matrix of the network
    status represents the health status of the persons.
    In this step, infectious persons infect their susceptible contacts
    with a certain probability.
    """
    n = len(people)
    x = np.zeros((n + 1), dtype=int)
    y = np.zeros((n + 1), dtype=int)

    # determine list of infectious persons from status
    for person in people:
        if person.status == parameters["INFECTIOUS"] or person.status == parameters["SYMPTOMATIC"]:
            x[person.person_id] = 1
        elif person.status == parameters["TRANSMITTER"] and rd.random() < parameters["P_TRANSMIT0"]:
            x[person.person_id] = 1

    total_infected = sum(x)
    prob = total_infected/(100*n)
    to_infect = [i for i in range(n) if rd.random() < prob]
    tre = 1
    for id in to_infect:
        person = people[id]
        if person.status == parameters["SUSCEPTIBLE"]:
            person.update_status(parameters["INFECTIOUS"])
            tracker_changes["currently infected"] += 1
            tracker_changes["total infected"] += 1
            tracker_changes["susceptible"] += -1
        elif person.status == parameters["VACCINATED"]:
            if rd.random() < parameters["P_TRANSMIT1"]:
                person.update_status(parameters["TRANSMITTER"])
                # todo hier is ook nog een probleempje
                tracker_changes['currently infected'] += 1
                tracker_changes['total infected'] += 1
                tracker_changes["transmitter"] += 1
                tracker_changes["vaccinated"] += -1
                person.update_days_since_infection(1)

    # propagate the infections
    # todo faster with dictionary
    for edge in network:
        i, j = edge
        y[i] += x[j]

    for person in people:
        # incorporate the daily probability of meeting a contact
        # taking into account the possibility of being infected twice

        if y[person.person_id] > 0:
            r = rd.random()
            if y[person.person_id] == 1:
                p = parameters["P_MEETING"]  # probability of a meeting with 1 infected contact
            else:
                p = 1 - (1 - parameters["P_MEETING"]) ** y[i]  # probability with more than 1
            if r < p:
                if person.status == parameters["SUSCEPTIBLE"]:
                    person.update_status(parameters["INFECTIOUS"])
                    tracker_changes["currently infected"] += 1
                    tracker_changes["total infected"] += 1
                    tracker_changes["susceptible"] += -1
                elif person.status == parameters["VACCINATED"]:
                    if rd.random() < parameters["P_TRANSMIT1"]:
                        person.update_status(parameters["TRANSMITTER"])
                        # todo hier is ook nog een probleempje
                        tracker_changes['currently infected'] += 1
                        tracker_changes['total infected'] += 1
                        tracker_changes["transmitter"] += 1
                        tracker_changes["vaccinated"] += -1
                        person.update_days_since_infection(1)

    return tracker_changes


def update(fatality, people, status_changes):
    """This function updates the status and increments the number
    of days that a person has been infected.
    For a new infection, days[i]=1.  For uninfected persons, days[i]=0.
    Input: infection fatality rate and age of persons i
    """
    new_status_changes = status_changes
    for person in people:
        if not person.status == SUSCEPTIBLE and not person.status == VACCINATED:
            new_days = person.how_many_days() + 1
            person.update_days_since_infection(new_days)

        if person.status == INFECTIOUS and person.days_since_infection == DAY_SYMPTOMS:
            if rd.random() < P_QUARANTINE:
                # i gets symptoms and quarantines
                person.update_status(QUARANTINED)
                new_status_changes["quarantined"] += 1

            else:
                # i gets symptoms but does not quarantine
                person.update_status(SYMPTOMATIC)
                new_status_changes["symptomatic"] += 1

        if (person.status == QUARANTINED) and person.days_since_infection == DAY_RECOVERY:
            new_status_changes["quarantined"] += -1
            if rd.random() < RATIO_HF * fatality[person.age]:
                person.update_status(HOSPITALISED)
                new_status_changes["hospitalized"] += 1
            else:
                person.update_status(RECOVERED)
                new_status_changes["recovered"] += 1
                new_status_changes["currently infected"] += -1

        if person.status == SYMPTOMATIC and person.days_since_infection == DAY_RECOVERY:
            new_status_changes["symptomatic"] += -1
            if rd.random() < RATIO_HF * fatality[person.age]:
                person.update_status(HOSPITALISED)
                new_status_changes["hospitalized"] += 1
            else:
                person.update_status(RECOVERED)
                new_status_changes["recovered"] += 1
                new_status_changes["currently infected"] += -1

        if person.status == HOSPITALISED and person.days_since_infection == DAY_RELEASE:
            new_status_changes["hospitalized"] += -1
            if rd.random() < 1 / RATIO_HF:
                person.update_status(DECEASED)
                new_status_changes["deceased"] += 1
                new_status_changes["currently infected"] += -1
            else:
                person.update_status(RECOVERED)
                new_status_changes["recovered"] += 1
                new_status_changes["currently infected"] += -1

        if person.status == TRANSMITTER and person.days_since_infection == NDAYS_TRANSMIT:
            person.update_status(VACCINATED)
            new_status_changes["transmitter"] += -1
            new_status_changes["currently infected"] += -1
            new_status_changes["vaccinated"] += 1

    return new_status_changes


def vaccinate(people, status_changes, order=0):
    """This function performs one time step (day) of the vaccinations.
    status represents the health status of the persons.
    Only the susceptible or recovered (after a number of days) are vaccinated
    """
    # todo make this exclusive and thereby faster
    new_status_changes = status_changes
    n = len(people)

    vacc_young = math.floor(BETA * VACC * n)  # today's number of vaccines for the young
    vacc_old = math.floor((1 - BETA) * VACC * n)  # and for the old

    # vaccinate the young, starting from the youngest
    count = 0
    for person in people:
        if person.status == SUSCEPTIBLE:
            if count < vacc_young and person.age >= STARTAGE:
                person.update_status(VACCINATED)
                new_status_changes["susceptible"] += -1
                new_status_changes["vaccinated"] += 1
                count += 1

        if person.status == RECOVERED and person.days_since_infection >= DAY_RECOVERY + NDAYS_VACC:
            if count < vacc_young and person.age >= STARTAGE:
                person.update_status(VACCINATED)
                new_status_changes["recovered"] += -1
                new_status_changes["vaccinated"] += 1
                count += 1

    # vaccinate the old, starting from the oldest
    count = 0
    reverse_people = people
    reverse_people.reverse()
    for person in reverse_people:
        if person.status == SUSCEPTIBLE:
            if person.vaccination_readiness == True:
                if count < vacc_old and person.age >= STARTAGE:
                    person.update_status(VACCINATED)
                    new_status_changes["susceptible"] += -1
                    new_status_changes["vaccinated"] += 1
                    count += 1

        if person.status == RECOVERED and person.days_since_infection >= DAY_RECOVERY + NDAYS_VACC:
            if count < vacc_old and person.age >= STARTAGE:
                person.update_status(VACCINATED)
                new_status_changes["recovered"] += -1
                new_status_changes["vaccinated"] += 1
                count += 1

    return new_status_changes


def run_model_household(parameters, data, people, households, contact_matrix, tracker, timesteps=100, start_vaccination=0):
    # Function for running the model. It wraps the vaccinate, infect and update functions
    for time in range(timesteps):
        sys.stdout.write('\r' + "Tijdstap: " + str(time))
        sys.stdout.flush()
        status_changes_0 = tracker.empty_changes()
        if time < start_vaccination:
            status_changes_1 = infect_cohabitants(parameters, people, households, status_changes_0)
            status_changes_2 = infect_standard(parameters, contact_matrix, people, status_changes_1)
            status_changes_3 = update(data['IFR'], people, status_changes_2)

            tracker.update_statistics(status_changes_3)
        else:
            status_changes_1 = infect_cohabitants(parameters, people, households, status_changes_0)
            status_changes_2 = infect_standard(parameters, contact_matrix, people, status_changes_1)
            status_changes_3 = update(data['IFR'], people, status_changes_2)
            status_changes_4 = vaccinate(people, status_changes_3)

            tracker.update_statistics(status_changes_4)

    return tracker


def run_model_standard(parameters, data, people, households, contact_matrix, tracker, timesteps=100, start_vaccination=0):
    # Function for running the model. It wraps the vaccinate, infect and update functions
    for time in range(timesteps):
        sys.stdout.write('\r' + "Tijdstap: " + str(time))
        sys.stdout.flush()
        status_changes_0 = tracker.empty_changes()
        if time < start_vaccination:
            #status_changes_1 = infect_cohabitants(parameters, people, households, status_changes_0)
            status_changes_2 = infect_standard(parameters, contact_matrix, people, status_changes_0)
            status_changes_3 = update(data['IFR'], people, status_changes_2)

            tracker.update_statistics(status_changes_3)
        else:
            #status_changes_1 = infect_cohabitants(parameters, people, households, status_changes_0)
            status_changes_2 = infect_standard(parameters, contact_matrix, people, status_changes_0)
            status_changes_3 = update(data['IFR'], people, status_changes_2)
            status_changes_4 = vaccinate(people, status_changes_3)

            tracker.update_statistics(status_changes_4)

    return tracker