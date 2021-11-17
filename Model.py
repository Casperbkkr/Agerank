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
    # todo dit kan sneller door willekeurig N*P0 elements uit range(0,N) te kiezen zonder terugleggen
    for person in people:
        if rd.random() < parameters["P0"]:
            person.update_status(INFECTIOUS)
            tracker_changes["currently infected"] += 1
            tracker_changes["total infected"] += 1
        else:
            person.update_status(SUSCEPTIBLE)
            tracker_changes["susceptible"] += 1

    return tracker_changes


def initialise_vaccination(parameters, order, tracker_changes):
    # Initializes a fraction of to population with a vaccination
    new_status_changes = tracker_changes

    # vacinate a fraction VACC0 of the population
    min_old1 = math.floor((1 - parameters["BETA0"]) * parameters["VACC0"] * parameters["N"])
    min_old = max(min_old1, 0)

    for i in range(min(min_old, len(order))):
        person = order.pop(0)
        if person.status == SUSCEPTIBLE:
            if person.vaccination_readiness == True:
                person.update_status(VACCINATED)
                new_status_changes["susceptible"] += -1
                new_status_changes["vaccinated"] += 1

    return new_status_changes, order


def vaccination_o(people, age_groups, type, start_age):
    # This is a function that creates a list of people to vaccinate to follow in the model
    if type == 1:  # Old to yound
        people = people[age_groups.iloc[start_age]:]
        people.reverse()
        order = people
    if type == 2:  # Young to old
        order = people[age_groups.iloc[start_age]:]
    if type == 3:  # Danish way
        tail = people[age_groups.iloc[start_age]:age_groups.iloc[50]]
        start = people[age_groups.iloc[50]:]
        start.reverse()
        order = start + tail
    if type == 4:  # custom
        one = people[age_groups.iloc[40]:age_groups.iloc[60]]
        two = people[age_groups.iloc[60]:]
        two.reverse()
        three = people[age_groups.iloc[start_age]:age_groups.iloc[40]]
        order = one + two + three
    return order


def initialize_model(parameters, files, order_type, tracker_changes):
    # This initializes everything for the model to run

    # Read age distribution and add to dataframe
    print("Reading age distribution from: " + files['Population_dataset'])
    data = read_age_distribution(files["Population_dataset"])

    # Add fatality rate to dataframe
    print("Reading infection fatality rates from: " + files["Fatality_distribution_dataset"])
    data["IFR"] = read_fatality_distribution(files["Fatality_distribution_dataset"], len(data.index))

    # Add corresponding age groups to dataframe
    # These age groups are all ages in the range(start_group[i],start_group[i+1])
    start_group = [0, 4, 12, 18, 25, 35, 45, 55, 65, 75]  # todo put this in parameters
    print("Creating age group class objects.")
    data["Age group class object"] = make_age_groups(data, start_group, len(data.index))

    # Determine how many people of each age there are
    print("Determining age distribution for population of " + str(parameters["N"]) + "people.")
    data["Start of age group"] = determine_age_distribution(parameters, data)

    # Read the file containing data about contacts longer then 15 minutes
    print("Creating age group contact distribution.")
    contact_data = read_contact_data(data, files["Polymod_participants_dataset"], files["Polymod_contacts_dataset"],
                                     parameters["PERIOD"])

    # Create people
    print("Creating people.")
    people, people_age_dict = create_people(parameters["N"], data, parameters["Vacc_readiness"])

    # determine vaccination order
    print("Determining vaccination order")
    vaccination_order = vaccination_o(people, data["Start of age group"], order_type, parameters["STARTAGE"])

    # create households
    print("Creating households")
    household_dict = make_households(parameters["N"], "a", files["Household_makeup_dataset"],
                                     files["People_in_household_dataset"], people_age_dict)

    # Create contact network
    print("Generating network.")
    contact_matrix = create_network(data, people, contact_data)

    # Initialize infection
    tracker_changes = initialise_infection(parameters, people, tracker_changes)

    # Initialize vaccination
    tracker_changes, vaccination_order = initialise_vaccination(parameters, vaccination_order, tracker_changes)

    return data, people, household_dict, contact_matrix, vaccination_order, tracker_changes, people_age_dict


def infect_cohabitants(parameters, people, house_dict, tracker_changes):
    # Method of infection for people in the same house.
    # todo needs to made faster with a sparse matrix instead of looking up everyones household. Also for further work.

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
                        tracker_changes['currently infected'] += 1
                        tracker_changes['total infected'] += 1
                        tracker_changes["transmitter"] += 1
                        person.update_days_since_infection(1)

    return tracker_changes


def infect_perturbation(parameters, people, tracker_changes):
    # this infects a fraction of the poplulation proportional to the the amount of infections
    n = len(people)
    x = np.zeros((n + 1), dtype=int)

    for person in people:
        if person.status == parameters["INFECTIOUS"] or person.status == parameters["SYMPTOMATIC"]:
            x[person.person_id] = 1
        elif person.status == parameters["TRANSMITTER"] and rd.random() < parameters["P_TRANSMIT0"]:
            x[person.person_id] = 1

    total_infected = sum(x)
    prob = 1 - (1 - parameters["P_ENCOUNTER"] * (total_infected / (parameters["N"] - 1))) ** (parameters["ENCOUNTERS"])
    to_infect = [i for i in range(n) if rd.random() < prob]
    for id in to_infect:
        person = people[id]
        if person.status == parameters["SUSCEPTIBLE"]:
            person.update_status(parameters["INFECTIOUS"])
            tracker_changes["currently infected"] += 1
            tracker_changes["total infected"] += 1
            tracker_changes["susceptible"] += -1
        elif person.status == parameters["VACCINATED"]:
            person.update_status(parameters["TRANSMITTER"])
            tracker_changes['currently infected'] += 1
            tracker_changes['total infected'] += 1
            tracker_changes["transmitter"] += 1
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

    # propagate the infections
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
                        tracker_changes['currently infected'] += 1
                        tracker_changes['total infected'] += 1
                        tracker_changes["transmitter"] += 1
                        # tracker_changes["vaccinated"] += -1
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
            # new_status_changes["vaccinated"] += 1

    return new_status_changes


def vaccinate(people, status_changes, order):
    """This function performs one time step (day) of the vaccinations.
    status represents the health status of the persons.
    Only the susceptible or recovered (after a number of days) are vaccinated
    """

    new_status_changes = status_changes
    n = len(people)

    # today's number of vaccines
    vacc = math.floor((1 - BETA) * VACC * n)  # and for the old
    for i in range(min(vacc, len(order))):
        person = order.pop(0)
        if person.status == SUSCEPTIBLE:
            if person.vaccination_readiness == True:
                person.update_status(VACCINATED)
                new_status_changes["susceptible"] += -1
                new_status_changes["vaccinated"] += 1

        if person.status == RECOVERED and person.days_since_infection >= DAY_RECOVERY + NDAYS_VACC:
            person.update_status(VACCINATED)
            new_status_changes["vaccinated"] += 1

    return new_status_changes, order


def run_model_household(parameters, data, people, households, contact_matrix, order, tracker, timesteps,
                        start_vaccination=0):
    # Function for running the model. It wraps the vaccinate, infection and update functions
    for time in range(timesteps):
        sys.stdout.write('\r' + "Tijdstap: " + str(time))
        sys.stdout.flush()
        status_changes_0 = tracker.empty_changes()
        if time < start_vaccination:
            status_changes_1 = infect_cohabitants(parameters, people, households, status_changes_0)
            status_changes_2 = infect_standard(parameters, contact_matrix, people, status_changes_1)
            status_changes_3 = infect_perturbation(parameters, people, status_changes_2)
            status_changes_4 = update(data['IFR'], people, status_changes_3)

            tracker.update_statistics(status_changes_4)
        else:
            status_changes_1 = infect_cohabitants(parameters, people, households, status_changes_0)
            status_changes_2 = infect_standard(parameters, contact_matrix, people, status_changes_1)
            status_changes_3 = infect_perturbation(parameters, people, status_changes_2)
            status_changes_4 = update(data['IFR'], people, status_changes_3)
            status_changes_5, order = vaccinate(people, status_changes_4, order)

            tracker.update_statistics(status_changes_5)

    return tracker


def model(parameters, filenames, type, timesteps=300):
    # this initializes and runs the entire model for a certain number of timesteps.
    # It returns a pandas dataframe containing all data at time t
    tracker = track_statistics()

    tracker_changes = tracker.init_empty_changes()
    data, people, households, contact_matrix, order, tracker_changes, people_dict = initialize_model(parameters,
                                                                                                     filenames,
                                                                                                     type,
                                                                                                     tracker_changes)
    # update the dataframe
    tracker.update_statistics(tracker_changes)

    # Run the model
    tracker = run_model_household(parameters, data, people, households, contact_matrix, order, tracker, timesteps - 1,
                                  0)

    return tracker
