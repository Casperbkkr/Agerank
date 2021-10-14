import numpy as np
import random as rd
import math

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.layouts import row

import multiprocessing as mp

# todo remove perfect vaccination
# todo add vaccination strategies
# todo add macro trackers and R tracker
# todo add multi processing

from read_data import read_age_distribution
from read_data import read_contact_data
from read_data import read_fatality_distribution

# Main input parameters of the simulation, you may want to vary these.
N = 100000 # number of persons in the network,
# a trade-off between accuracy and speed
BETA = 0.0  # fraction of young among the daily vaccinated persons
NDAYS = 90  # number of days of the simulation

# For the following parameters choose sensible values, as realistic as possible.

# Probability parameters (0 <= P <= 1 must hold)
P0 = 0.003  # probability of infection at time 0
P_MEETING = 0.004  # probability of meeting a contact on a given day
# and becoming infected.
# A base value in a non-lockdown situation would be 0.02,
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


# todo add different vaccines and their effectiveness
class vaccine(object):
    def __init__(self, name, effectivenes):
        self.name = name
        self.effectivenes = effectivenes


# todo add different viri
class virus(object):
    def __init__(self, virus, lethality):
        self.virus = virus
        self.lethalithy = lethality


class track_statistics(object):
    def __init__(self, tracker_id):
        self.tracker_id = tracker_id
        self.susceptible = [0]
        self.infectious = [0]
        self.symptomatic = [0]
        self.quarantined = [0]
        self.hospitalised = [0]
        self.recovered = [0]
        self.vaccinated = [0]
        self.transmitter = [0]
        self.deceased = [0]

    def update_all(self, tracker_changes):
        self.susceptible.append(self.susceptible[-1] + tracker_changes["susceptible"])
        self.infectious.append(self.infectious[-1] + tracker_changes["infected"])
        self.symptomatic.append(self.symptomatic[-1] + tracker_changes["symptomatic"])
        self.quarantined.append(self.quarantined[-1] + tracker_changes["quarantined"])
        self.hospitalised.append(self.hospitalised[-1] + tracker_changes["hospitalized"])
        self.recovered.append(self.recovered[-1] + tracker_changes["recovered"])
        self.vaccinated.append(self.vaccinated[-1] + tracker_changes["vaccinated"])
        self.transmitter.append(self.transmitter[-1] + tracker_changes["transmitter"])
        self.deceased.append(self.deceased[-1] + tracker_changes["deceased"])


class person(object):
    def __init__(self, person_id, age, status=SUSCEPTIBLE, vaccination_readiness=True, days_since_infection=0):
        self.person_id = person_id  # corresponds to the index within the adjacency matrix
        self.age = age  # age of the person
        self.status = status  #
        self.days_since_infection = days_since_infection
        self.vaccination_readiness = vaccination_readiness

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


class age_group_class(group_class):
    def ages_in_group(self, age1, age2):
        return [i for i in range(age1, age2)]

    def __init__(self, age_group_id, from_age, to_age):
        self.ages = self.ages_in_group(from_age, to_age)
        self.age_group_id = age_group_id
        super().__init__(age_group_id)


def determine_age_distribution(N, population, ages):
    '''
    determine age distribution for persons in network to be created
    :param N: Number of people to be modelled in teh population
    :param population: population[k] = number of persons of age k
    :return:
    This function returns a list with the amount of people of a certain age in the network.
    '''

    start_age = np.zeros((len(ages)), dtype=int)
    total = sum(population)
    print('Total population of the country = ', total)
    partial_sum = 0  # partial sum
    for k in range(len(ages)):
        fraction = partial_sum / total
        start_age[k] = math.floor(fraction * N)
        partial_sum += population[k]  # psum = number of persons aged <= k

    return start_age


def age_dict(inp, number_of_ages, ages):
    out = {}
    number_of_age_groups = len(inp)
    for i in range(0, number_of_age_groups - 1):
        Group = age_group_class(i, inp[i], inp[i + 1])
        for j in range(inp[i + 1] - inp[i]):
            out[inp[i] + j] = Group

    final_group = age_group_class(number_of_age_groups - 1, inp[-1], number_of_ages)
    for k in range(ages[-1] - inp[-1] + 1):
        out[inp[-1] + k] = final_group

    return out


def remove_duplicates(dictionary):
    """
        Helper function for removing duplicate age group key and value pairs.
        input:
        Dictionary
        output:
        Dictionary without duplicate values
    """
    temp = []
    res = dict()
    for key, val in dictionary.items():
        if val not in temp:
            temp.append(val)
            res[key] = val
    return res


def create_people(groups, start_age, nages, N, vaccination_readiness):
    new_groups = groups
    people = []
    for age in nages[:-2]:  # add people of all but highest age
        for i in range(start_age[age], start_age[age + 1]):  # add the fraction belonging to that age
            rand = rd.randrange(0, 1)
            if rand < vaccination_readiness:
                people.append(person(i, age, False))
            else:
                people.append(person(i, age, True))  # create person and add to list of people
            new_groups[age].add_member(people[i])

    for i in range(start_age[-1], N + 1):  # highest age
        rand = rd.randrange(0, 1)
        if rand < vaccination_readiness:
            people.append(person(i, age, False))
        else:
            people.append(person(i, age, True))  # create person and add to list of people
        new_groups[nages[-1]].add_member(people[-1])

    return people, new_groups


def create_subnetwork(group1, group2, degree, i0, j0, output):
    n = group1.size_group()
    m = group2.size_group()
    # determine whether the block is a diagonal block,
    # to avoid creating contacts with yourself
    if n == m and i0 == j0:
        isdiagonal = True
    else:
        isdiagonal = False

    # remove some degenerate cases
    if m <= 0 or n <= 0 or (isdiagonal and n == 1):
        return []

    # handle other special cases
    if (isdiagonal and degree >= n - 1) or (not isdiagonal and degree >= n):
        # the matrix should be full
        out = []
        for person1 in group1.members:
            for person2 in group2.members:
                if not person1.person_id == person2.person_id:  # no edges to self
                    out.append((i0 + person1.person_id, j0 + person2.person_id))
                    out.append((j0 + person2.person_id, i0 + person1.person_id))
        output.put(out)
        return
    # determine the number of trials needed to create
    # the desired number of edges, using some basic probability theory
    p = 1 / (m * n)  # probability of a matrix element a(i,j)
    # becoming nonzero in one trial
    if isdiagonal:
        trials = math.floor(math.log(1 - degree / (n - 1)) / math.log(1 - p))
    else:
        trials = math.floor(math.log(1 - degree / n) / math.log(1 - p))
    print('number of trials for block', i0, j0, ' is ', trials)

    out = []
    for k in range(trials):
        r1 = rd.randint(0, m - 1)
        i = group2.members[r1].person_id
        r2 = rd.randint(0, n - 1)
        j = group1.members[r2].person_id
        if not i == j:
            out.append((i, j))
            out.append((j, i))

    output.put(out)
    return



def create_network(groups, degrees):
    """This function creates an n by n adjacency matrix A
    that defines a random network with n vertices,
    where vertex i has a number of contacts determined
    by the degree matrix d.

    ngroups = number of age groups represented in the network
    size[g] = size of age group g in the network, i.e. the number
              of persons in the age group. The sum of all sizes is n.
    d[gi][gj] = average degree of a vertex in age group gi, considering
                only connections to age group gj.

    This is a fast method, linear in the number of edges |E|,
    so it can be used to create large networks.
    a is stored as a list of pairs (i,j), sorted by rows, and within
    each row by column index.
    """
    Groepen = remove_duplicates(groups)
    ngroups = len(Groepen.keys())
    Groepen = list(Groepen.values())

    D = []
    i0 = 0
    for gi in range(ngroups):
        j0 = 0
        for gj in range(ngroups):
            # size is the number of persons of an age group
            # d[gi][gj] is the degree of a block, which is a submatrix
            # containing all contacts between age groups gi and gj.
            j0 += Groepen[gj].size_group()
            D.append([Groepen[gi], Groepen[gj], degrees[gi][gj], i0, j0])
        i0 += Groepen[gi].size_group()

    output = mp.Queue()

    processes = [mp.process(create_subnetwork, args=(gi, gj, dg, i0, j0, output)) for gi, gj, dg, i0, j0 in D]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    results = [output.get() for p in processes]
    # remove duplicates from the list
    out_set = set(out)
    a = list(out_set)
    a.sort()

    return a


def initialise_infection(people, P0, tracker_changes, n=N):
    """ This function creates an array of n vertices (persons)
    and randomly initialises a fraction P0 of the persons
    as infected, denoted by status[i]=INFECTIOUS.
    Otherwise, status[i]=SUSCEPTIBLE.
    """

    # infect a fraction P0 of the population
    for person in people:
        if rd.random() < P0:
            person.update_status(INFECTIOUS)
            tracker_changes["infected"] += 1
        else:
            person.update_status(SUSCEPTIBLE)
            tracker_changes["susceptible"] += 1

    return tracker_changes


def initialise_vaccination(n, people, BETA0, VACC0, tracker_changes):
    # vacinate a fraction VACC0 of the population
    max_young1 = math.floor(BETA0 * VACC0 * n)
    max_young = min(max_young1, n - 1)  # just a precaution
    min_old1 = math.floor(n - (1 - BETA0) * VACC0 * n)
    min_old = max(min_old1, 0)

    new_vaccinations = 0
    # todo find out why this starts at age 0 with vaccinating
    for i in range(max_young + 1):  # todo is this oversimplified?
        if people[i].status == SUSCEPTIBLE:
            tracker_changes["susceptible"] += -1
        if people[i].status == INFECTIOUS:
            tracker_changes["infected"] += -1
        people[i].update_status(VACCINATED)
        tracker_changes["vaccinated"] += 1

    for i in range(min_old, n):
        if people[i].status == SUSCEPTIBLE:
            tracker_changes["susceptible"] += -1
        if people[i].status == INFECTIOUS:
            tracker_changes["infected"] += -1
        people[i].update_status(VACCINATED)
        tracker_changes["vaccinated"] += 1

    return tracker_changes


def infect(network, people, tracker_changes):
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
        if person.status == INFECTIOUS or person.status == SYMPTOMATIC:
            x[person.person_id] = 1
        elif person.status == TRANSMITTER and rd.random() < P_TRANSMIT0:
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
                p = P_MEETING  # probability of a meeting with 1 infected contact
            else:
                p = 1 - (1 - P_MEETING) ** y[i]  # probability with more than 1
            if r < p:
                if person.status == SUSCEPTIBLE:
                    person.update_status(INFECTIOUS)
                    tracker_changes["infected"] += 1
                    tracker_changes["susceptible"] += -1
                elif person.status == VACCINATED:
                    if rd.random() < P_TRANSMIT1:
                        person.update_status(TRANSMITTER)
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

    for person in people:
        if not person.status == SUSCEPTIBLE and not person.status == VACCINATED:
            new_days = person.how_many_days() + 1
            person.update_days_since_infection(new_days)

        if person.status == INFECTIOUS and person.days_since_infection == DAY_SYMPTOMS:
            status_changes["infected"] += -1
            if rd.random() < P_QUARANTINE:
                # i gets symptoms and quarantines
                person.update_status(QUARANTINED)
                status_changes["quarantined"] += 1

            else:
                # i gets symptoms but does not quarantine
                person.update_status(SYMPTOMATIC)
                status_changes["symptomatic"] += 1

        if (person.status == QUARANTINED) and person.days_since_infection == DAY_RECOVERY:
            status_changes["quarantined"] += -1
            if rd.random() < RATIO_HF * fatality[person.age]:
                person.update_status(HOSPITALISED)
                status_changes["hospitalized"] += 1
            else:
                person.update_status(RECOVERED)
                status_changes["recovered"] += 1

        if person.status == SYMPTOMATIC and person.days_since_infection == DAY_RECOVERY:
            status_changes["symptomatic"] += -1
            if rd.random() < RATIO_HF * fatality[person.age]:
                person.update_status(HOSPITALISED)
                status_changes["hospitalized"] += 1
            else:
                person.update_status(RECOVERED)
                status_changes["recovered"] += 1

        if person.status == HOSPITALISED and person.days_since_infection == DAY_RELEASE:
            status_changes["hospitalized"] += -1
            if rd.random() < 1 / RATIO_HF:
                person.update_status(DECEASED)
                status_changes["deceased"] += 1
            else:
                person.update_status(RECOVERED)
                status_changes["recovered"] += 1

        if person.status == TRANSMITTER and person.days_since_infection == NDAYS_TRANSMIT:
            person.update_status(VACCINATED)
            status_changes["transmitter"] += -1
            status_changes["vaccinated"] += 1

    return status_changes


def vaccinate(people, status_changes, order=0):
    """This function performs one time step (day) of the vaccinations.
    status represents the health status of the persons.
    Only the susceptible or recovered (after a number of days) are vaccinated
    """
    n = len(people)

    vacc_young = math.floor(BETA * VACC * n)  # today's number of vaccines for the young
    vacc_old = math.floor((1 - BETA) * VACC * n)  # and for the old

    # vaccinate the young, starting from the youngest
    count = 0
    for person in people:
        if person.status == SUSCEPTIBLE:
            if count < vacc_young and person.age >= STARTAGE:
                person.update_status(VACCINATED)
                status_changes["susceptible"] += -1
                status_changes["vaccinated"] += 1
                count += 1

        if person.status == RECOVERED and person.days_since_infection >= DAY_RECOVERY + NDAYS_VACC:
            if count < vacc_young and person.age >= STARTAGE:
                person.update_status(VACCINATED)
                status_changes["recovered"] += -1
                status_changes["vaccinated"] += 1
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
                    status_changes["susceptible"] += -1
                    status_changes["vaccinated"] += 1
                    count += 1

        if person.status == RECOVERED and person.days_since_infection >= DAY_RECOVERY + NDAYS_VACC:
            if count < vacc_old and person.age >= STARTAGE:
                person.update_status(VACCINATED)
                status_changes["recovered"] += -1
                status_changes["vaccinated"] += 1
                count += 1

    return status_changes


def initialize_model(N, BETA, vaccination_readiness):
    # initializes network of N people
    # returns the network and initial status
    print('n = ', N, 'BETA = ', BETA)

    # read age distribution of persons in the country
    # population[k] = number of persons of age k
    population = read_age_distribution()
    ages = [age for age in range(len(population))]
    number_of_ages = len(ages)

    # print number of adults in the population
    adults = 0
    for k in range(18, len(ages)):
        adults += population[k]
    print('Adult population = ', adults)

    # create age distribution for network of size N
    age_distribution = determine_age_distribution(N, population, ages)

    # starting ages of the groups
    start_group = [0, 4, 12, 18, 25, 35, 45, 55, 65, 75]

    # create dictionary with age groups
    groepen1 = age_dict(start_group, number_of_ages, ages)

    # Create all people in te network and add to dictionary
    people, groepen2 = create_people(groepen1, age_distribution, ages, N, vaccination_readiness)

    # read in contact data and create sparse matrix with people in the network
    contact_data = read_contact_data(groepen2)
    fatality_distribution = read_fatality_distribution(number_of_ages)

    # create contact network
    contact_matrix = create_network(groepen2, contact_data)
    print("Contact matrix complete")

    tracker = track_statistics(1)
    status_changes_init_0 = {"susceptible": 0, "quarantined": 0, "symptomatic": 0, "hospitalized": 0, "recovered": 0,
                             "deceased": 0,
                             "vaccinated": 0, "infected": 0, "transmitter": 0}

    # initialize network with infection and vaccination.
    status_changes_init_1 = initialise_infection(people, P0, status_changes_init_0)
    status_changes_init_2 = initialise_vaccination(N, people, BETA0, VACC0, status_changes_init_1)

    tracker.update_all(status_changes_init_2)

    return tracker, contact_matrix, people, fatality_distribution


def run_model(timesteps, contact_matrix, people, fatality_distribution, tracker):
    for time in range(timesteps):
        print("timestep: ", time)
        status_changes_0 = {"susceptible": 0, "quarantined": 0, "symptomatic": 0, "hospitalized": 0, "recovered": 0,
                            "deceased": 0,
                            "vaccinated": 0, "infected": 0, "transmitter": 0}

        status_changes_1 = infect(contact_matrix, people, status_changes_0)
        status_changes_2 = update(fatality_distribution, people, status_changes_1)
        status_changes_3 = vaccinate(people, status_changes_2)

        tracker.update_all(status_changes_3)

    return tracker


#### Main Code #####


tracker_1, contact_matrix, people, fatality_distribution = initialize_model(N, 0.0, 0.95)
timesteps = 3*356
tracker_2 = run_model(timesteps, contact_matrix, people, fatality_distribution, tracker_1)

source = ColumnDataSource(data={'time': [t for t in range(timesteps)],
                                'infected': tracker_2.infectious,
                                'deceased': tracker_2.deceased,
                                'recovered': tracker_2.recovered,
                                'transmitter': tracker_2.transmitter,
                                'symptomatic': tracker_2.symptomatic,
                                'hospitalized': tracker_2.hospitalised,
                                'vaccinated': tracker_2.vaccinated
                                }
                          )
p1 = figure(
    title="Covid-19",
    x_axis_label='days',
    y_axis_label='people',
    tools='reset,save,pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out')

# add a line renderer with source
p1.line(
    x='time',
    y='infected',
    legend_label='infected',
    line_width=1,
    line_color="red",
    source=source)

p1.line(
    x='time',
    y='hospitalized',
    legend_label='hospitalized',
    line_width=1,
    line_color="purple",
    source=source)

p1.line(
    x='time',
    y='transmitter',
    legend_label='transmitter',
    line_width=1,
    line_color="orange",
    source=source)

p1.line(
    x='time',
    y='symptomatic',
    legend_label='symptomatic',
    line_width=1,
    line_color="green",
    source=source)

p1.add_tools(
    HoverTool(
        tooltips=[('time', '@time'),
                  ('infected', '@infected'),
                  ('transmitter', '@transmitter'),
                  ('symptomatic', '@symptomatic'),
                  ('hospitalized', '@hospitalized'),
                  ('vaccinated', '@vaccinated')]))

p1.legend.orientation="horizontal"

p2 = figure(
    title="recovered",
    x_axis_label='days',
    y_axis_label='people',
    tools='reset,save,pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out')

p2.line(
    x='time',
    y='recovered',
    legend_label='recovered',
    line_width=1,
    line_color="green",
    source=source)
'''
p2.line(
    x='time',
    y='vaccinated',
    legend="W",
    line_width=1,
    line_color="blue",
    source=source)
'''
p2.line(
    x='time',
    y='deceased',
    legend_label='deceased',
    line_width=1,
    line_color="blue",
    source=source)

p2.add_tools(
    HoverTool(
        tooltips=[('time', '@time'),
                  ('recovered', '@recovered'),
                  ('deceased', '@deceased')]))
p2.legend.orientation="horizontal"
# show the results
show(row(p1, p2))
