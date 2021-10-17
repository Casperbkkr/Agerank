import numpy as np
import pandas as pd
import math
import random as rd
import sys

# todo remove perfect vaccination
# todo add vaccination strategies
# todo add macro trackers and R tracker


from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.layouts import row

# Main input parameters of the simulation, you may want to vary these.
N = 100000  # number of persons in the network,
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


def read_age_distribution(age_dist_file, csv_or_txt="txt"):
    """This function reads the population data, each line containing
    the number of persons of a certain age. The lowest age is 0,
    the highest age contains everyone of that age and higher.
    These lines are followed by a line that contains the total number of persons,
    which is used as a check for completeness of the file.

    The function returns a list with the population for each age.
    """

    # Read in population per year of the Netherlands on January 1, 2020
    # Source: CBS, Statistics Netherlands, https://opendata.cbs.nl
    if csv_or_txt == "txt":
        # print('\n**** Population of the Netherlands on Jan. 1, 2020. Source: CBS\n')
        population = pd.read_csv(age_dist_file, sep=" ", header=None)
        population.columns = ["Number of people"]

    if csv_or_txt == "csv":
        # print('\n**** Population of the Netherlands on Jan. 1, 2020. Source: CBS\n')
        population = pd.read_csv(age_dist_file)
        population.columns = ["Number of people"]

    population.index.name = "Age"
    return population


def read_fatality_distribution(fatality_distribution_file, number_of_ages):
    """This function reads the age-specific infection fatality rates
    (IFR) for covid-19 infections, as given in Table 3 of the article
    'Assessing the age specificity of infection fatality rates
    for COVID‐19: systematic review, meta‐analysis,
    and public policy implications, by A. T. Levin et al.,
    European Journal of Epidemiology (2020),
    https://doi.org/10.1007/s10654-020-00698-1

    The function returns a dictionary with the IFR for each age as its value and age as key.
    """

    # Read in fatality per age group

    # print(' Source: Levin et al. (2020)\n')
    start_of_age_group = []
    rate = []
    with open(fatality_distribution_file, 'r') as infile:
        for line in infile:
            data = line.split()
            start_of_age_group.append(int(data[0]))
            rate.append(float(data[1]))

    # determine fatality for each age k
    fatality = []
    for g in range(len(start_of_age_group) - 1):
        for k in range(start_of_age_group[g], start_of_age_group[g + 1]):
            fatality.append(rate[g])
    g = len(start_of_age_group) - 1
    for k in range(start_of_age_group[g], number_of_ages):  # last rate
        fatality.append(rate[g])

    return fatality


def make_age_groups(dataframe, inp, number_of_ages):
    out = []
    number_of_age_groups = len(inp)
    for i in range(0, number_of_age_groups - 1):
        Group = age_group_class(i, inp[i], inp[i + 1])
        for j in range(inp[i + 1] - inp[i]):
            out.append(Group)

    final_group = age_group_class(number_of_age_groups - 1, inp[-1], number_of_ages)
    for k in range(dataframe.index[-1] - inp[-1] + 1):
        out.append(final_group)

    return out


def read_contact_data(dataframe, file1, file2):
    """This function reads the participants and contacts per age group
    from the POLYMOD project.

    Source: J. Mossong et al., "Social Contacts and Mixing Patterns Relevant
    to the Spread of Infectious Diseases", PLOS Medicine (2008),
    https://doi.org/10.1371/journal.pmed.0050074

    Choices made for inclusion:
    - only participants in NL, who keep a diary
    - both physical and nonphysical contact of > 15 minutes duration
    - only if all data of a particpant were complete

    Input: ngroups = number of age groups
           group = array of group indices for the ages
    """

    nages = len(dataframe.index)  # number of ages
    ngroups = dataframe["Age group class object"].nunique()

    groepen = dataframe["Age group class object"].tolist()
    group = list(dict.fromkeys(groepen))

    # print('\n**** Data from the POLYMOD contact study from 2008.')
    # print('Source: J. Mossong et al.\n')

    # determine ages of participants
    participants = np.zeros(nages, dtype=int)
    with open(file1, 'r') as infile:
        for line in infile:
            data = line.split()
            k = int(data[3])  # age of participant
            if k < nages:
                participants[k] += 1
    # print('Number of participants for each age:\n', participants)

    # todo count the participants by age group
    participants_group = np.zeros((ngroups), dtype=int)
    for r in range(nages):
        participants_group[groepen[r].id] += participants[k]
    # print('Number of participants for each age group:', participants_group)
    # count the participants by age group

    # create contact matrix C based on a period of 30 days
    contacts = np.zeros((nages, nages), dtype=int)
    frequency = np.zeros((6), dtype=int)  # todo wat doet dit?
    with open(file2, 'r') as infile:
        for line in infile:
            data = line.split()
            age_i = int(data[0])  # age of participant
            age_j = int(data[1])  # estimated age of contact
            freq = int(data[2])  # 1=daily, 2=weekly, 3= monthly,
            # 4=few times per year, 5=first time
            if age_i < nages and age_j < nages:
                if freq == 1:
                    contacts[age_i][age_j] += 20  # daily contact, not in the weekend
                elif freq == 2:
                    contacts[age_i][age_j] += 4  # weekly contact
                elif freq == 3:
                    contacts[age_i][age_j] += 1  # monthly contact

    # count the contacts by age group and symmetrise the matrix by computing B = C + C^T.
    # assumption: each contact is assumed to be registered only once (by the diary keeper).

    b = np.zeros((ngroups, ngroups), dtype=int)

    for age_i in range(nages):
        for age_j in range(nages):
            gi = groepen[age_i].age_group_id
            gj = groepen[age_j].age_group_id
            b[gi][gj] += contacts[age_i][age_j] + contacts[age_j][age_i]

    # the total number of contacts of a person in age group gi with
    # a person in age group gj in a month is b[gi][gj]
    # print('Number of contacts for each age group:\n', b)

    # the average number of contacts of a person in age group gi with
    # a person in age group gj in a month is b[gi][gj]/participants_group[gi]
    # print('Average number of different contacts in a month per person for each age group:\n')
    degree = np.zeros((ngroups, ngroups), dtype=float)
    for gi in range(ngroups):
        for gj in range(ngroups):
            # the degree takes into account that some contacts
            # are with the same person
            degree[gi][gj] = b[gi][gj] / (PERIOD * participants_group[gi])
    # print(degree)

    return degree


def determine_age_distribution(N, dataframe):
    '''
    determine age distribution for persons in network to be created
    :param N: Number of people to be modelled in teh population
    :param population: population[k] = number of persons of age k
    :return:
    This function returns a list with the amount of people of a certain age in the network.
    '''

    start_age = np.zeros((len(dataframe.index)), dtype=int)
    total = sum(dataframe["Number of people"])
    partial_sum = 0  # partial sum
    for k in range(len(dataframe.index)):
        fraction = partial_sum / total
        start_age[k] = math.floor(fraction * N)
        partial_sum += dataframe["Number of people"][k]  # psum = number of persons aged <= k
    return start_age


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


def create_people(N, dataframe, vaccination_readiness):
    people = []
    for age in dataframe.index[:-1]:  # add people of all but highest age
        for i in range(dataframe['Start of age group'][age],
                       dataframe['Start of age group'][age + 1]):  # add the fraction belonging to that age
            rand = rd.randrange(0, 1)
            if rand < vaccination_readiness:
                people.append(person(i, age, False))
            else:
                people.append(person(i, age, True))  # create person and add to list of people
            dataframe["Age group class object"][age].add_member(people[i])
    # create people of the highest age
    age = dataframe.index[-1]

    for i in range(dataframe['Start of age group'].iloc[-1], N):
        rand = rd.randrange(0, 1)
        if rand < vaccination_readiness:
            people.append(person(i, age, False))
        else:
            people.append(person(i, age, True))  # create person and add to list of people
        dataframe["Age group class object"].iloc[-1].add_member(people[i])
    return people


def create_subnetwork(group1, group2, degree, i0, j0):
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
        # todo kan dit sneller?
        for person1 in group1.members:
            for person2 in group2.members:
                if not person1.person_id == person2.person_id:  # no edges to self
                    out.append((i0 + person1.person_id, j0 + person2.person_id))
                    out.append((j0 + person2.person_id, i0 + person1.person_id))
        return out
    else:
        # determine the number of trials needed to create
        # the desired number of edges, using some basic probability theory
        p = 1 / (m * n)  # probability of a matrix element a(i,j)
        # becoming nonzero in one trial
        if isdiagonal:
            trials = math.floor(math.log(1 - degree / (n - 1)) / math.log(1 - p))
        else:
            trials = math.floor(math.log(1 - degree / n) / math.log(1 - p))

        out = []
        for k in range(trials):
            r1 = rd.randint(0, m - 1)
            i = group2.members[r1].person_id
            r2 = rd.randint(0, n - 1)
            j = group1.members[r2].person_id
            if not i == j:
                out.append((i, j))
                out.append((j, i))

        return out


def create_network(dataframe, people, contact_data):
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
    ngroups = dataframe['Age group class object'].nunique()
    Groepen = dataframe['Age group class object'].unique()

    out = []
    i0 = 0
    teller = 1
    for gi in range(ngroups):
        j0 = 0
        for gj in range(ngroups):
            # size is the number of persons of an age group
            # d[gi][gj] is the degree of a block, which is a submatrix
            # containing all contacts between age groups gi and gj.
            out += create_subnetwork(Groepen[gi], Groepen[gj], contact_data[gi][gj], i0, j0)
            sys.stdout.write('\r' + "Blok: " + str(teller))
            sys.stdout.flush()
            teller += 1
            j0 += Groepen[gj].size_group()
        i0 += Groepen[gi].size_group()

    # remove duplicates from the list
    # todo Is sorteren van deze lijst strikt nodig?
    a = list(dict.fromkeys(out))

    return a

    # a.sort()
    # print("lijst gesorteerd")

    # return a


def initialise_infection(people, P0, tracker_changes):
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


def initialise_vaccination(N, people, BETA0, VACC0, tracker_changes):
    # Initializes a fraction of to population with a vaccination

    # vacinate a fraction VACC0 of the population
    max_young1 = math.floor(BETA0 * VACC0 * N)
    max_young = min(max_young1, N - 1)  # just a precaution
    min_old1 = math.floor(N - (1 - BETA0) * VACC0 * N)
    min_old = max(min_old1, 0)

    new_vaccinations = 0
    # todo find out why this starts at age 0 with vaccinating
    for i in range(max_young + 1):  # todo is this oversimplified?
        if people[i].status == SUSCEPTIBLE:
            tracker_changes["susceptible"] += -1
        if people[i].status == INFECTIOUS:
            tracker_changes["currently infected"] += -1
            tracker_changes['total infected'] += -1
        people[i].update_status(VACCINATED)
        tracker_changes["vaccinated"] += 1

    # todo hier gaat iets mis
    for i in range(min_old, N - 1):
        if people[i].status == SUSCEPTIBLE:
            tracker_changes["susceptible"] += -1
        if people[i].status == INFECTIOUS:
            tracker_changes["currently infected"] += -1
            tracker_changes['total infected'] += -1
        people[i].update_status(VACCINATED)
        tracker_changes["vaccinated"] += 1

    return tracker_changes


def initialize_model(N, files, parameters, tracker_changes):
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
    print("Determening age distribution for population of " + str(N) + "people.")
    data["Start of age group"] = determine_age_distribution(N, data)

    # Read the file containing data about contacts longer then 15 minutes
    print("Creating age group contact distribution.")
    contact_data = read_contact_data(data, 'participants_polymod_NL.txt', 'contacts_polymod_NL.txt')

    # Create people
    print("Creating people.")
    people = create_people(N, data, 0.95)

    # Create contact network
    print("Generating network.")
    contact_matrix = create_network(data, people, contact_data)

    # Initialize infection
    tracker_changes = initialise_infection(people, P0, tracker_changes)

    # Initialize vaccination if necessary
    tracker_changes = initialise_vaccination(N, people, BETA0, VACC0, tracker_changes)

    return data, people, contact_matrix, tracker_changes


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
                p = P_MEETING  # probability of a meeting with 1 infected contact
            else:
                p = 1 - (1 - P_MEETING) ** y[i]  # probability with more than 1
            if r < p:
                if person.status == SUSCEPTIBLE:
                    person.update_status(INFECTIOUS)
                    tracker_changes["currently infected"] += 1
                    tracker_changes["total infected"] += 1
                    tracker_changes["susceptible"] += -1
                elif person.status == VACCINATED:
                    if rd.random() < P_TRANSMIT1:
                        person.update_status(TRANSMITTER)
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


def run_model(data, people, contact_matrix, tracker, timesteps=100, start_vaccination=0):
    # Function for running the model. It wraps the vaccinate, infect and update functions
    for time in range(timesteps):
        sys.stdout.write('\r' + "Tijdstap: " + str(time))
        sys.stdout.flush()
        status_changes_0 = tracker.empty_changes()
        if time < start_vaccination:
            status_changes_1 = infect(contact_matrix, people, status_changes_0)
            status_changes_2 = update(data['IFR'], people, status_changes_1)

            tracker.update_statistics(status_changes_2)
        else:
            status_changes_1 = infect(contact_matrix, people, status_changes_0)
            status_changes_2 = update(data['IFR'], people, status_changes_1)
            status_changes_3 = vaccinate(people, status_changes_2)

            tracker.update_statistics(status_changes_3)

    return tracker


timesteps = 450

tracker = track_statistics()

tracker_changes = tracker.init_empty_changes()
print("Initializing model")
data, people, contact_matrix, tracker_changes = initialize_model(N, 1, 2, tracker_changes)
tracker.update_statistics(tracker_changes)

print("Running model")
tracker = run_model(data, people, contact_matrix, tracker, timesteps - 1)
print("Finished")

source = ColumnDataSource(data={'time': [t for t in range(timesteps + 2)],
                                'infected': tracker.data['currently infected'],
                                'deceased': tracker.data['deceased'],
                                'recovered': tracker.data['recovered'],
                                'transmitter': tracker.data['transmitter'],
                                'symptomatic': tracker.data['symptomatic'],
                                'hospitalized': tracker.data['hospitalized'],
                                'vaccinated': tracker.data['vaccinated'],
                                'total_infected': tracker.data['total infected']
                                }
                          )
p1 = figure(
    title="Title",
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

p1.legend.orientation = "horizontal"

p2 = figure(
    title="recovered",
    x_axis_label='days',
    y_axis_label='people',
    tools='reset,save,pan,wheel_zoom,box_zoom,xzoom_in,xzoom_out')

p2.line(
    x='time',
    y='total_infected',
    legend_label='total infected',
    line_width=1,
    line_color="red",
    source=source)

p2.line(
    x='time',
    y='recovered',
    legend_label='recovered',
    line_width=1,
    line_color="green",
    source=source)

p2.line(
    x='time',
    y='vaccinated',
    legend_label='vaccinated',
    line_width=1,
    line_color="blue",
    source=source)

p2.line(
    x='time',
    y='deceased',
    legend_label='deceased',
    line_width=1,
    line_color="orange",
    source=source)

p2.add_tools(
    HoverTool(
        tooltips=[('time', '@time'),
                  ('recovered', '@recovered'),
                  ('total infected', '@total_infected'),
                  ('deceased', '@deceased')]))

p2.legend.orientation = "horizontal"
# show the results
show(row(p1, p2))
