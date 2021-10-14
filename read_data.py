import numpy as np

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


def read_age_distribution():
    """This function reads the population data, each line containing
    the number of persons of a certain age. The lowest age is 0,
    the highest age contains everyone of that age and higher.
    These lines are followed by a line that contains the total number of persons,
    which is used as a check for completeness of the file.

    The function returns a list with the population for each age.
    """

    # Read in population per year of the Netherlands on January 1, 2020
    # Source: CBS, Statistics Netherlands, https://opendata.cbs.nl

    print('\n**** Population of the Netherlands on Jan. 1, 2020. Source: CBS\n')
    population = []
    with open('CBS_NL_population_20200101.txt', 'r') as infile:
        for line in infile:
            data = line.split()
            population.append(int(data[0]))  # take the first number from the line
    print(population)

    low = 0
    print('lowest age = ', low, '  Number of persons: ', population[low])
    high = len(population) - 1
    print('highest age >= ', high, '  Number of persons: ', population[high])

    return population


def read_fatality_distribution(number_of_ages):
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

    print('\n**** Infection Fatality Rate for Covid-19 per age group.')
    print(' Source: Levin et al. (2020)\n')
    start_of_age_group = []
    rate = []
    with open('fatality_rates_Levin2020.txt', 'r') as infile:
        for line in infile:
            data = line.split()
            start_of_age_group.append(int(data[0]))
            rate.append(float(data[1]))
    print(start_of_age_group)
    print(rate)

    # determine fatality for each age k
    fatality = {}
    for g in range(len(start_of_age_group) - 1):
        for k in range(start_of_age_group[g], start_of_age_group[g + 1]):
            fatality[k] = rate[g]
    g = len(start_of_age_group) - 1
    for k in range(start_of_age_group[g], number_of_ages):  # last rate
        fatality[k] = rate[g]

    return fatality


def read_contact_data(groepen):
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

    nages = len(groepen)  # number of ages
    ngroups = len(remove_duplicates(groepen))

    group = []
    i = 0
    for B in list(remove_duplicates(groepen).values()):
        for j in range(len(B.ages)):
            group.append(i)
        i += 1

    print('\n**** Data from the POLYMOD contact study from 2008.')
    print('Source: J. Mossong et al.\n')

    # determine ages of participants
    participants = np.zeros(nages, dtype=int)
    with open('participants_polymod_NL.txt', 'r') as infile:
        for line in infile:
            data = line.split()
            k = int(data[3])  # age of participant
            if k < nages:
                participants[k] += 1
    print('Number of participants for each age:\n', participants)

    # count the participants by age group
    participants_group = np.zeros((ngroups), dtype=int)
    for k in range(nages):
        participants_group[group[k]] += participants[k]
    print('Number of participants for each age group:', participants_group)
    # count the participants by age group

    # create contact matrix C based on a period of 30 days
    contacts = np.zeros((nages, nages), dtype=int)
    frequency = np.zeros((6), dtype=int) # todo wat doet dit?
    with open('contacts_polymod_NL.txt', 'r') as infile:
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
    #G = list(groepen.values())
    #a = np.zeros(len(G))
    #for i in range(len(G)):
        #temp = G[i].age_group_id
        #§§a[i] = temp

    for age_i in range(nages):
        for age_j in range(nages):
            gi = list(groepen.values())[age_i].age_group_id
            gj = list(groepen.values())[age_j].age_group_id
            b[gi][gj] += contacts[age_i][age_j] + contacts[age_j][age_i]

    # the total number of contacts of a person in age group gi with
    # a person in age group gj in a month is b[gi][gj]
    print('Number of contacts for each age group:\n', b)

    # the average number of contacts of a person in age group gi with
    # a person in age group gj in a month is b[gi][gj]/participants_group[gi]
    print('Average number of different contacts in a month per person for each age group:\n')
    degree = np.zeros((ngroups, ngroups), dtype=float)
    for gi in range(ngroups):
        for gj in range(ngroups):
            # the degree takes into account that some contacts
            # are with the same person
            degree[gi][gj] = b[gi][gj] / (PERIOD * participants_group[gi])
    print(degree)

    return degree


def remove_duplicates(dictionary):
    temp = []
    res = dict()
    for key, val in dictionary.items():
        if val not in temp:
            temp.append(val)
            res[key] = val
    return res
