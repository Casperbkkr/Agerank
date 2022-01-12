from Classes import age_group_class

import numpy as np
import pandas as pd
import math


def read_age_distribution(age_dist_file: str, csv_or_txt: str = "txt") -> pd.DataFrame:
    """This function reads the population data, each line containing
    the number of persons of a certain age. The lowest age is 0,
    the highest age contains everyone of that age and higher.
    These lines are followed by a line that contains the total number of persons,
    which is used as a check for completeness of the file.

    The function returns a list with the population for each age.
    :param age_dist_file: filename
    :param csv_or_txt: choose whether it is a .csv or .txt file
    :return: dataframe containing the age distribution
    """

    # Read in population per year of the Netherlands on January 1, 2020
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


def read_fatality_distribution(fatality_distribution_file: str, number_of_ages: int) -> list:
    """This function reads the age-specific infection fatality rates
    (IFR) for covid-19 infections, as given in Table 3 of the article
    'Assessing the age specificity of infection fatality rates
    for COVID‐19: systematic review, meta‐analysis,
    and public policy implications, by A. T. Levin et al.,
    European Journal of Epidemiology (2020),
    https://doi.org/10.1007/s10654-020-00698-1

    The function returns a dictionary with the IFR for each age as its value and age as key.
    :param fatality_distribution_file: filename
    :param number_of_ages: number of ages present in the population to be modelled
    :return: list with the Infection Fatality rate
    """
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


def make_age_groups(dataframe: pd.DataFrame, inp: list, number_of_ages: int) -> list:
    """
    Creates the age groups present in the network.
    :param dataframe: dataframe with data needed to create the age groups
    :param inp: the number of age groups in the population
    :param number_of_ages: number of ages in the population
    :return: lsit with age groups
    """
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


def read_contact_data(dataframe: pd.DataFrame, participants_file: str, contacts_file: str, PERIOD: int) \
                     -> np.ndarray:
    """This function reads the participants and contacts per age group
    from the POLYMOD project.

    Choices made for inclusion:
    - only participants in NL, who keep a diary
    - both physical and nonphysical contact of > 15 minutes duration
    - only if all data of a particpant were complete

           :param dataframe: dataframe containing data about the population
           :param participants_file: filename
           :param contacts_file: filename
           :param PERIOD: frequency of meeting
           :return: matrix with the contact data
    """

    nages = len(dataframe.index)  # number of ages
    ngroups = dataframe["Age group class object"].nunique()

    groepen = dataframe["Age group class object"].tolist()

    # print('\n**** Data from the POLYMOD contact study from 2008.')
    # print('Source: J. Mossong et al.\n')

    # determine ages of participants
    participants = np.zeros(nages, dtype=int)
    with open(participants_file, 'r') as infile:
        for line in infile:
            data = line.split()
            k = int(data[3])  # age of participant
            if k < nages:
                participants[k] += 1

    participants_group = np.zeros((ngroups), dtype=int)
    for r in range(nages):
        participants_group[groepen[r].id] += participants[k]
    # count the participants by age group

    # create contact matrix C based on a period of 30 days
    contacts = np.zeros((nages, nages), dtype=int)
    with open(contacts_file, 'r') as infile:
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


def determine_age_distribution(parameters: dict, dataframe: pd.DataFrame) -> list:
    """
    determine age distribution for persons in network to be created
    This function returns a list with the amount of people of a certain age in the network.
    :param parameters: parameters needed to create the network
    :param dataframe: dataframe with all read in data so far
    :return: list with start of people of that age if L is the list then L[10]-L[9] is the number of people of age 10]
    """
    start_age = np.zeros((len(dataframe.index)), dtype=int)
    total = sum(dataframe["Number of people"])
    partial_sum = 0  # partial sum
    for k in range(len(dataframe.index)):
        fraction = partial_sum / total
        start_age[k] = math.floor(fraction * parameters["N"])
        partial_sum += dataframe["Number of people"][k]  # psum = number of persons aged <= k

    return start_age


def read_makeup_households(file: str) -> pd.DataFrame:
    """
    Reads data about the makeup of households
    :param file: filename
    :return: dataframe with data about the make up of households
    """
    makeup_data = pd.read_csv(file)
    makeup_data["One parent household"] = (
            makeup_data["Eenouderhuishoudens"] / makeup_data["Totaal particuliere huishoudens"])
    makeup_data = makeup_data.drop(["Leeftijd referentiepersoon", "Perioden", "Regio's", "Meerpersoonshuishoudens"],
                                   axis=1)
    makeup_data = makeup_data.drop(
        ["Totaal niet-gehuwde paren", "Totaal gehuwde paren", "Eenouderhuishoudens"],
        axis=1)
    return makeup_data.iloc[[4]].reset_index(drop=True)


def read_child_distribution(file: str) -> pd.DataFrame:
    """
    Reads data about how children are distributed among family sizes.
    :param file: filename
    :return: dataframe with distribution of children
    """
    child_data = pd.read_csv(file)
    child_data["Tweeouderhuishoudens"] = child_data["Tweeouderhuishoudens gehuwd"] + child_data[
        "Tweeouderhuishoudens ongehuwd"]
    child_data = child_data.drop(["Tweeouderhuishoudens gehuwd", "Tweeouderhuishoudens ongehuwd"], axis=1)
    child_data = child_data.rename(index={0: 1, 1: 2, 2: 3})

    for i in [0, 1]:
        total = sum(child_data.iloc[:, i])
        p_dist = [j / total for j in child_data.iloc[:, i]]
        child_data.iloc[:, i] = p_dist

    return child_data
