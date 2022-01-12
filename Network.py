import random
import random as rd
import sys
import pandas as pd

from Read_data import *
from Classes import *


def couple_childless(house_dict: dict, people_dict: dict, amount_people: object, household_id: int) \
                    -> [dict, dict, object, int]:
    """This function creates the number of couples without children estimated to be in the network and places them in two person houses

    input:
    house_dict = dictionary with key size of house and value number available
    people_dict = dictionary with key age and value people not in a house of that age
    amount_people = dataframe with the number of people of a certain age living in a couple.
    household_id = value that gives every household a key id for easier identification

    This function changes all of the input variables and gives them back in with the new household accounted for

    output:
    house_dict = dictionary with key size of house and value number available
    people_dict = dictionary with key age and value people not in a house of that age
    amount_people = dataframe with the number of people of a certain age living in a couple.
    household_id = value that gives every household a key id for easier identification
    """

    couples = list(amount_people["Fraction couple without children"])
    number_of_couples = int(sum(couples) / 2)

    for couple in range(number_of_couples):
        available_ages = [index for index in range(len(couples)) if couples[index] > 0]
        amount_available = [index for index in couples if index > 0]
        age_1 = available_ages[0]
        person_1 = people_dict[age_1].pop(0)
        couples[age_1] += -1
        available_ages1 = [index for index in range(len(couples)) if couples[index] > 0]
        age_2 = random.choice(available_ages1[:5])
        person_2 = people_dict[age_2].pop(0)
        couples[age_2] += -1

        house_dict[2].append(household_class(household_id, 2))
        house_dict[2][-1].add_member(person_1)
        person_1.update_household(house_dict[2][-1])
        house_dict[2][-1].add_member(person_2)
        person_2.update_household(house_dict[2][-1])
        household_id += 1

    return house_dict, people_dict, amount_people, household_id


def how_many_children(available_children: list) -> int:
    """
    This function chooses 1,2 or 3 children depending on whether there are still that many children available
    :param available_children: list with the amount of household with that many children left to be placed
    :return: a random number chosesn from that list
    """

    dict = {}
    for i in range(len(available_children)):
        if available_children[i] > 0:
            dict[i + 1] = available_children[i]
    number_of_children = random.choice(list(dict.keys()))

    return number_of_children


def make_households(N: int, file1: str, file2: str, file3: str, people_dict: dict) -> dict:
    """"
    This function creates all the households in the simulation. It first calculates how many of each type there should be.
    It then proceeds to create these. It uses demographical data as input.
    input:
    :param N: Size of network
    :param file1: filename
    :param file2: filename
    :param file3: filename
    :param people_dict: dictionary with key age and value all people objects of that age
    :return: dictionary with household objects. Key is size and value list of household objects
    """
    makeup_data = read_makeup_households(file1)
    houses = calculate_houses(N, makeup_data)
    child_dist = read_child_distribution(file3)
    household_data = read_households(N, file2)

    amount_people, total_child_or_couple, non_couples = calculate_household(N, household_data)

    household_id = 0
    house_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    # available_houses = houses.loc[0, :].values.tolist()[1:]

    house_dict, people_dict, amount_people, household_id = couple_childless(house_dict, people_dict,
                                                                            amount_people,
                                                                            household_id)

    # couples with children
    couples = list(amount_people["Fraction couple with children"])
    children = list(amount_people["Fraction child"])
    number_of_couples = int(sum(couples) / 2)

    one_parent = houses["One Parent"][0]
    one_parent_dist = [int(one_parent * j) for j in child_dist.iloc[:, 0]]

    one_parent_houses = [i * j for i, j in zip(one_parent_dist, [1, 2, 3])]

    two_parent = number_of_couples
    two_parent_dist = [int(two_parent * j) for j in child_dist.iloc[:, 1]]

    two_parent_houses = [i * j for i, j in zip(two_parent_dist, [1, 2, 3])]

    # The number of children does not match up with the households therefore we will add some 4 and 5 person households.
    # determine the amount of children left to be placed

    remaining_children = int(sum(children) - sum(one_parent_houses) - sum(two_parent_houses))

    four_houses = round(2 / 3 * remaining_children)
    while (remaining_children - four_houses) % 2 != 0:
        four_houses += 1

    two_parent_dist[2] -= four_houses
    two_parent_dist.append(four_houses)
    five_houses = (remaining_children - four_houses) / 2
    two_parent_dist[2] -= five_houses
    two_parent_dist.append(five_houses)
    two_parent_houses = [i * j for i, j in zip(two_parent_dist, [1, 2, 3, 4, 5])]

    assert sum(two_parent_houses) + sum(one_parent_houses) == sum(
        children)  # check that te children to be placed is equal to the children

    # create one parent households
    for parent in range(sum(one_parent_dist)):
        # choose a parent from the available ages
        available_ages = [index for index in list(people_dict.keys())[19:] if len(people_dict[index]) > 0]
        age = random.choice(available_ages)
        person = people_dict[age].pop(0)

        number_of_children = how_many_children(one_parent_dist)
        one_parent_dist[number_of_children - 1] -= 1
        house_dict[1 + number_of_children].append(household_class(household_id, 1 + number_of_children))
        house_dict[1 + number_of_children][-1].add_member(person)
        person.update_household(house_dict[1 + number_of_children][-1])

        for j in range(number_of_children):
            available_ages = [index for index in range(len(children)) if children[index] > 0]
            age_child = random.choice(available_ages)
            child = people_dict[age_child].pop(0)
            children[age_child] += -1
            house_dict[1 + number_of_children][-1].add_member(child)
            child.update_household(house_dict[1 + number_of_children][-1])
        household_id += 1

    couples = list(amount_people["Fraction couple with children"])
    # two parent households
    R = []
    for couple in range(int(sum(two_parent_dist))):
        number_of_children = how_many_children(two_parent_dist)
        two_parent_dist[number_of_children - 1] -= 1

        house_dict[2 + number_of_children].append(household_class(household_id, 2 + number_of_children))
        R.append(house_dict[2 + number_of_children][-1])

        for j in range(number_of_children):
            available_ages = [index for index in range(len(children)) if children[index] > 0]
            age_child = random.choice(available_ages)
            child = people_dict[age_child].pop(0)
            children[age_child] += -1
            house_dict[2 + number_of_children][-1].add_member(child)
            child.update_household(house_dict[2 + number_of_children][-1])

    for house in R:  # choose parents for the previously created house with children
        available_ages1 = [index for index in range(len(couples)) if couples[index] > 0]
        age_1 = random.choice(available_ages1)
        if len(people_dict[age_1]) == 0:  # imperfect fix for bug preventing final parent from being placed
            break
        person_1 = people_dict[age_1].pop(0)
        couples[age_1] += -1

        available_ages2 = []
        while available_ages2 == []:  # find partners close in age but extend search of non available
            if sum([index for index in range(len(couples)) if couples[index] > 0][age_1:]) > 0:
                available_ages2 = [index for index in range(len(couples)) if couples[index] > 0][
                                  max(18, age_1 - 5):age_1 + 5]
            else:
                available_ages2 = [index for index in range(len(couples)) if couples[index] > 0]

        age_2 = random.choice(available_ages2)
        if len(people_dict[age_2]) == 0:  # imperfect fix for bug preventing final parent from being placed
            break
        person_2 = people_dict[age_2].pop(0)
        couples[age_2] += -1

        house.add_member(person_1)
        person_1.update_household(house)
        house.add_member(person_2)
        person_2.update_household(house)
        household_id += 1

    return house_dict


def create_people(N: int, dataframe: pd.DataFrame, vaccination_readiness: float) -> [list, dict]:
    """
    input:
    N = number of people to be created
    dataframe = contains data about age distribution
    vaccination_readiness = willingness to vaccinate

    output:
    people = list of person objects
    dict = dictionary with key age and value a list of the person objects of that age.
    """


    dict = {}
    people = []

    # some changes to alter data to a usable type
    household_data = read_households(N, "Datafiles/Personen_in_huishoudens_naar_leeftijd_en_geslacht.csv")
    amount_people = calculate_household(N, household_data)[0]
    amount_people2 = (amount_people["Fraction child"] + amount_people["Fraction couple with children"] + amount_people[
        "Fraction couple without children"]).tolist()
    # amount_people.insert(0, 0)

    R = dataframe["Start of age group"].tolist()
    a_age = [R[i] - R[i - 1] for i in range(1, len(R))]
    added = 0
    for i in range(len(amount_people2)):
        if a_age[i] < amount_people2[i]:
            added += amount_people2[i] - a_age[i]
            a_age[i] = int(amount_people2[i])

    for i in range(int(added)):
        index = [k for k in range(32, len(amount_people2)) if a_age[k] > amount_people2[k]]
        remove = random.choice(index)
        a_age[remove] -= 1

    a_age.insert(0, 0)

    for i in range(1, len(a_age)):
        a_age[i] = a_age[i] + a_age[i - 1]
        dataframe["Start of age group"].iloc[i] = a_age[i]

    for age in dataframe.index[:-1]:  # add people of all but highest age
        dict[age] = []
        for i in range(dataframe['Start of age group'][age],
                       dataframe['Start of age group'][age + 1]):  # add the fraction belonging to that age
            rand = rd.randrange(0, 1)
            if rand < vaccination_readiness:
                people.append(person(i, age, False))
                dict[age].append(people[-1])
            else:
                people.append(person(i, age, True))  # create person and add to list of people
                dict[age].append(people[-1])
            dataframe.loc[age, "Age group class object"].add_member(people[-1])

    # create people of the highest age
    age = dataframe.index[-1]
    dict[age] = []
    for i in range(dataframe['Start of age group'].iloc[-1], N):
        rand = rd.randrange(0, 1)
        if rand < vaccination_readiness:
            people.append(person(i, age, False))
            dict[age].append(people[-1])
        else:
            people.append(person(i, age, True))  # create person and add to list of people
            dict[age].append(people[-1])
        dataframe["Age group class object"].iloc[-1].add_member(people[-1])

    return people, dict


def create_subnetwork(group1: age_group_class, group2: age_group_class, degree: float, i0: int, j0: int) -> list:
    """
    This function links two age groups to the degree given by the Polymod study.
    :param group1: age group object
    :param group2: age group object
    :param degree: number of connections needed to be made between the two age groups
    :param i0: offset in main matrix
    :param j0: offset in main matrix
    :return: sparse matrix list with all connections between the two age groups
    """
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
            if not i == j:  # no links to self
                if group1.members[r2].household != 0:  # make sure the network is not over connected
                    if group2.age_group_id not in group1.members[r2].overestimate:
                        out.append((i, j))
                        out.append((j, i))
                        over = [T for T in group1.members[r2].household.members if
                                (T != group1.members[r2]) and (T.age in group2.ages)]
                        group1.members[r2].overestimation(group2.age_group_id)
                        for T in over:
                            group1.members[r2].overestimation(group2.age_group_id)

                    if group1.members[r2].overestimate[group2.age_group_id] < degree:
                        out.append((i, j))
                        out.append((j, i))
                        group1.members[r2].overestimation(group2.age_group_id)
                    else:
                        continue

                if group1.members[r2].household == 0:
                    out.append((i, j))
                    out.append((j, i))

        return out


def create_network(dataframe: pd.DataFrame, people: list, contact_data: np.ndarray) -> list:
    """This function creates an n by n adjacency matrix A
    that defines a random network with n vertices,
    where vertex i has a number of contacts determined
    by the degree matrix d.

    This is a fast method, linear in the number of edges |E|,
    so it can be used to create large networks.
    a is stored as a list of pairs (i,j), sorted by rows, and within
    each row by column index.
    :param dataframe: pandas dataframe containing data needed to create the network
    :param people: list of people objects
    :param contact_data: matrix M with degree of connectednes between age groups. M_{i,j} is average degree between age group i and age group j
    :return: sparse matrix list with all connections between the two age groups
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
    a = list(dict.fromkeys(out))

    return a


def read_households(N: int, household_file: str) -> pd.DataFrame:
    """
     Read the file containing data about the make up of household demographics
    :param N: number of people in the network to be simulated
    :param household_file: filename
    :return: pandas dataframe with data about households
    """
    household_data = pd.read_csv(household_file)
    household_data = household_data.fillna(0)  # replace missing data with zero values
    household_data = household_data.drop([0, 97, 98, 99, 100])  # remove extra data
    household_data = household_data.drop(["Geslacht", "Leeftijd", "Perioden"], axis=1)

    # Change notation of number to not use "." for seperation
    for column in household_data.columns:
        household_data[column] = household_data[column].astype(str).apply(lambda x: x.replace('.', ''))
        household_data[column] = household_data[column].astype(int)

    household_data = household_data.set_axis(['Children living at home', 'Couple without children', 'A', 'B'], axis=1)
    household_data["Couple with children"] = household_data["A"] + household_data["B"]
    household_data = household_data.drop(["A", "B"], axis=1)  # remove extra data

    # the amount of children in the network is determined by the amount of couples

    household_data = household_data.reset_index(drop=True)

    return household_data


def calculate_household(N: int, dataframe: pd.DataFrame) -> [pd.DataFrame, int, int]:
    """
    Calculate how many households of what type are present in the network.
    :param N: number of people in the network to be modelled
    :param dataframe: pandas dataframe containing data about households.
    :return: how many households of what type are present in the network.
    """
    # calculate fractions of population
    total = read_age_distribution('Datafiles/CBS_NL_population_20200101.txt').sum()[0]
    total_child_or_couple = round(N * dataframe.sum().sum() / total)

    out = pd.DataFrame()
    new_names = ["Fraction child", "Fraction couple without children", "Fraction couple with children"]
    total_people = dataframe.sum().sum()
    for column, name in zip(dataframe.columns, new_names):
        out[name] = round(total_child_or_couple * (dataframe[column] / total_people))
    for i in ["Fraction couple without children", "Fraction couple with children"]:
        if out.loc[:, i].sum() % 2 != 0:
            age = rd.choice(range(19, len(out[i].tolist())))
            out.loc[age, i] = out.loc[age, i] + 1
            r = 3

    # make sure number of people in data matches N. Must remove from children since otherwise couples possibly uneven.
    while out.sum().sum() > total_child_or_couple:
        max = out["Fraction child"].idxmax()
        out.loc[max, "Fraction child"] = out.loc[max, "Fraction child"] - 1

    while out.sum().sum() < total_child_or_couple:
        max = out["Fraction child"].idxmax()
        out.loc[max, "Fraction child"] = out.loc[max, "Fraction child"] + 1

    assert out.sum().sum() == total_child_or_couple

    return out, total_child_or_couple, N - total_child_or_couple


def calculate_houses(N: int, dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the number of houses in the network needed for this size network according to demographic data
    :param N: number of people in the network to be modelled
    :param dataframe: pandas dataframe with data bout households
    :return: dataframe with  the number of houses of certain sizes present in the network
    """
    out = pd.DataFrame()
    makeup_data_list = dataframe.values[0][:-1]

    people_in_household = [1, 1, 2, 3, 4, 5]
    fractions = [i / makeup_data_list[0] for i in makeup_data_list]
    number_of_households = [int((N * i) // k) for i, k in zip(fractions, people_in_household)]
    number_of_households[0] = sum(number_of_households[1:])
    column_names = ["Total", "One person", "Two Person", "Three person", "Four person", "Five or more person"]

    for name, value in zip(column_names, number_of_households):
        out[name] = [value]

    out["One Parent"] = int(dataframe["One parent household"] * sum(number_of_households[2:]))

    return out
