import pandas as pd

class track_statistics(object):
    """This Class is used to track how many people are in a certain state at time t"""
    def __init__(self, tracker_id: int = 1):
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

    def update_statistics(self, tracker_changes: dict):
        # This method is used to safe the numbers for a new timestep
        # it takes a dataframe with the new numbers a input and appends it
        self.data.index.name = "timestep"
        self.data = self.data.append(tracker_changes, ignore_index=True)

    def init_empty_changes(self) -> dict:
        # returns an empty dictionary for tracking the next step
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

    def empty_changes(self) -> dict:
        #returns the numbers form the previous timestep so that they can be updated
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
        # saves the statistics to csv file. Takes a file name as input.
        self.data.to_csv(filename)

    def read_data(self, filename: str) -> pd.DataFrame:
        # reads data from csv. Usefull for loading old statistics.
        # input is a filename
        return pd.read_csv(filename)


class person(object):
    """The person class object"""
    def __init__(self, person_id, age, status=0, vaccination_readiness=True, days_since_infection=0):
        self.person_id = person_id  # corresponds to the index within the adjacency matrix
        self.age = age  # age of the person
        self.status = status  # health status
        self.days_since_infection = days_since_infection # keeps track of the days since infection
        self.vaccination_readiness = vaccination_readiness # I/O value but determines in the person will take the vaccine when offered
        self.household = 0 #default not in a houehold. Will be placed into one later
        self.overestimate = {} #used for not having a to high number of connections

    def overestimation(self, inp):
        #checks if the numbe rof connections to a certain age group is too high.
        if inp in self.overestimate.keys():
            self.overestimate[inp] += 1
        else:
            self.overestimate[inp] = 1


    def update_household(self, household: object):
        # used for placing people into a houshold.
        # takes a household class object as input
        self.household = household

    def update_status(self, new_status: int):
        # updates the status.
        # takes status as input
        self.status = new_status

    def update_days_since_infection(self, new_days: int):
        # keeps track of the days since infection
        self.days_since_infection = new_days

    def how_many_days(self) -> int:
        # used for finding out how many days someone has been infected
        return self.days_since_infection


class group_class(object):
    # parent class for all groups
    def __init__(self, group_id: int):
        self.id = group_id
        self.members = []

    def add_member(self, person: object):
        # add person to the group
        self.members.append(person)

    def size_group(self) -> int:
        # returns the size of the group
        return len(self.members)


class household_class(group_class):

    def __init__(self, household_id: int, number_of_members: int):
        self.number_of_members = number_of_members
        super().__init__(household_id)


class age_group_class(group_class):
    # age group class. Used for making the Polymod network.
    def ages_in_group(self, age1: int, age2: int) -> list:
        # returns the ages in the group.
        return [i for i in range(age1, age2)]

    def __init__(self, age_group_id: int, from_age: int, to_age: int):
        self.ages = self.ages_in_group(from_age, to_age)
        self.age_group_id = age_group_id
        super().__init__(age_group_id)

