import pandas as pd

class track_statistics(object):
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
    def __init__(self, person_id, age, status=0, vaccination_readiness=True, days_since_infection=0):
        self.person_id = person_id  # corresponds to the index within the adjacency matrix
        self.age = age  # age of the person
        self.status = status  #
        self.days_since_infection = days_since_infection
        self.vaccination_readiness = vaccination_readiness
        self.household = 0
        self.overestimate = {}

    def overestimation(self, inp):
        if inp in self.overestimate.keys():
            self.overestimate[inp] += 1
        else:
            self.overestimate[inp] = 1


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

