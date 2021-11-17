import os
# This file contains the default datasets

# Source: CBS, Statistics Netherlands, https://opendata.cbs.nl
Population_dataset = os.path.join("Datafiles", "CBS_NL_population_20200101.txt")

# Source: Levin et al. (2020)
Fatality_distribution_dataset = os.path.join("Datafiles", "fatality_rates_Levin2020.txt")

# Source: J. Mossong et al., "Social Contacts and Mixing Patterns Relevant
# to the Spread of Infectious Diseases", PLOS Medicine (2008),
# https://doi.org/10.1371/journal.pmed.0050074
Polymod_contacts_dataset = os.path.join("Datafiles", "contacts_polymod_NL.txt")
Polymod_participants_dataset = os.path.join("Datafiles", "participants_polymod_NL.txt")

# Source: CBS
Child_distribution_dataset = os.path.join("Datafiles", "Kinder verdeling.csv")

# Source: CBS, Statistics Netherlands, https://opendata.cbs.nl
Household_makeup_dataset = os.path.join("Datafiles", "Huishoudens__samenstelling__regio_11102021_154809.csv")

# Source: CBS, Statistics Netherlands, https://opendata.cbs.nl
People_in_household_dataset = os.path.join("Datafiles", "Personen_in_huishoudens_naar_leeftijd_en_geslacht.csv")


# now wrap everything into a dictionary for easy access:
def filenames_dictionary():
    return {"Population_dataset": Population_dataset,
            "Fatality_distribution_dataset": Fatality_distribution_dataset,
            "Polymod_contacts_dataset": Polymod_contacts_dataset,
            "Polymod_participants_dataset": Polymod_participants_dataset,
            "Child_distribution_dataset": Child_distribution_dataset,
            "Household_makeup_dataset": Household_makeup_dataset,
            "People_in_household_dataset": People_in_household_dataset}
