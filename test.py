import csv
import pandas as pd

def huishoudens_opbouw(csv_file):
    A = pd.read_csv(csv_file)
    A["fractie eenpersoons"] = A["Particuliere huishoudens: samenstelling/Eenpersoonshuishouden (aantal)"] / A["Particuliere huishoudens: samenstelling/Totaal particuliere huishoudens (aantal)"]
    A["fractie tweepersoons"] = A["Particuliere huishoudens: grootte/Meerpersoonshuishouden/2 personen (aantal)"] / A["Particuliere huishoudens: samenstelling/Totaal particuliere huishoudens (aantal)"]
    A["fractie driepersoons"] = A["Particuliere huishoudens: grootte/Meerpersoonshuishouden/3 personen (aantal)"] / A["Particuliere huishoudens: samenstelling/Totaal particuliere huishoudens (aantal)"]
    A["fractie vierpersoons"] = A["Particuliere huishoudens: grootte/Meerpersoonshuishouden/4 personen (aantal)"] / A["Particuliere huishoudens: samenstelling/Totaal particuliere huishoudens (aantal)"]
    A["fractie meerpersoons"] = A["Particuliere huishoudens: grootte/Meerpersoonshuishouden/5 of meer personen (aantal)"] / A["Particuliere huishoudens: samenstelling/Totaal particuliere huishoudens (aantal)"]
    return A

B = huishoudens_opbouw("Huishoudens__samenstelling__regio_11102021_154809.csv")

print(B["fractie driepersoons"], B["fractie tweepersoons"])


