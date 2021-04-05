# model.x = pyo.Var([1,2,3,4,5,6,7,8,9,10], domain=pyo.NonNegativeReals)

# AC       = IC * r/(1-1/(1+r)^lt) [€/kW]
# AC_wind  = 1100*0.05/(1-1/(1+0.05)^25)
# AC_PV    = 600*0.05/(1-1/(1+0.05)^25)
# AC_gas   = 550*0.05/(1-1/(1+0.05)^30)
# AC_hydro = 0

# Constraints:
# Maximum capacity:   WindMax_swe  = 280*10^6[kW]
#                     PVMax_swe    = 75*10^6
#                     HydroMax_swe = 14*10^6

#                     sum(Wind_swe[time])   <= WindMax_swe
#                     sum(PV_swe[time])     <= PVMax_swe
#                     sum(Hydro_swe[time])  <= hydroMax_swe

#                     WindMax_dan  = 90*10^6
#                     PVMax_dan    = 60*10^6

#                     sum(Wind_dan[time])   <= WindMax_dan
#                     sum(PV_dan[time])     <= PVMax_dan

#                     WindMax_ger  = 180*10^6
#                     PVMax_ger    = 460*10^6

#                     sum(Wind_ger[time])   <= WindMax_ger
#                     sum(PV_ger[time])     <= PVMax_ger

# Demand:          Wind_swe[time] + PV_swe[time] + Hydro_swe[time] + Gas_swe[time]  >= Demand_swe[time]
#                  Wind_dan[time] + PV_dan[time]                   + Gas_dan[time]  >= Demand_dan[time]
#                  Wind_ger[time] + PV_ger[time]                   + Gas_ger[time]  >= Demand_ger[time]


# # objective: minimize
#   sum_time ((costRunning_energy)*Energy_swe[time]+...)+ InvCostEnergy_swe +...
#
#       costRunning_energy = variable cost + Fuel cost
#       investmentCostEnergy = AC * antal kraftverk
#       antal kraftverk = avrundat upp(max(Energy)/ maxenergy_kraftverk))
#
#
#       costRunning_energy = [€/MWh]
#       Energy_swe[time] = [MWh]
#       ==>  costRunning_energy * Energy_swe[time] = [€]
#
#   InvCostEnergy =  investmentCostEnergy = AC * antal kraftverk  [€]
#


from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt

# CONTROL VARIABLES
batteryOn = False


# DATA
countries = ['DE', 'DK', 'SE']
techs = ['Wind', 'PV', 'Gas', 'Hydro', 'Battery']
efficiency = {'Wind': 1, 'PV': 1, 'Gas': 0.4, 'Hydro': 1, 'Battery': 0.9}
discountrate = 0.05

# READ CSV
print("========================= START LOADING DATA =========================")
input_data = pd.read_csv('data/TimeSeries.csv', index_col=[0])
cap_max_data = pd.read_csv('data/capMax.csv', index_col=[0])  # MWh
cost_data = pd.read_csv('data/costs.csv', index_col=[0])
print("========================= DATA LOADED =========================")


# UTILITY FUNCTIONS
def annualize_cost(tech):
    lifetime = cost_data.loc[tech]["lifetime"]
    return discountrate/(1-1/(1+discountrate) ^ lifetime)


def investment_cost(model):
    sum(max([model.prod[node, tech, t] for t in model.time])*cost_data.loc[tech][investment_cost]
        * annualize_cost(tech) for tech in model.gens for node in model.nodes)


def capacity_max(model, n, g):
    capMax = {}
    if g in cap_max_data.columns:
        capMax[n, g] = float(cap_max_data[g].loc[cap_max_data.index == n])
        return 0.0, capMax[n, g]
    elif g == 'Battery' and not batteryOn:
        return 0.0, 0.0
    else:
        return 0.0, None


def demandData():
    demand = {}
    for n in model.nodes:
        for t in model.time:
            demand[n, t] = input_data.iloc[t][f"Load_{n}"]
    return demand


print("========================= CREATE MODEL =========================")
model = ConcreteModel()

print("========================= CREATE SETS =========================")
# SETS
model.nodes = Set(initialize=countries, doc='countries')
model.time = Set(initialize=input_data.index, doc='hours')
model.gens = Set(initialize=techs, doc="Technologies")

print("========================= SET PARAMETERS =========================")
# PARAMETERS
model.demand = Param(model.nodes, model.time, initialize=demandData())
model.efficiency = Param(
    model.gens, initialize=efficiency, doc='Conversion efficiency')


print("========================= CREATE VARIABLES =========================")
# VARIABLES
model.prod = Var(model.nodes, model.gens, model.time,
                 domain=NonNegativeReals,
                 doc="Production")
model.capa = Var(model.nodes, model.gens,
                 bounds=capacity_max, doc='Generator cap')


print("========================= SET CONSTRAINTS =========================")
# CONSTRAINTS


def production_capacity_rule(model, nodes, gens, time):
    return model.prod[nodes, gens, time] <= model.capa[nodes, gens]


model.production_constraint = Constraint(model.nodes, model.gens,
                                         model.time, rule=production_capacity_rule)


def demand_rule(model, nodes, gens, time):
    return sum([model.prod[nodes, tech, time] * model.efficiency[tech] for tech in techs]) >= model.demand[nodes, time]


model.demand_constraint = Constraint(model.nodes, model.gens,
                                     model.time, rule=demand_rule)


print("========================= SET OBJECTIVE FUNCTION =========================")
# OBJECTIVE FUNCTION


def objective_rule_1(model):
    total = {}

    for tech in model.gens:
        total[tech] = 0

    for node in model.nodes:
        for tech in model.gens:
            for t in model.time:
                prod = model.prod[node, tech, t].value
                vc = cost_data.loc[tech]["variable_cost"]
                fc = cost_data.loc[tech]["fuel_cost"]
                ac = annualize_cost(prod, tech)

                total[tech] += prod * \
                    (cost_data.loc[tech]["variable_cost"] +
                     cost_data.loc[tech]["fuel_cost"])

    for tech in model.gens:
        val = total[tech]
        if (val > 0):
            total[tech] += annualize_cost(val, tech)

    return sum(total.values())


def objective_rule(model):
    return sum(model.prod[node, tech, t] * (cost_data.loc[tech]["variable_cost"] + cost_data.loc[tech]["fuel_cost"]) for node in model.nodes for tech in model.gens for t in model.time) + investment_cost(model)


model.objective = Objective(
    rule=objective_rule, sense=minimize, doc='Objective function')


if __name__ == '__main__':
    from pyomo.opt import SolverFactory
    import pyomo.environ
    import pandas as pd

    opt = SolverFactory("gurobi_direct")
    opt.options["threads"] = 4
    print("========================= SOLVING MODEL =========================")

    results = opt.solve(model, tee=True)

    results.write()

    # Reading output - example
    capTot = {}
    for n in model.nodes:
        for g in model.gens:
            capTot[n, g] = model.capa[n, g].value/1e3  # GW

    costTot = value(model.objective) / 1e6  # Million EUR
