import pyomo.environ as pyo
from pyomo.opt import SolverFactory

bikes = ["Bike 1", "Bike 2", "Bike 3"]
I = [3000, 7000, 1000, 290]
P = [4000, 11000, 1150, 340]
budget = 19000

# Create model
model = pyo.ConcreteModel()

# Set variables
model.x = pyo.Var([i for i in range(1, len(I)+1)],
                  domain=pyo.NonNegativeIntegers)


# Set objective function
model.OBJ = pyo.Objective(
    expr=sum([(P[i] - I[i])*model.x[i+1] for i in range(0, len(I))]),
    sense=pyo.maximize)

# Set constraints
model.Constraint1 = pyo.Constraint(
    expr=sum([I[i]*model.x[i+1] for i in range(0, len(I))]) <= budget)


# Create solver and solve
opt = SolverFactory("gurobi_direct")
# opt.options["threads"] = 4
print("")
print('============================ Solving ============================')
results = opt.solve(model, tee=True)
results.write()

print("")
print('============================ Results ============================')
print("Print values for each variable explicitly")
for i in model.x:
    print(str(model.x[i]), model.x[i].value)
print("")

#
# Print values for all variables
#
print("Print values for all variables")
for v in model.component_data_objects(pyo.Var):
    print(str(v), v.value)
