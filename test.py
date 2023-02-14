import numpy as np
from sklearn.tree import DecisionTreeRegressor
from db.models import SimulationResults


results = SimulationResults.objects.all()

print(results)