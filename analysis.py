from db.models import GaResults
import numpy as np

results = GaResults.objects.all().last()

sol = results.solution.get('local')

print(sol[0])
# np.savetxt('ga-results.txt', sol[0])
