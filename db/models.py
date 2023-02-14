from django.db import models
from manage import init_django


init_django()

# define models here

class Swarm(models.Model):
    id=models.AutoField(primary_key=True)
    seed_particle = models.JSONField(default=list)
    total_particles = models.IntegerField(default=15)
    c1 = models.FloatField(default=1.49445)
    c2 = models.FloatField(default=1.49445)
    w = models.FloatField(default=0.7298)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Swarm: {self.id}"


class Solutions(models.Model):
    id=models.AutoField(primary_key=True)
    swarm = models.ForeignKey(Swarm, on_delete=models.CASCADE)  
    particles = models.JSONField(default=list)
    best_particles = models.JSONField(default=list)
    timestamp = models.DateTimeField(auto_now_add=True)

    def to_dict(self):
        return {
            "id": self.id,
            "swarm": self.swarm.id,
            "particles": self.particles,
            "best_particles": self.best_particles,
            "timestamp": self.timestamp,
        }
    

class Simulation(models.Model):
    id=models.AutoField(primary_key=True)
    timestamp = models.DateTimeField(auto_now_add=True)

class SimulationResults(models.Model):
    id=models.AutoField(primary_key=True)
    simulation = models.ForeignKey(Simulation, on_delete=models.CASCADE)
    current_solution = models.JSONField(default=dict)
    best_solution = models.JSONField(default=dict)

    def to_dict(self):
        return {
            'current_solution':self.current_solution.get('solutions'),
            'best_solution':self.best_solution.get('solutions'),
        }

class Ga(models.Model):
    id=models.AutoField(primary_key=True)
    timestamp = models.DateTimeField(auto_now_add=True)

class GaResults(models.Model):
    id=models.AutoField(primary_key=True)
    simulation=models.ForeignKey(Ga, on_delete=models.CASCADE)
    solution=models.JSONField(default=dict)


class Seed(models.Model):
    id=models.AutoField(primary_key=True)
    timestamp = models.DateTimeField(auto_now_add=True)

class SeedResults(models.Model):
    id=models.AutoField(primary_key=True)
    iteration=models.IntegerField(default=0)
    mse=models.FloatField(default=0)
    seed=models.ForeignKey(Seed, on_delete=models.CASCADE)
    solution=models.JSONField(default=dict)

    
    





