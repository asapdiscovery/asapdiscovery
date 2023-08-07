from asapdiscovery.data.fitness import parse_fitness_json

df = parse_fitness_json("SARS-CoV-2-Mpro")
df.to_csv("test.csv")