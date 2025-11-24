import mteb

model_name = "sentence-transformers/all-MiniLM-L6-v2"

model = mteb.get_model(model_name)

tasks = mteb.get_tasks(tasks=["Banking77Classification"])

results = mteb.evaluate(model, tasks=tasks)

print(results)