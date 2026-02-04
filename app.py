from predictor import get_predicted_value, helper

symptoms = input("Enter symptoms (comma separated): ")
user_symptoms = [s.strip() for s in symptoms.split(",")]

disease = get_predicted_value(user_symptoms)
desc, pre, med, die, wrk = helper(disease)

print("\nPredicted Disease:", disease)
print("\nDescription:", desc)
print("\nPrecautions:", pre)
print("\nMedicines:", med)
print("\nDiet:", die)
print("\nWorkout:", wrk)
