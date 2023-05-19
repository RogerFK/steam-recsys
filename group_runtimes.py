import os

result_path = "experiments/results/"

recommender_names = [ f.path for f in os.scandir(result_path) if f.is_dir() ]
print("recommender,runtime")
total_time = 0
for recommender_name in recommender_names:
    with open(recommender_name + "/time_in_seconds.txt", "r") as f:
        total_time += (time := float(f.read()))
        print(f"\"{recommender_name.split('/')[-1]}\",{time:.2f}")

print(f"Total time: {total_time:.2f} seconds")