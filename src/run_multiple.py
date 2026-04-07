import subprocess
import re
import time

print("="*50)
print("RUNNING GA MULTIPLE TIMES")
print("="*50)

results = []
times = []
runs = 3

for i in range(runs):
    print(f"\nRun {i+1}/{runs}")
    print("-"*30)
    
    start = time.time()
    result = subprocess.run(['python', 'main.py'], capture_output=True, text=True)
    elapsed = time.time() - start
    
    print(result.stdout)
    
    match = re.search(r'Best Distance: (\d+\.?\d*)', result.stdout)
    if match:
        dist = float(match.group(1))
        results.append(dist)
        times.append(elapsed)
        print(f"Distance: {dist:.2f}")
        print(f"Time: {elapsed:.2f}s")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for i, dist in enumerate(results):
    print(f"Run {i+1}: {dist:.2f} ({times[i]:.2f}s)")

print(f"\nBest: {min(results):.2f}")
print(f"Worst: {max(results):.2f}")
print(f"Average: {sum(results)/len(results):.2f}")
print(f"Avg Time: {sum(times)/len(times):.2f}s")