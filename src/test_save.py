import os
from datetime import datetime

def test_save():
    print(f"Current directory: {os.getcwd()}")
    
    results_dir = 'results'
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created '{results_dir}' directory")
    else:
        print(f"'{results_dir}' directory already exists")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_file = f'{results_dir}/test_file_{timestamp}.txt'
    
    with open(test_file, 'w') as f:
        f.write("This is a test file")
    
    print(f"Test file saved to: {test_file}")
    
    if os.path.exists(test_file):
        print("SUCCESS! File exists!")
        print(f"Full path: {os.path.abspath(test_file)}")
    else:
        print("FAILED! File not found!")

if __name__ == "__main__":
    test_save()