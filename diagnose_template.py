import pickle
import os

# Define the path to the pickle file
pickle_file_path = 'path/to/your/pickle_file.pkl'

# Function to load pickle file and inspect the data
def inspect_pickle_data(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
        # Assuming data structure has 'comprehensive_statistics' key
        if 'comprehensive_statistics' in data:
            stats = data['comprehensive_statistics']
            print("Comprehensive Statistics:")
            
            # Inspecting each property
            for key in ['text', 'background', 'colors', 'animation properties']:
                if key in stats:
                    print(f"{key.capitalize()}: {stats[key]}")
                else:
                    print(f"{key.capitalize()} not found in comprehensive_statistics.")
        else:
            print("'comprehensive_statistics' key not found in the data.")


# Call the inspection function
inspect_pickle_data(pickle_file_path)
