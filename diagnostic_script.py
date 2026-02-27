# Diagnostic Script for Inspecting Template Data Structure

# This script inspects the template data structure for the lyrical-video project.

import json
import os

# Path to the templates directory
TEMPLATES_DIR = os.path.join(os.getcwd(), 'path/to/templates') # Update the path as necessary

# Function to retrieve and print template data

def inspect_templates(directory):
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.json'):  # Assuming templates are JSON files
                with open(os.path.join(directory, filename), 'r') as file:
                    data = json.load(file)
                    print(f'\nTemplate: {filename}')
                    print(json.dumps(data, indent=4))  # Pretty print the JSON data
    except Exception as e:
        print(f'Error reading templates: {e}')  

# Run the inspection
inspect_templates(TEMPLATES_DIR)
