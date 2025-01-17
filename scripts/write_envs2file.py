import os

def write_env_to_file(file_path):
  # Open the file in write mode
  with open(file_path, 'w') as file:
      # Iterate over all environment variables
      for key, value in os.environ.items():
          # Write each environment variable to the file
          file.write(f"{key}={value}\n")

# Specify the path to the .env file
env_file_path = '.env'

# Write the environment variables to the .env file
write_env_to_file(env_file_path)

print(f"Environment variables have been written to {env_file_path}")
