# Define empty array
inputs=()

# Read user input into array
for i in {1..3}; do
    read -p "Enter value $i: " value
    inputs+=("$value")
done

# Print all elements
echo "You entered: ${inputs[@]}"
