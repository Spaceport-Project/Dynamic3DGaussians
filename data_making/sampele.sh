#!/bin/bash

my_function() {
    echo "This is inside the function."
    
    # Exit the script from within the function
    exit 1
    
    echo "This line will never be executed."
}

echo "Before calling the function."

my_function

echo "This line will never be executed because the script exits in the function."
