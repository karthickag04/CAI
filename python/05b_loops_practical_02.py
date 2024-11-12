# 1. **For Loop Example: Iterating over a list**
fruits = ["apple", "banana", "cherry"]
# Looping through each element in the 'fruits' list
for fruit in fruits:
    print(fruit)  # Prints each fruit name

# Output:
# apple
# banana
# cherry


# 2. **For Loop Example: Using range()**
# range(5) generates a sequence of numbers from 0 to 4
for i in range(5):  
    print(i)  # Prints numbers 0 to 4

# Output:
# 0
# 1
# 2
# 3
# 4


# 3. **For Loop Example: Iterating through a string**
text = "hello"
# Looping through each character in the string 'text'
for char in text:
    print(char)  # Prints each character

# Output:
# h
# e
# l
# l
# o


# 4. **While Loop Example: Repeating a task until a condition is met**
count = 0
# While loop continues as long as count is less than 5
while count < 5:
    print(count)  # Prints count value
    count += 1  # Increment count to avoid infinite loop

# Output:
# 0
# 1
# 2
# 3
# 4


# 5. **While Loop Example: Infinite loop with break**
# Creating an infinite loop, which will break after one iteration
while True:
    print("This is an infinite loop")  # Prints message
    break  # Breaks the loop after one iteration

# Output:
# This is an infinite loop


# 6. **Break Statement Example: Exiting the loop early**
for i in range(10):
    if i == 5:
        break  # Exits the loop when i equals 5
    print(i)  # Prints numbers from 0 to 4

# Output:
# 0
# 1
# 2
# 3
# 4


# 7. **Continue Statement Example: Skipping an iteration**
for i in range(5):
    if i == 3:
        continue  # Skips the rest of the loop when i equals 3
    print(i)  # Prints numbers except 3

# Output:
# 0
# 1
# 2
# 4


# 8. **Else Clause in Loops Example: Executing code after a loop finishes**
for i in range(5):
    print(i)  # Prints numbers from 0 to 4
else:
    print("Loop completed without break.")  # This runs after the loop finishes

# Output:
# 0
# 1
# 2
# 3
# 4
# Loop completed without break.


# 9. **Else Clause Example: Skipping 'else' when break is used**
for i in range(5):
    if i == 3:
        break  # Exits the loop when i equals 3
    print(i)  # Prints numbers from 0 to 2
else:
    print("Loop completed without break.")  # This will not be executed

# Output:
# 0
# 1
# 2


# 10. **Nested Loops Example: Looping through multiple dimensions (nested loops)**
# Outer loop
for i in range(3):
    # Inner loop
    for j in range(2):
        print(f"i={i}, j={j}")  # Prints combination of i and j

# Output:
# i=0, j=0
# i=0, j=1
# i=1, j=0
# i=1, j=1
# i=2, j=0
# i=2, j=1


# 11. **Nested While Loop Example**
i = 0
# Outer loop
while i < 3:
    j = 0
    # Inner while loop
    while j < 2:
        print(f"i={i}, j={j}")  # Prints combination of i and j
        j += 1  # Increment inner loop counter
    i += 1  # Increment outer loop counter

# Output:
# i=0, j=0
# i=0, j=1
# i=1, j=0
# i=1, j=1
# i=2, j=0
# i=2, j=1


# 12. **Using range with for and else example**
# We can use range in loops and combine it with else
for i in range(5):
    print(i)  # Prints numbers from 0 to 4
else:
    print("Finished looping through range.")  # This executes after the loop

# Output:
# 0
# 1
# 2
# 3
# 4
# Finished looping through range.
