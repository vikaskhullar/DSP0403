print("Hello World")

#Demonstrate input() and print() Functions

country = input("Which country do you live in?")
print("I live in {0}".format(country))


#Demonstrate the Positional Change of Indexes of Arguments

a = 10
b = 20
print("The values of a is {0} and b is {1}".format(a, b))
#Interchange the positions
print("The values of b is {1} and a is {0}".format(a, b))


# Demonstrate the Use of f-strings with print() Function

country = input("Which country do you live in?")

print(f"I live in {country}")


# Demonstrate int() Casting Function

float_to_int = int(3.5)
string_to_int = int("1")
print(f"After Float to Integer Casting the result is {float_to_int}")
print(f"After String to Integer Casting the result is {string_to_int}")


# Demonstrate float() Casting Function

int_to_float = float(4)
string_to_float = float("1")
print(f"After Integer to Float Casting the result is {int_to_float}")
print(f"After String to Float Casting the result is {string_to_float}")

# Demonstrate str() Casting Function

int_to_string = str(8)
float_to_string = str(3.5)
print(f"After Integer to String Casting the result is {float_to_string}")
print(f"After Float to String Casting the result is {float_to_string}")

# Demonstrate chr() Casting Function

ascii_to_char = chr(100)
print(f'Equivalent Character for ASCII value of 100 is {ascii_to_char}')

