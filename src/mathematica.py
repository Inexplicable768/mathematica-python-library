from typing import Callable
import tkinter as tk
import platform,sys,os,time

### MASTER CLASS ###

class Mathamatica:
    def __init__(self):
        self.name = ""
        self.version = "1.0.0"
        self.inited = False
        self.aboutInf = {}
        self.tempWindows = []
        self.init()
    def init(self, run_debug=False):
        if __name__ == "__main__" or type(__spec__) is None:
            print("Running as main script, not loading __spec__ info.")
            raise RuntimeError("Script is being run as main and may not work as intentended. Please change your config") 
        else:
            __spec__.custom_data = {
                "author": "Alex Hauptman",
                "version": self.version,
                "description": "A fast and efficent library for advanced mathematics, with easy usage and syntax"
            }
        self.inited = True
    def get_version(self,printt=False):
        """Shows current version"""
        if not printt:
            return self.version
        else:
            print(self.version)
    def what_is(self,function: Callable[[int | float, int | float], int | float]) -> str:
        return function.__doc__
    def about(self,printt=False,version=None) -> str:
        """Tells information about the current or inputed version (changelog)"""
        if version == None:
            if printt == True:
                print(self.aboutInfo[self.version])
            return self.aboutInfo[self.version]
        elif type(version) is str:
            if printt == True:
                print(self.aboutInfo[version])
            return self.aboutInfo[version]
        else:
            raise Exception("About function requires valid version name")
    def device_specs(self,printt=False) -> dict:
        """Returns information about the users machine
        Helpful for knowing computing capabilities
        """
        data = {
            "System": platform.system(),
            "Release": platform.release(),
            "Version": platform.version(),
            "Machine": platform.machine(),
            "Processor": platform.processor(),
            "Number of Cores": os.cpu_count(),
            "Processor Speed": self.processor_speed(),
            "Device Name": platform.node(),
            
        }
        if printt == True:
            print(data)
        return data
    def processor_speed(self) -> float:
        """Calculates the users processor speed at a given moment"""
        pass
    def computation_test(self, breakpoint=10000) -> dict:
        """Time the amount of itterations until the breakpoint your PC can execute a simple algorithm
    Take caution using this at high breakpoint levels as the program may hang"""
        start_time = time.process_time()
        stack = ""
        for x in range(breakpoint):
            if self.random_bool() == 0:
                stack+="M"
            else:
                stack = ""
        stack = ""
        end_time = time.process_time()
        cpu_time = end_time - start_time
        return {
            "Raw Time: ": cpu_time,
            "Itterations:":breakpoint
        }
    def save_data(self,data: list | set | tuple | dict):
        """An easier way to download the results or data of something"""
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        file_name = "mathematica-out.txt"
        file_path = os.path.join(downloads_path, file_name)
        dat = data
        with open(file_path, "w") as file:
            file.write(dat)
    # TRIG FUNCTIONS
    def round_to(self, number, num_places):
        return round(number, num_places)
    def sign(self, x: float | int) -> int:
        """Returns the sign (+/-) of a float or int"""
        if x < 0:
            return -1
        elif x > 0:
            return 1
        elif x == 0:
            return 0
        else:
            raise ValueError(f"Cannot compute sign of input '{x}' ")
    def deg(self, radians, pi_prec=5):
        """Convert radians to degrees"""
        return radians * (180/self.PI(pi_prec))
    def rad(self, degrees, pi_prec=5):
        """Convert degrees to radians"""
        return degrees / (180 / self.PI(pi_prec))
    def sinc(self, x:int | float) -> int | float:
        """Calculate sinc for x (in radians)"""
        return self.sin(x) / x
    def sin(self, x, pi_prec=10, dec_places=10):
        """Calculate sine for x (in radians)"""
        x = x % (2 * self.PI(pi_prec))
        result = 0
        for n in range(dec_places):
            term = ((-1) ** n) * (x ** (2 * n + 1)) / self.factorial(2 * n + 1)
            result += term
        return result
    def cos(self, x, pi_prec=10, dec_places=10):
        """Calculate cosine for x (in radians)"""
        result = self.sqrt(1 - (self.sin(x)) ** 2)
        return result
    def tan(self, x:int | float):
        return self.sin(x) / self.cos(x)
    def PI(self, num_digits=15):
        pi = 3
        sign = 1  
        i = 2
        digits_achieved = 0
        while digits_achieved < num_digits:
            # The Nilakantha series 
            pi += sign * (4 / (i * (i + 1) * (i + 2)))
            sign *= -1  
            i += 2  
            pi_str = str(pi)
            decimal_part = pi_str.split('.')[1]  
            digits_achieved = len(decimal_part)
        # Return the calculated Pi to n digits
        return str(round(pi, num_digits))
    def E(self, num_digits=15):
        return (1 + (1/num_digits))**num_digits
    def factorial(self, x):
        if x == 0 or x == 1:
            return 1
        for i in range(x, 1, -1):
            x*=i
        return i
    def gamma_function(self):
        pass
    def sqrt(self, x:float | int) -> float | int:
        return x ** .5
    def cbrt(self, x:float | int) -> float | int:
        return x ** (1/3)
    def hypot(self, a, b):
        """Returns the hypotanuse given sides a and b of a right triangle"""
        return self.sqrt((a**2 + b**2))
    def distance(self, point_a, point_b):
        return (((point_b.x - point_a.x) ** 2) + ((point_b.y - point_a.y)**2)) ** .5
    def n_distance():
        pass
    def nthroot(self, n:float | int, x:float | int, precision) -> float | int : 
        '''Find the nth root of a float. Uses a binary first search'''
        if x < 0 and n % 2 == 0:
            raise ValueError("Cannot compute even root of a negative number")
        if x == 0:
            return 0
        return x ** (1/n)
    def abs(self, x: int | float | complex):
        if x < 0:
            return -x
        else:
            return x
    def ceil(self, x:float | int) -> float | int:
        return self.trunc(x) + 1 
    def floor():
        pass
    def trunc(self, x:float):
        return int(x)
    def max(self, a, b) -> int | float:
        """Return the max of 2 integers or floats"""
        if a > b:
            return a
        return b
    def min(self, a, b):
        if a < b:
            return a
        return b
    def exp(self, x):
        """Returns e^x"""
        return self.E() ** x
    def random_bool():
        """Randomly returns True or False"""
        pass
    def random_int(self, min, max):
        pass
    def rand(self, seed:int):
        if seed == None or "":
            seed = 0x123
    def pick_random(self, list:list | set):
        """Pick a random value from a list"""
    def list_shuffle(self, array):
        """Returns a shuffled version of an inputted array"""
        pass
    def add_matrix(self, a: list | set | "Matrix", b: list | set | "Matrix"):
        isMatrixClass = False
        if type(a) is Matrix:
            a = a.matrix
        if type(b) is Matrix:
            b = b.matrix
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            raise ValueError("Dimension mismatch - matricies cannot be added")
        result = [[0 for i in range(len(a[0]))] for i in range(len(b))]
        if not isMatrixClass:
            return result
        return Matrix(result)
    def subtract_matrix(self, a:list | set | "Matrix", b: list | set | "Matrix"):
        isMatrixClass = False
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            raise ValueError("Dimension mismatch - matricies cannot be added")
    def multiply_matrix(self, a:list | set | "Matrix", b: list | set | "Matrix"):
        pass
    def dot_product():
        pass
    def add_complex(self, a:complex, b:complex):
        return complex(a.real + b.real, a.imag - b.imag)
    def subtract_complex(self, a:complex, b:complex):
        return complex(a.real - b.real, a.imag - b.imag)
    def multiply_complex():
        pass
    def divide_complex():
        pass
    def pow_complex():
        pass
    def conjugate():
        pass
    def max_list(self, array:list):
        pass
    def min_list(self, array:list):
        pass
    def mean(self, array:list):
        pass
    def mode(self, array:list):
        pass
    def median(self, array:list):
        pass
    def solve_system(self, equations: list["Equation"]):
        pass
    def base_n_add(self,num1:str, num2:str, base:int):
        """Take any 2 numbers of base n and add them"""
        def base_n_to_decimal(number, base):
            """Convert base-n number to decimal."""
            decimal_value = 0
            for i, digit in enumerate(reversed(number)):
                # Convert digit to decimal, supports digits > 9
                if digit.isdigit():
                    digit_value = int(digit)
                else:  # Support 'A', 'B', etc. for bases > 10
                    digit_value = ord(digit.upper()) - ord('A') + 10
                decimal_value += digit_value * (base ** i)
            return decimal_value
        
        def decimal_to_base_n(number, base):
            """Convert decimal number to base-n representation."""
            if number == 0:
                return "0"
            
            digits = []
            while number > 0:
                remainder = number % base
                if remainder < 10:
                    digits.append(str(remainder))
                else:  # Convert remainder >= 10 to 'A', 'B', etc.
                    digits.append(chr(ord('A') + remainder - 10))
                number //= base
            return ''.join(reversed(digits))
        
        # Convert both numbers to decimal
        decimal_num1 = base_n_to_decimal(num1, base)
        decimal_num2 = base_n_to_decimal(num2, base)
        
        # Add the two decimal numbers
        decimal_sum = decimal_num1 + decimal_num2
        
        # Convert the sum back to base-n
        return decimal_to_base_n(decimal_sum, base)
    def logic_and(self, a, b):
        return a and b
    def logic_or(self, a, b):
        return a or b
    def logic_not(self, a):
        return not a
    def logic_nand():
        pass
    def logic_xor():
        pass
    def logic_nor():
        pass
    def logic_xnor():
        pass
    def equals(self, a, b):
        """Returns true if 2 values are equal"""
        if a == b:
            return True
        return False

### DATA TYPES ###

class Tensor(Mathamatica):
    """A representation of an n dimensonal list"""
    def __init__(self, array: list, dimensions: tuple | list):
        self.tensor = array
    def sum():
        pass
    def size():
        pass
    def size_bytes():
        pass
class Matrix(Mathamatica):
    """A class that represents a matrix. All standard functions can be used with either a matrix class or built in list, set or tuple
        however using a matrix class is suggested as it allows methods to be performed on a matrix to get information about it.
        operations can be perfromed using Mathamatica.add_matrix(Matrix, Matrix) for example
    """
    def __init__(self, array: list, dimensions: tuple | list | None):
        self.matrix = array
    def sum(self) -> float | int:
        """Sum of all values in the matrix"""
        dimensions = self.get_dimensions()
        i,j = dimensions["rows"], dimensions["cols"]
        sum = 0
        for x in range(i):
            for y in range(j):
                sum+=self.matrix[i][j]
        if type(sum) is not int or float:
            raise TypeError("Matrix must contain numerical values of int or float or complex")
        return sum
    def sum_squared(self):
        return
    def size(self):
        """Alternative method of Matrix.get_dimensions"""
        return self.get_dimensions()
    def size_bytes(self):
        """Calculate the size of a 2D list (and its contents) in bytes."""
        size = sys.getsizeof(self.matrix)  # Base size of the outer list
        for sublist in self.matrix:
            size += sys.getsizeof(sublist)  # Size of each inner list
            for item in sublist:
                size += sys.getsizeof(item)  # Size of each element
        return size
    def get_dimensions(self) -> int:
        """Returns dimensions (rows and columns) of a matrix"""
        return {
            "rows":len(self.matrix),
            "cols":len(self.matrix[0])
        }
    def get_inverse(self):
        """Returns the inverse matrix using Gauss-Jordan elimination method"""
        n = len(self.matrix)
        identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        augmented_matrix = [self.matrix[i] + identity[i] for i in range(n)]
        for i in range(n):
            diag_element = augmented_matrix[i][i]
            if diag_element == 0:
                raise ValueError("Matrix is singular.")
            for j in range(2 * n):
                augmented_matrix[i][j] /= diag_element
            for k in range(n):
                if k != i:
                    factor = augmented_matrix[k][i]
                    for j in range(2 * n):
                        augmented_matrix[k][j] -= factor * augmented_matrix[i][j]
        inverse_matrix = [row[n:] for row in augmented_matrix]
        return Matrix(inverse_matrix)
    def max(self):
        """Return the largest value in the matrix"""
        if self.matrix == []:
            raise ValueError("Matrix cannot be nonetype or empty")
        largest = self.matrix[0]
        for x in self.matrix:
            if x > largest:
                largest = x
        return largest
    def transpose():
        pass
    def rotate90clockwise():
        pass
    def rotate90counter():
        pass
    
class Vector2():
    def __init__(self, a, b):
        self.x = a
        self.y = b
        self.vector = [a, b]
class TwoVarDataSet():
    """A 2 variable statistical data set. Perform methods to calculate or graph certian things"""
    def __init__(self):
        pass
    def get_2var_stats():
        pass
    def get_association():
        pass
    def graph_scatterplot():
        pass
    def graph_bar():
        pass
    def graph_residual_plot():
        pass
class Vector():
    pass
class Frac():
    def __init__(self, fraction:str):
        self.fraction = str(fraction)
        self.demoninator, self.demon = self.fraction.split("/")
    def reduce(self):
        pass
    def get_demon():
        pass
    def get_numer():
        pass
class Function():
    pass
class Equation():
    pass
class Unit():
    """A class defining a numerical value with a unit allowing you to convert between them easily"""
    def __init__(self, unit, value):
        pass
    def convert_to(self, unit):
        pass
class FunctionGraph():
    """Graph a set of functions in a window"""
    def __init__(self, functions:list[str]):
        self.functions = functions
    def open():
        pass
    def plot():
        pass
    def value_at(self, function, x):
        pass
    def scale(self, width, height):
        pass
    def close():
        pass
    def save():
        pass
class Scatterplot():
    pass
class Piechart():
    def __init__(self, labels: list, colors: list[tuple], values: list):
        self.labels = labels
        self.colors = colors
        self.values = values
    def plot():
        pass
    def close():
        pass
class Enum():
    """Data type"""
    def __init__(self, args:list):
        self.args = args





