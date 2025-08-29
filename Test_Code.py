# Welcome to Python in Cursor!
# This is your first Python file

def greet(name="World"):
    """A simple greeting function"""
    return f"Hello, {name}!"

def main():
    print("🎉 Welcome to Python development in Cursor!")
    print(greet())
    print(greet("Python Developer"))
    
    # Let's do some basic Python operations
    numbers = [1, 2, 3, 4, 5]
    print(f"\nNumbers: {numbers}")
    print(f"Sum: {sum(numbers)}")
    print(f"Average: {sum(numbers) / len(numbers):.2f}")

if __name__ == "__main__":
    main() 