def return_example():
    print("This is the first line.")
    return "Return statement"
    print("This line will not be executed.")

def yield_example():
    print("This is the first line.")
    yield "Yield statement"
    print("This line will be executed after the first yield.")
    yield "Second yield statement"
    print("This line will be executed after the second yield.")

# 使用return关键字
result_return = return_example()
print(result_return)

# 使用yield关键字
result_yield = yield_example()
print(next(result_yield))
print(next(result_yield))