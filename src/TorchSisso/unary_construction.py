import torch
import torch.nn.functional as F

# Define user-defined functions
def fun1(x):
    return 2 * x + 1  # Example transformation 1

def fun2(x):
    return x ** 2  # Example transformation 2

def fun3(x):
    return x / 2  # Example transformation 3

class FunctionApplier:
    def construct_function(self, functions_list, var):
        results = torch.empty(var.shape[0],0)  # To store results from each tuple
        text_representations = []  # To store text representations
        
        for functions_tuple in functions_list:
            result = var  # Start with the input variable for each tuple
            text1 = ''  # Initialize text1 as an empty string
            text2 = ''  # Initialize text2 as an empty string
            if not isinstance(functions_tuple, tuple):
                functions_tuple = (functions_tuple,)
            
            for func in functions_tuple:
                if func in globals() and callable(globals()[func]):
                    # Apply the user-defined function to the entire 2D tensor
                    result = globals()[func](result)  # Apply to the entire tensor
                    text1 = f"{func}(" + text1  # Prepare text representation
                    text2 = ")" + text2  # Close the text representation
                elif func in ['relu', 'sigmoid', 'tanh', 'softmax']:
                    # Apply neural network functions
                    result = getattr(F, func)(result)
                    text1 = f"{func}(" + text1
                    text2 = ")" + text2
                elif func.startswith('pow'):
                    # Handle pow with an exponent
                    match = re.search(r'pow\((\d+)\)', func)  # Find exponent
                    if match:
                        exponent = int(match.group(1))  # Extract exponent
                        result = torch.pow(result, exponent)  # Apply pow
                        text2 = f")**{exponent}" + text2
                        text1 = "(" + text1
                else:
                    # Apply standard torch functions
                    result = getattr(torch, func)(result)  
                    text1 = f"{getattr(torch, func).__name__}(" + text1
                    text2 = ")" + text2
            
            results = torch.cat((results,result),dim=1)#.append(result)  # Store the final result after applying all functions in the tuple
            text_representations.append([text1,text2])  # Store the text representation

        return results, text_representations

# =============================================================================
# # Example usage
# if __name__ == "__main__":
#     applier = FunctionApplier()
#     
#     # Create a 2D tensor
#     my_tensor = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
# 
#     # Define the list of function tuples to apply
#     functions_to_apply = [('fun1','fun3', 'relu', 'pow(2)' ,'neg'), ('sin', 'cos'),('abs')]  # Apply multiple functions in sequence
# 
#     # Apply the functions to the tensor
#     results, text_representations = applier.construct_function(functions_to_apply, my_tensor)
#     print(results.shape)
# 
#     # Print results and their representations
#     for i, result in enumerate(results):
#         print(f"Result Tensor after applying functions {functions_to_apply[i]}:")
#         print(result)
#         print(f"Text Representation: {text_representations[i]}")
#         print()
# 
# =============================================================================
