from src.create_and_load_dataset import load_dataset
from src.trainer import training_function
from src.load_fine_tuned_model import query_solver
if __name__ == "__main__":
    load_dataset()
    training_function()
    query =  input("How can i help you?  ")
    query_solver(query)
    
    
    