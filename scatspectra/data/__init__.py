import pickle
import pkg_resources

# To load the data from a piclke file
def load_data(filename):
    filepath = pkg_resources.resource_filename(__name__, 'data/' + filename)
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

# Variable to access the data
snp_data = load_data('snp_WSJ_08_02_2024.pkl')