import numpy as np

def create_lag_file(file_path, value):
    # Create an array containing the single float value
    data = np.array(value, dtype=float)
    
    # Save the array to the specified file path
    np.save(file_path, data)


# Example usage
file_path = '/mnt/datastore/Harry/Cohort11_april2024/of/M20_D20_2024-05-21_13-47-52_OF1/lag.npy'  # Replace 'your_directory' with the desired location
value = 70.64  # Replace 3.14 with the desired float value
create_lag_file(file_path, value)  
a = float(np.load('/mnt/datastore/Harry/Cohort11_april2024/of/M20_D20_2024-05-21_13-47-52_OF1/lag.npy'))
print(f'lag.npy file created at {file_path} with value {value}')

'''
M20_D19_2024-05-20_13-45-54_OF1 - April 20th broken! First recording that is broken     added lag.npy
M20_D19_OF2                                             - April 20th presumably broken! added lag.npy                                             
M21_D18_OF1                                                - April 20th presumably broken! added lag.npy
M21_D18_OF2                                              - April 20th presumably broken! added lag.npy
M20_D20_2024-05-21_13-47-52_OF1 - April 21st broken!  '''
