#############################################################################
"""Class of methods to post-process lethe results with pyvista"""
#############################################################################

#Import modules
import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm

#Define class:
class Lethe_pyvista_tools():

    def __init__(self, case_path, prm_file_name):

        self.path_case = case_path
        self.prm_file = prm_file_name

        #Read .prm file to dictionary
        #Create dictionary
        self.prm_dict = {}


        #Use .prm path as argument
        with open(self.path_case + '/' + self.prm_file) as file:
            #Loop trhough lines in .prm
            for line in file:
                #If the line has 'subsection' in it (and it is not commented)
                if 'subsection' in line and not '#' in line:
                    #Remove "subsetction"
                    subsection_clean_line = line.replace('subsection', '')
                    #Clean line from spaces and assign key-value
                    subsection_clean_line = subsection_clean_line.strip()
                #Else, if the line has 'set' in it (and it is not commented)
                elif 'set' in line and not '#' in line:
                    #Remove "set" from string "line"
                    clean_line = line.replace('set', '')
                    #Split the string in [variable, value]
                    clean_line = clean_line.split('=')
                    #Clean line from spaces
                    for element in range(len(clean_line)):
                        clean_line[element] = clean_line[element].strip()
                    #Convert values to float when possible
                    try:
                        clean_line[1] = float(clean_line[1])
                    except:
                        pass
                    #Define [variable, value] as key and value in the dictionary
                    #If 'set' is a 'Function expression' or 'type'
                    if clean_line[0] == 'Function expression' or clean_line[0] == 'type':
                        #Attribute the name of the subsection above
                        self.prm_dict[subsection_clean_line] = clean_line[1]
                    else:
                        #Otherwise, attribute the set name
                        self.prm_dict[clean_line[0]] = clean_line[1]

            print(f'Successfully constructed. To see the .prm dictionary, print($NAME.prm_dict)')
        #Define path where vtu files are
        self.path_output = self.path_case + self.prm_dict['output path'].replace('.', '')

    def reader(self, i = 0):
        exec(f'self.df_{i} = pv.read(f\'{self.path_output}/{self.list_vtu[i]}\')')

    #Read fluid or particle information from vtu files
    def read_lethe_to_pyvista(self, pvd_name, first = 0, last = None, interval = 1):

        self.pvd_name = pvd_name
        #Read name of files in .pvd file
        files = pd.read_csv(f'{self.path_output}{pvd_name}',sep='"',skiprows=6, usecols=[1, 5], names = ['time', 'vtu'])
        #clean data from NaN's
        files = files.dropna()
        #Create a list of time-steps
        self.time_list = files['time'].tolist()
        #Create a list of all files' names
        self.list_pvtu = files['vtu'].tolist()
        #Format files' names
        self.list_vtu = [i.replace('.pvtu', '.0000.vtu') for i in self.list_pvtu]

        if last == None:
            self.list_vtu = self.list_vtu[first::interval]
            self.time_list = self.time_list[first::interval]
            self.first = first
            self.interval = interval
            self.last = len(self.time_list) - 1
        else:
            self.list_vtu = self.list_vtu[first:last:interval]
            self.time_list = self.time_list[first:last:interval]
            self.first = first
            self.interval = interval
            self.last = last

        #Read VTU data
        N_vtu = len(self.list_vtu)
        pbar = tqdm(total = N_vtu, desc="Reading VTU files")
        for i in range(len(self.list_vtu)):
            #Read DF from VTU files
            self.reader(i)
            #exec(self.df_{i})
            pbar.update(1)

        print(f'Written .df_timestep from timestep = 0 to timestep = {len(self.list_vtu)-1}')

    #Write modifications on each df to VTU file
    def write_vtu(self):

        #Create list of time steps as strings
        time_list_str = [str(i) for i in self.time_list]
        time_list_str = [i.replace(".0", "") for i in time_list_str]

        #Write modified PVD to match new VTU files
        with open(f'{self.path_output}/{self.pvd_name}') as pvd_in:
            with open(f'{self.path_output}/mod_{self.pvd_name}', 'w') as pvd_out:
                for line in pvd_in:
                    #If line refers to vtu file
                    if "DataSet" in line:
                        #Check if timestep is within time_list of the read data
                        for timestep in time_list_str:
                            if timestep in line:
                                line = line.replace('.pvtu', '.0000.vtu')
                                line = line.replace('file="', 'file="mod_')
                                pvd_out.write(line)
                                time_list_str.remove(timestep)
                    
                    #Write config lines
                    else:
                        line = line.replace('.pvtu', '.0000.vtu')
                        line = line.replace('file="', 'file="mod_')
                        pvd_out.write(line)

        N_vtu = len(self.list_vtu)
        pbar = tqdm(total = N_vtu, desc="Writting new VTU and PVD files")
        for i in range(len(self.list_vtu)):
            #Write modified VTU file
            exec(f"self.df_{i}.save(f\'{self.path_output}/mod_{self.list_vtu[i]}\')")

            pbar.update(1)
        print("Modified .vtu and .pvd files with prefix mod_ successfully written")

    #Sort all data given reference array 
    def sort_by_array(self, reference_array_name):
        
        pbar = tqdm(total = len(self.time_list), desc = f"Sorting dataframe by {reference_array_name}")
        for i in range(len(self.time_list)):
            exec(f"self.df_{i}.points = self.df_{i}.points[self.df_{i}[reference_array_name].argsort()]")
            for name in self.df_0.array_names:
                exec(f"self.df_{i}[name] = self.df_{i}[name][self.df_{i}[reference_array_name].argsort()]")
            pbar.update(1)

    #Creates or modifies array
    def array_modifier(self, reference_array_name = "ID", new_array_name = "new_array", restart_array = False,  condition = "", array_values = 0, standard_value = 0, reference_time_step = 0, time_dependent = False, write_new_vtu = True):

        print("Generating array based on condition and array_value")

        #Sort all data by reference_array_name
        print(f"Sort array by {reference_array_name}")
        self.sort_by_array(reference_array_name)

        #Length of the new array
        exec(f'global new_array_len; new_array_len = len(self.df_{reference_time_step}[reference_array_name])')

        #Time array
        t = self.time_list

                #Create list of array names
        array_names = self.df_0.array_names
        array_names.append("x")
        array_names.append("y")
        array_names.append("z")
        array_names.append("u")
        array_names.append("v")
        array_names.append("w")
        array_names.append("f_x")
        array_names.append("f_y")
        array_names.append("f_z")
        array_names.append("t")

        #Restart array if asked or array does not exist
        if restart_array == True or new_array_name not in array_names:
            #Create array if does not exist
            new_array = np.repeat(standard_value, new_array_len)
            print(f"Creating array '{new_array_name}' with standard_value {standard_value}")

            #Push array to all pyvista arrays
            pbar = tqdm(total = len(self.time_list), desc = f"Creating array: {new_array_name}")
            for i in range(len(self.time_list)):
                exec(f'self.df_{i}[new_array_name] = new_array')
                pbar.update(1)

        else:
            #Reading array from reference timestep
            print("Reading previous array")
            new_array = eval(f'self.df_{reference_time_step}[new_array_name]')


        #Create a list of array names that are used either in
        #"conditions" or in "array_values"
        new_variables = set([])

        #Prepare "condition" and "array_value" for elementwise loop
        #Note that "k" is used here because it is the specific counter
        #that will be used for testing the "condition" further
        for name in array_names:
            if name in condition:
                condition = condition.replace(name, name + "[k]")

                #If one of the variables used in "condition"
                #is a pyvista array, create a list with the
                #name of the variable for further manipulation
                if name in self.df_0.array_names:
                    exec(f"global {name}; {name} = self.df_{reference_time_step}[name]")
                    new_variables.add(name)
        
        if type(array_values) == type(str()):
            for name in array_names:
                if name in array_values:
                    array_values = array_values.replace(name, name + "[k]")

                    #If one of the variable used in "array_value"
                    #is a pyvista array, create a list with the
                    #name of the variable for further manipulation
                    if name in self.df_0.array_names:
                        exec(f"global {name}; {name} = self.df_{reference_time_step}[name]")
                        new_variables.add(name)
        
        #If results vary with time:
        if time_dependent:
            ("Creating time-dependent array:")
            pbar = tqdm(total = len(self.time_list), desc = f"Looping through time-steps")
            for i in range(len(self.time_list)):
                #Assign velocities and positions to variables using the ith time step
                exec(f'global x; x = self.df_{i}.points[:, 0]')
                exec(f'global y; y = self.df_{i}.points[:, 1]')
                exec(f'global z; z = self.df_{i}.points[:, 2]')

                
                #In case velocity is written with caps V or v
                if "velocity" in self.df_0.array_names:
                    exec(f'global u; u = self.df_{i}["velocity"][:, 0]')
                    exec(f'global v; v = self.df_{i}["velocity"][:, 1]')
                    exec(f'global w; w = self.df_{i}["velocity"][:, 2]')

                
                elif "Velocity" in self.df_0.array_names:
                    exec(f'global u; u = self.df_{i}["Velocity"][:, 0]')
                    exec(f'global v; v = self.df_{i}["Velocity"][:, 1]')
                    exec(f'global w; w = self.df_{i}["Velocity"][:, 2]')

                #In case of FemForce
                if "FemForce" in self.df_0.array_names:
                    exec(f'global f_x; f_x = self.df_{i}["FemForce"][:, 0]')
                    exec(f'global f_y; f_y = self.df_{i}["FemForce"][:, 1]')
                    exec(f'global f_z; f_z = self.df_{i}["FemForce"][:, 2]')

                #Update lists used either in "condition" or "array_value":
                for variable in new_variables:
                    exec(f"{variable} = self.df_{i}[variable]")

                #Reading array from reference timestep
                new_array = eval(f'self.df_{i}[new_array_name]')

                #Fill new_array with array_value
                for k in range(new_array_len):
                    if eval(condition):
                        if type (array_values) == type(int(1)):
                            new_array[k] = array_values
                        elif type(array_values) == type(np.array([])) or type(array_values) == type([]):
                            new_array[k] = array_values[k]
                        else:
                            new_array[k] = eval(array_values)
                
                #Assign new_array to pyvista dataframe
                exec(f"self.df_{i}[new_array_name] = new_array")
                pbar.update(1)

        #If not time dependent
        else:
            print(f"Creating array based on time-step number: {reference_time_step}")
            print(f"Corresponding time: {self.time_list[reference_time_step]}")
            #Assign velocities and positions to variables using reference_time_step
            exec(f'global x; x = self.df_{reference_time_step}.points[:, 0]')
            exec(f'global y; y = self.df_{reference_time_step}.points[:, 1]')
            exec(f'global z; z = self.df_{reference_time_step}.points[:, 2]')

            
            #In case velocity is written with caps V or v
            if "velocity" in self.df_0.array_names:
                exec(f'global u; u = self.df_{reference_time_step}["velocity"][:, 0]')
                exec(f'global v; v = self.df_{reference_time_step}["velocity"][:, 1]')
                exec(f'global w; w = self.df_{reference_time_step}["velocity"][:, 2]')

            
            elif "Velocity" in self.df_0.array_names:
                exec(f'global u; u = self.df_{reference_time_step}["Velocity"][:, 0]')
                exec(f'global v; v = self.df_{reference_time_step}["Velocity"][:, 1]')
                exec(f'global w; w = self.df_{reference_time_step}["Velocity"][:, 2]')

            #In case of FemForce
            if "FemForce" in self.df_0.array_names:
                exec(f'global f_x; f_x = self.df_{reference_time_step}["FemForce"][:, 0]')
                exec(f'global f_y; f_y = self.df_{reference_time_step}["FemForce"][:, 1]')
                exec(f'global f_z; f_z = self.df_{reference_time_step}["FemForce"][:, 2]')

            #Fill new_array with array_value
            pbar = tqdm(total = new_array_len, desc = f"Creating new array named: {new_array_name}")
            for k in range(new_array_len):
                if eval(condition):
                    if type (array_values) == type(int(1)):
                        new_array[k] = array_values
                    elif type(array_values) == type(np.array([])) or type(array_values) == type([]):
                        new_array[k] = array_values[k]
                    else:
                        new_array[k] = eval(array_values)
                pbar.update(1)

            #Assign new_array to pyvista dataframe
            exec(f"self.df_{reference_time_step}[new_array_name] = new_array")
            
            #Use the same values for all time steps
            #Note that "reference_array_name" is used as criterium here
            #for sorting purposes, and that it can be changed
            #according to the user by changin the parameter "reference_array_name"
            #to any other array name in the original pyvista arrays
            #(self.df_0.array_names, for example)
            pbar = tqdm(total = len(self.time_list), desc = f"Assigning {new_array_name} to dataframes")
            for i in range(len(self.time_list)):
                exec(f"self.df_{i}[new_array_name] = self.df_{reference_time_step}[new_array_name]")
                pbar.update(1)
    
        if write_new_vtu:
            self.write_vtu()
