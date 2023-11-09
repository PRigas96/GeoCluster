import numpy as np
import random
import src.geometry as geo

def loadData(numberOfData):
    """
        Load data from data folder
    
        Parameters:
            numberOfData (int): number of data to load
            
        Returns:
            data (np.array): data

        Important:
            The data must be in the following format:
            [x0, y0, w, h, theta] where:
            x0, y0: coordinates of the center of the square
            w, h: width and height of the square
            theta: rotation angle of the square in rad
    """
    print("Loading data...")
    ref = './data/squares/' + str(numberOfData) +'/'+ str(numberOfData) 
    data = np.load(ref+'sq.npy')
    data[:, -1] = np.deg2rad(data[:, -1])
    datapoints = np.load(ref+'qp.npy')
    print("Data loaded.")

    return data, datapoints

def create_data_3d(numberOfData, x0, y0, z0, width, height, depth, theta, psi, phi, cube=True, axis_aligned=True):
    """
        Create numberOfData data
    
        Parameters:
            numberOfData (int): number of data to load
            
        Returns:
            data (np.array): data

        Important:
            The data are in the following format:
            [x0, y0, z0, w, h, d, theta, phi] where:
            x0, y0: coordinates of the center of the cuboid
            w, h, z: width, height and depth of the cuboid
            theta: rotation angle of the square in rad
            phi: second
    """
    data = []
    cnt = 0 #Keep track of cuboids created so far

    #Maximum number of collisions allowed - If reached, the generation of cuboids stops
    maximum_num_of_collisions = 0.05 * numberOfData
    num_of_collisions = 0

    while cnt != numberOfData:
        #Store the results for every 500 cuboids created, starting from 1.000 and ending to 10.000
        if 1000 <= cnt <= 10000 and (cnt - 1000) % 500 == 0:
            print("reached ", cnt, " squares!")
            np.save(f"./data_v2/10000sq/{cnt}sq_1_4.npy", data)

        print("creating square ", cnt, "\n")
        #Create the square with random center of mass
        x = random.choice(x0)
        y = random.choice(y0)
        z = random.choice(z0)

        #Check if the cuboid is actually a cube or not. Shape it accordingly
        if(cube):
            wid = hei = dep = random.choice(width)
        else:
            wid = random.choice(width)
            hei = random.choice(height)
            dep = random.choice(depth)

        if(axis_aligned):
            th = ps = ph = 0
        else:
            th = random.choice(theta)
            ps = random.choice(psi)
            ph = random.choice(phi)

        current_cuboid = np.array([x, y, z, wid, hei, dep, th, ps, ph])
        #Check if the current cuboid intersects with any of the cuboids in the data
        if not geo.check_intersection(data, current_cuboid):
            data.append(current_cuboid)
            cnt += 1
            num_of_collisions = 0
        else:
            num_of_collisions += 1
            print("num of collisions is: ", num_of_collisions)
            #Terminate if maximum number of collisions reached
            if num_of_collisions == maximum_num_of_collisions:
                print(
                    "max number of collisions reached! dataset creation terminates!\n"
                )
                break
    return data
    
