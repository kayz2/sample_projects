
def test_prime(n):
    """ Primality test of an integer greater than 1 """
    if n==2:
        return True
    #we only need to check divisors up to square root of a number
    for i in range(2,int(n**(.5))+1):
        if n % i ==0:
            return False
    else:
        return True

def list_of_first_primes(x):
    """ Creates a list of the first x primes """
    primes_1 =[]
    i=2
    while len(primes_1) < x:
        if test_prime(i):
            primes_1.append(i)
        i += 1
    return primes_1

primes = list_of_first_primes(123456)
sum_of_primes = sum(primes)




import pandas as pd

data_1 = pd.read_csv('/Users/krzysztofpawelec/Desktop/Python_projects/random/circle.csv')
#function that preprocesses pandas dataframe
def preprocess_data(data):
    """Preprocessig data. Input is a pandas dataframe with column names:
    point_id, point_x, point_y. Output are four dictionaries with point_id as
    common key and values: x coordiante, y coordiante, difference of consecutive
    x coordinates and difference of consecutive y coordinates
    """
    #take the diffrence of consecutive x coordiantes and consecutive y
    #coordinates, create new columns for each
    data['point_x_diff'] = data.point_x.diff()
    data['point_y_diff'] = data.point_y.diff()
    #create four dictionaries where key is point_id and values are point_x,
    #point_y, point_x_diff and point_y_diff
    x_coord_dict = dict(zip(data.point_id, data.point_x))
    y_coord_dict = dict(zip(data.point_id, data.point_y))
    x_coord_diff_dict = dict(zip(data.point_id, data.point_x_diff))
    y_coord_diff_dict = dict(zip(data.point_id, data.point_y_diff))
    return x_coord_dict, y_coord_dict, x_coord_diff_dict, y_coord_diff_dict

#anomaly detector
def find_anomaly(x_coord_dict, y_coord_dict, x_diff_dict, y_diff_dict):
    """Detects points on a circle, centered at (0,0), that are listed
     counterclockwise. Use output of preprocess_data as input.
      """
    for i in range(1,137):
        #subdivide into foru cases, each corresponding to a quadrant on the
        #cartesian coordiante plane
        #quadrant 4
        if 0 <= x_coord_dict.get(i) <= 1 and -1 <= y_coord_dict.get(i) <= 0 :
            #expected difference in x and y coordinates if coordiantes are
            #listed clockwise
            if x_diff_dict.get(i) <= 0  and y_diff_dict.get(i) <=0:
                pass
            #returns anomaly, i.e. counterclockwise listing
            else:
                return print(i)
        #quadrant 3
        elif -1 <= x_coord_dict.get(i) <= 0 and -1 <= y_coord_dict.get(i) <= 0:
            if x_diff_dict.get(i) <= 0  and y_diff_dict.get(i) >=0:
                pass
            else:
                return print(i)
        #quadrant 2
        elif -1 <= x_coord_dict.get(i) <= 0 and 0 <= y_coord_dict.get(i) <= 1:
            if x_diff_dict.get(i) >= 0  and y_diff_dict.get(i) >=0:
                pass
            else:
                return print(i)
        #quadrant 1
        elif 0 <= x_coord_dict.get(i) <= 1 and 0 <= y_coord_dict.get(i) <= 1:
            if x_diff_dict.get(i) >= 0  and y_diff_dict.get(i) <=0:
                pass
            else:
                return print(i)

a, b, c, d = preprocess_data(data_1)
anomaly = find_anomaly(a, b, c, d)
