import sys
from math import sqrt

class Route():
    
    def __init__(self, route, x, y, time):
        self.x = x
        self.y = y
        self.routeID = route
        self.time = time
        
    def distance(self, route2): # compute the distance between two routes
        return sqrt((self.x-route2.x)**2 + (self.y-route2.y)**2)
    
    def __str__(self):
        return 'route=%s, x=%d, y=%d' % (self.routeID, self.x, self.y)
    
    
def compute_tot_distance(route_list):
    tot_dist = 0
    for i in range(0, len(route_list)-1): # for each route of the bus
        current_route = route_list[i]
        next_route = route_list[i+1]
        dist = current_route.distance(next_route)
        tot_dist += dist
    return tot_dist

def compute_tot_time(route_list):
    tot_time = 0
    for r in route_list: # for each route of the bus
        tot_time += r.time
    return tot_time
  
if __name__ == "__main__":
    
    # dictionary <bus,[routes]>, all the routes covered by a bus
    bus_route = {}
    route_buses = {}
    with open(sys.argv[1]) as file:
        for line in file:
            busID, lineID = line.split()[0:2]; # i can use them as a string
            x, y, time = [int(i) for i in line.split()[2:]] # convert to integer
            
            # for each bus keep track of the list of routes
            r = Route(lineID, x, y, time)
            if busID not in bus_route:
                bus_route[busID] = [r] # add new pair <bus,route> to the dictionary
            else:
                bus_route[busID].append(r)
            
            # for each route keep track of the buses
            if lineID not in route_buses:
                route_buses[lineID] = [busID]
            else:
                route_buses[lineID].append(busID) 
                
    param = sys.argv[2]
    if param == '-b': # next parameter is the busID
        bus = sys.argv[3]
        route_list = bus_route[bus]
        tot_dist = compute_tot_distance(route_list)
        print('Distance for bus %s is %f' % (bus, tot_dist))
    elif param == '-l': # next parameter is the lineID
        line = sys.argv[3]
        bus_list = set(route_buses[line])
        bus_speeds = []
        for bus in bus_list:
            # for each bus compute the total distance and time amount
            route_list = bus_route[bus]
            tot_dist = compute_tot_distance(route_list)
            tot_time = compute_tot_time(route_list)
            bus_speeds.append(tot_dist / tot_time)
        print('Average speed for route %s is %f' % (line, sum(bus_speeds) / len(bus_speeds)))
    else:
        print('Error: parameter not valid')
        exit()        
            

    
            
            
            