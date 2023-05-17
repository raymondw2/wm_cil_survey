import numpy as np
import torch
##Helper functions

def one_hot_lambda(grid_idx, lam):
    one_hot = np.zeros(lam*lam)
    one_hot[grid_idx] = 1
    return one_hot
def find_one(matrix):
    # Convert the matrix to a numpy array
    arr = np.array(matrix)
    # Find the indices of the non-zero elements in the array
    indices = np.argwhere(arr == 1)
    # If no non-zero element is found, return None
    if len(indices) == 0:
        return None
    # Extract the coordinates of the first non-zero element
    i, j = indices[0]
    # Return the coordinates as a tuple
    return np.array([i, j])


def get_grid_code(x,y,lam, dx = 0, dy = 0):
    return ((x+dx) % lam + ((y+dy) % lam)*lam)


class Wall():
    def __init__(self, left, right, lvalue, hvalue, name, size=5):
        self.left = left 
        self.right = right
        self.hvalue = hvalue #high value
        self.lvalue = lvalue #low value
        self._env = None
        self.noise = np.random.random(2*size+1)*0.1
        self.size = size
        self.name = name
        
        #self.another_post = another.current_location
        
        
    
    
    def generate_signal(self, point):
        self.point = point
        if self.left <= point and point <= self.right:
            return self.hvalue + self.noise[int(point)+self.size]
        else:
            return self.lvalue + self.noise[int(point)+self.size]
        
    def generate_signal_from_arr(self, arr):
        return np.array([self.generate_signal(point) for point in arr])
                
class SquareMaze():
    """
    TODO: Add boundaries
    """
    def __init__(self,size = 5, observation_size = 64, name = "Env 1"):
        self.name = name
        self.size = size
        self.goal = None
        self.locations = np.arange(size**2)
        
        self._observation_size = observation_size
        
        self.current_location = np.array([0,0])
        self.previous_move = 2 #initialized facing up, variable used to keep track of going up. 
        
        
        self._fov = 120
        
        
        self.possible_moves = np.array([[1, 0], [-1,0], [0,1], [0,-1]]) #Move left right up down (respectively)
        self.mesh_moves = np.array([[0,1], [0,-1], [1, 0], [-1,0]])
        # previous head direction
        self.prev_move = 2
            
        self.prev_move_global = 0
        
        self.AssignGoalLocation()
        self.success = False
        
        self.optimal_length = np.sum(np.abs(self.goal_location - self.current_location))
        
        self.observations = {}
        for i in np.arange(-size, size +1):
            self.observations[i] = {}
            for j in np.arange(-size, size +1):
                self.observations[i][j] = np.random.randn(self._observation_size)
        self.explain_global = ["east","west", "north", "south"]
        self.explain_allo = ["forward", "right", "back", "left"]
        self.allo_action_converter = np.asarray([[0, 2, 3, 1],
                                                 [2, 0, 1, 3],
                                                 [1, 3, 0, 2],
                                                 [3, 1, 2, 0]])
        self.action_converter = np.asarray([
                            [0, 1, 2, 3],
                            [1, 0, 3, 2],
                            [2, 3, 1, 0],
                            [3, 2, 0, 1]])
    
        self.generate_walls()
        
        print("Initialized")
        print("Goal Location: ", self.goal_location)
    
    
    def get_all_grid_code(self, dx = 0, dy = 0):
        l1, l2, l3 = 11, 12, 13

        #get grid code for each lambda 

        x, y = self.current_location

        #1 small because not counting walls
        new_size = self.size - 1

        new_x = x + new_size 
        new_y = new_size - y

        concat_one_hot = []
        for lam in [l1, l2, l3]:
            one_gen = one_hot_lambda(get_grid_code(new_x, new_y, lam, dx, dy), lam)
            concat_one_hot.append(one_gen)

        concat_one_hot = np.concatenate(concat_one_hot)
        
        return concat_one_hot
    
    def get_grid_code(self, device, dx = 0, dy = 0):
        l1_code, l2_code, l3_code = self.get_2d_grid_code(device, dx, dy)
        

        lams = [11,12,13]

        all_phase_diff = []
        for i, gc in enumerate([l1_code, l2_code, l3_code]):
            curr_loc = find_one(gc[0][0].cpu().numpy())
            curr_goa = find_one(gc[0][1].cpu().numpy())

            one_phase_diff = (curr_loc - curr_goa + lams[i]) % lams[i]
            all_phase_diff.append(one_phase_diff)

        return np.hstack(all_phase_diff)

    def get_2d_grid_code(self, device, dx = 0, dy = 0):
        l1, l2, l3 = 11, 12, 13

        #get grid code for each lambda 

        x, y = self.current_location

        #1 small because not counting walls
        new_size = self.size - 1

        new_x = x + new_size 
        new_y = new_size - y
        
        #goal x, goal y
        gx, gy = self.goal_location
        
        goal_x = gx + new_size
        goal_y = new_size - gy 

        concat_one_hot = []
        
        def helper(lam, x, y, gx, gy, dx, dy, device):
            lam_code = one_hot_lambda(get_grid_code(x, y, lam, dx, dy), lam).reshape((lam,lam))
            lam_goal_code = one_hot_lambda(get_grid_code(gx, gy, lam, dx, dy), lam).reshape((lam,lam))
            
            return torch.Tensor(np.stack([lam_code, lam_goal_code])).unsqueeze(0).to(device)
        
        
        l1_code = helper(l1, new_x, new_y, goal_x, goal_y, dx, dy, device)
        l2_code = helper(l2, new_x, new_y, goal_x, goal_y, dx, dy, device)
        l3_code = helper(l3, new_x, new_y, goal_x, goal_y, dx, dy, device)
        
        return l1_code, l2_code, l3_code
    
    def generate_walls(self, ):
        self.nwall = Wall(-self.size/2, self.size/2, np.random.rand(), np.random.rand()+2, "n", self.size)
        self.swall = Wall(self.size/2, -self.size/2, np.random.rand(), np.random.rand()+2, "s", self.size)
        self.ewall = Wall(self.size/2, -self.size/2, np.random.rand(), np.random.rand()+2, "e", self.size) #Left and right endpoints are flipped for facing the wall coordinates
        self.wwall = Wall(-self.size/2, self.size/2, np.random.rand(), np.random.rand()+2, "w", self.size)
        
    
    def distance_to_wall(self,):
        '''Calculate the distance to wall'''
        
        self.n = self.size - self.current_location[1] 
        self.s = np.abs(-self.size-self.current_location[1])
        self.e = self.size-self.current_location[0]
        self.w = np.abs(-self.size - self.current_location[0])
    
    def nsew_r_multiplier(self, name):
        """Flip multipler for right value
            returns the sign of the right part of the wall for 0,0 coordinates
        """
        if name == "n" or name == "w":
            return 1
        else:
            return -1
    
    
    def observation_signal(self, front, left, right, fdis, ldis, rdis):
        """Input wall value """
        fname = front.name
        lname = left.name
        rname = right.name
        
        y,left_from_front, right_from_front = self.observation_helper(fdis, ldis, rdis)
        
        indx = 0 if fname in ["n", "s"] else 1
        
        vision_length = 0
        front_length = 0

        if left_from_front is None:
            front_left_endpoint = self.current_location[indx]-y
            left_left_endpoint = None
            left_right_endpoint = None
            
            
            left_length = 0
            front_length += y
            vision_length += y
        else:
            front_left_endpoint = -self.nsew_r_multiplier(fname)*self.size
            left_left_endpoint = left_from_front
            if left_from_front >= self.size:
                left_left_endpoint = self.size - left_from_front
            left_right_endpoint = self.nsew_r_multiplier(lname)*self.size

            vision_length += self.size #Length from the front
            front_length += self.size

            left_length = np.abs(left_left_endpoint)

            vision_length += left_length # Add extra part

        if right_from_front is None:
            #print('Here')
            front_right_endpoint = self.current_location[indx]+self.nsew_r_multiplier(fname)*y
            self.y = y
            right_right_endpoint = None
            right_left_endpoint = None

            right_length = 0
            vision_length += y #Adding y if we don't hit right wall
            front_length += y
        else:
            #print("there")
            front_right_endpoint = self.nsew_r_multiplier(fname)*self.size
            right_right_endpoint = right_from_front #Point on the right side from the front
            
            right_left_endpoint = -self.nsew_r_multiplier(rname)*self.size
            
            if right_from_front >= self.size:
                #print("this")
                right_right_endpoint = self.size - right_from_front
            
            

            vision_length += self.size #Length from the front
            front_length += self.size

            right_length = np.abs(right_right_endpoint)
            vision_length += right_length # Add extra part
        
        front_percent = np.round(front_length / vision_length * self._observation_size)
        
        left_percent = np.round(left_length / vision_length * self._observation_size)
        right_percent = np.round(right_length / vision_length * self._observation_size)
        
        
        if left_percent == 0:
            if right_percent == 0:
                front_percent = self._observation_size
            else:
                right_percent = self._observation_size - front_percent
        else:
            if right_percent == 0:
                left_percent = self._observation_size - front_percent
            else:
                right_percent = self._observation_size - left_percent-front_percent
                
        self.front_right_endpoint = front_right_endpoint
        
        front_vision_idx = np.linspace(front_left_endpoint, front_right_endpoint, int(front_percent))
        self.front_vision_idx = front_vision_idx
        front_vision = front.generate_signal_from_arr(front_vision_idx)
        
        
        
        
        leftover_size = self._observation_size -front_vision_idx.shape[0]
        
        if left_percent != 0:
            left_vision_idx = np.linspace(left_left_endpoint, left_right_endpoint, int(left_percent))
            
            left_vision = left.generate_signal_from_arr(left_vision_idx)
            
            total_vision = np.append(left_vision, front_vision)
            
            
        else:
            total_vision = front_vision
            
        if right_percent != 0:
            right_vision_idx = np.linspace(right_left_endpoint, right_right_endpoint, int(right_percent))
            right_vision = right.generate_signal_from_arr(right_vision_idx)
            
            total_vision = np.append(total_vision, right_vision)
            
            self.debug = right_vision
            
        
        return total_vision #+ self.observations[self.current_location[0]][self.current_location[0]]
        
        #print(f"Left: {left_length}, Right: {right_length}, Front: {front_length}")
        
        #print(f"Left: {left_percent}, Right: {right_percent}, Front: {front_percent}")
        
    def observation_helper(self, front, left, right):
        
        """Computes the end points on the wall, returns dictionary of left right and up points
        Front: (left end, right end)
        
        If vision signal includes left and right walls
        Left: (left end, right end) 
        Right: (left end, right end)
        
        
        """
        half_fov = self._fov/2
        half_fov_rad = np.deg2rad(half_fov)
        comp_half_fov = 90-half_fov
        
        comp_half_fov_rad = np.deg2rad(comp_half_fov)
        
        y = front * np.tan(half_fov_rad)
                
        if left < y:
            left_from_front = (y - left) * np.tan(comp_half_fov_rad)
        else:
            left_from_front = None
        
        if right < y:
            right_from_front = (y - right) * np.tan(comp_half_fov_rad)
        else:
            right_from_front = None
        
        return y, left_from_front, right_from_front
                 
                
            
            
        
        
    def AssignGoalLocation(self,):
        self.goal_location = np.random.randint(-self.size+1, self.size, 2) #Plus 1 to avoid on wall
        
    def place_goal_location(self, place):
        print(f"New Placed Goal Location: {place}")
        self.goal_location = place
        
    def place_agent_random(self,):
        self.current_location = np.random.randint(-self.size+1, self.size, 2) #Plus 1 to avoid on wall, no need for second as it is up to but not inclusive
        
    def place_agent_specific(self, place):
        self.current_location = place
        
    def distance_from_point_to_goal(self,point):
        return np.linalg.norm(point - self.goal_location)
    def distance_to_goal(self,):
        return self.distance_from_point_to_goal(self.current_location)
    def find_optimal_move(self):
        return np.argmin(
            [self.distance_from_point_to_goal(self.current_location+movement) 
                 for movement in self.possible_moves]
        )
    def step(self, move):
        if self.check_success():  
            self.success = True
        else:
            move_idx = move #self.action_converter[self.prev_move][move]
            if np.max(np.abs(self.current_location+self.possible_moves[move_idx])) < self.size:
                self.current_location = self.current_location + self.possible_moves[move_idx]

                self.prev_move = move

                self.prev_move_global = move_idx
            else:
                
                #TODO add bounce
                if self.prev_move_global == 0:
                    self.prev_move_global = 1
                elif self.prev_move_global == 1:
                    self.prev_move_global = 0
                elif self.prev_move_global == 2:
                    self.prev_move_global = 3
                else:
                    self.prev_move_global = 2
                
                #print("Violation")
                #pass

        
    def check_success(self,):
        return np.all(self.current_location == self.goal_location)
    
    
    def get_vision(self,):
        self.distance_to_wall()
        
        if self.prev_move_global==0: #facing e
            return self.observation_signal(self.ewall, self.nwall, self.swall, self.e, self.n, self.s)
        elif self.prev_move_global == 1: #facing w
            return self.observation_signal(self.wwall, self.swall, self.nwall, self.w, self.s, self.n)
        elif self.prev_move_global == 2: #facing n
            return self.observation_signal(self.nwall, self.wwall, self.ewall, self.n, self.w, self.e)
        else:
            return self.observation_signal(self.swall, self.ewall, self.wwall, self.s, self.e, self.w)
    
    def reset(self, place = None, direction = 0):
        self.current_location = np.array([0,0])
        self.success = False
        self.prev_move_global = direction #Direction it faces
        if place is None:
            self.place_agent_random()
        else:
            self.place_agent_specific(place)
        self.optimal_length = np.sum(np.abs(self.goal_location - self.current_location))
        

#%%
