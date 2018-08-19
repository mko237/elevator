import numpy as np
from Elevator import Elevator
from Passenger import Passenger

class Switch(object):
	def __init__(self):
		self.up = " "
		self.down = " "

	def reset(self):
		self.up = " "
		self.down = " "

# Building Class
class Building(object):
	def __init__(self, total_elevator_num, height, max_people):
		self.target = 0

		self.total_elevator_num = total_elevator_num
		self.max_people_in_floor = max_people
                self.num_actions = 5

		#each elevator has max capacity of 10
		self.elevators = []
		for idx in range(total_elevator_num):
			self.elevators.append(Elevator(idx, 10, height))

		self.height = height
		self.people_in_floors = []
		self.floor_button = []
		for idx in range(height):
			self.people_in_floors.append([])
			self.floor_button.append(Switch())

	def get_reward(self, prev_people):
                #energy = 0 if action ==
		res = self.get_arrived_people() - prev_people# - 1
		#res = (self.get_arrived_people() - prev_people) - self.get_building_capcity() # need to disable default cost in agent if used. setting a very low (negative default cost should disable
		res = (self.get_arrived_people() - prev_people)*(1+self.get_building_capacity()) - (self.max_people_in_floor+self.get_building_capacity()) # need to disable default cost in agent if used. setting a very low (negative default cost should disable
		#res = (self.get_arrived_people() - prev_people)*10  - (self.get_wait_time()*.001)
		#res = (self.get_arrived_people() - prev_people)*1  - (self.get_distance_from_max_wait_floor())
		#res = (self.get_arrived_people() - prev_people)*2*self.height  - (self.get_distance_from_max_wait_floor())-1
		#res = (self.get_arrived_people() - prev_people)*2*self.height  - (self.get_distance_from_max_wait_floor())-1
		#res = (self.get_arrived_people() - prev_people)*1  - self.get_people_to_move()
		return res
        def get_arrived_people_wait_time(self):
            pass
	# check number of people in ground floor
	def get_arrived_people(self):
		return len(self.people_in_floors[0])

	# this function is not currently used
	def get_wait_time(self):
		total = 0
		#for people in self.people_in_floors[1:]:
		#	for p in people:
		#		total += p.wait_time

		#for elevator in self.elevators:
		#	for p in elevator.curr_people:
		#		total += p.wait_time

                wait_time = []
		for people in self.people_in_floors[1:]:
                        floor_total = .1
			for p in people:
				floor_total += p.wait_time
                        wait_time.append(floor_total)

                wait_time = np.array(wait_time)
                wait_time = list(wait_time/wait_time.sum())

		return wait_time
        def get_people_to_move(self):
            num_people = self.target - len(self.people_in_floors[0])
            return num_people

        def get_building_capacity(self):
            num_people = self.get_people_to_move()
            max_people = (self.height-1)*self.max_people_in_floor
            return float(num_people)/float(max_people)

        def get_distance_from_max_wait_floor(self):
                distance = 0
                for e in  self.elevators:
                       distance += np.abs(e.curr_floor
                                      - (np.argmax(self.get_wait_time())
                                         + 1
                                        )
                                     )
                return distance

	# state of the building will be fed into the network as an input

	def get_state(self):
		res = [float(len(elem))/float(self.max_people_in_floor) if idx > 0 else float(len(elem))/float(self.target) for idx, elem in enumerate(self.people_in_floors)]

		for e in self.elevators:
			res.append(float(e.curr_floor)/float(self.height))
			res.append(float(len(e.curr_people))/float(e.max_people))

                for floor in self.get_wait_time():
                    res.append(floor)
                res.append(self.get_building_capacity())


                self.increment_wait_time()
		return res

	# clears the building
	def empty_building(self):
		self.people_in_floors = []
		for idx in range(self.height):
			self.people_in_floors.append([])

		for e in self.elevators:
			e.empty()
		self.target = 0

	def generate_people(self, prob):
		#generate random people in building and button press in each floor
		for floor_num in range(1, self.height):
			if np.random.random() < prob and len(self.people_in_floors[floor_num]) < self.max_people_in_floor:
				people = np.random.randint(1,6)
				if len(self.people_in_floors[floor_num]) + people > self.max_people_in_floor:
					people = self.max_people_in_floor - len(self.people_in_floors[floor_num])

				tmp_list = []
				for p in range(people):
					tmp_list.append(Passenger())

				self.people_in_floors[floor_num] += tmp_list
				self.target += people

                                # if np.random.random() < 0.5 and floor_num < self.height:
				# 	self.floor_button[floor_num].up = "^"
				# elif floor_num > 0:
				# 	self.floor_button[floor_num].down = "v"


	# actions can be redefined
	def perform_action(self, action):
		for idx,e in enumerate(self.elevators):
			if action[idx] == 4:
                                pass # no-op
			elif action[idx] == 3:
				# print "unload"
				res = e.unload_people(self.people_in_floors[e.curr_floor], self.max_people_in_floor)
				for p in res:
					self.people_in_floors[e.curr_floor].append(p)
			elif action[idx] == 2:
				# print "load"
                                #if e.curr_floor != 0:
                                self.people_in_floors[e.curr_floor] = e.load_people(self.people_in_floors[e.curr_floor])
			elif action[idx] == 1:
				# print "up"
				e.move_up()
			elif action[idx] == 0:
				# print "down"
				e.move_down()

	def increment_wait_time(self):
		for people in self.people_in_floors[1:]:
			for p in people:
				p.wait_time+=1

		for elevator in self.elevators:
			for p in elevator.curr_people:
				p.wait_time+=1

	def print_building(self, step,epsilon=None,show_floors=True,action=None,batch_num=None,random_action=None,reward=None):
		if show_floors:
			for idx in reversed(range(1,self.height)):
				print "======================================================="
				print "= Floor #%02d ="%idx,
				for e in self.elevators:
					if e.curr_floor == idx:
						print "  Lift #%d"%e.idx,
					else:
						print "         ",

				print " "
				# print "=   %c  %c   ="%(self.floor_button[idx].up, self.floor_button[idx].down),
				print "=  Waiting  =",
				for e in self.elevators:
					if e.curr_floor == idx:
						print "    %02d   "%len(e.curr_people),
					else:
						print "          ",
				print " "
				print "=    %03d    ="%len(self.people_in_floors[idx])


		print "======================================================="
		print "= Floor #00 =",
		for e in self.elevators:
			if e.curr_floor == 0:
				print "  Lift #%d"%e.idx,
			else:
				print "         ",

		print " "
		# print "=   %c  %c   ="%(self.floor_button[idx].up, self.floor_button[idx].down),
		print "=  Arrived  =",
		for e in self.elevators:
			if e.curr_floor == 0:
				print "    %02d   "%len(e.curr_people),
			else:
				print "          ",
		print " "
		print "=    %03d    ="%len(self.people_in_floors[0])
		print "======================================================="
		print ""
		print "People to move: %d "%(self.target - len(self.people_in_floors[0]))
		print "Total # of people: %d"%self.target
		print "Step: %d"%step
		if epsilon:
			print "Epsilon: %.3f, %s "%(epsilon,random_action)
		if action is not None:
			print "Action: %s"%action
		if batch_num is not None:
			print "Batch idx: %s"%batch_num
		if reward is not None:
			print "Reward: %.5f"%reward

                print self.get_wait_time()
                print self.get_distance_from_max_wait_floor()
