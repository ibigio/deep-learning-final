import math, random
import numpy as np
import re

class Game:
	
	def __init__(self, players, num_dice=5):
		self.players = players
		self.active_players = players
		self.rounds = []

	def play(self):
		starting_player_indx = 0
		while len(self.active_players) > 1:

			print(starting_player_indx)
			r = Round(self.active_players, starting_player_indx)
			loser = r.play_round()
			self.roudnds.append(r)
			if len(self.active_players[loser].cup.dice) is 0:
				del self.active_players[loser]
			starting_player_indx = loser % len(self.active_players)

		print('Player',self.active_players[0],'wins the game with',len(self.active_players[0].cup.dice),'dice!')
		return self.active_players[0]



class Round:
	
	def __init__(self, players, starting_player_indx=0):
		self.players = players
		self.__hands = [player.cup.shake().dice for player in self.players]
		self.total_dice = sum([len(h) for h in self.__hands])
		self.calls = []
		self.cur_call = None
		self.starting_player_indx = starting_player_indx
		self.cur_player_indx = starting_player_indx
		# self.direction = None

	def print_round(self):
		print('==================')
		print('Players:', [p.name for p in self.players])
		print('Dice:', self.total_dice)
		# print('->' if self.direction is 1 else '<-')
		print('Calls:', [str(call) for call in self.calls])
		print('Starter:', self.starting_player_indx)
		print(self.players[self.prev_player()], 'calls', self.cur_call)
		print('Player', self.players[self.cur_player_indx], 'calls:')

	def play_round(self):
		while True:

			# get call
			call = None
			self.print_round()
			while not self.call_is_valid(call):
				call = self.players[self.cur_player_indx].make_call(self)

			self.calls.append(call)

			# handle pull
			if call.q is 0:
				break

			# next player
			self.cur_call = call
			self.cur_player_indx = self.next_player()

		loser = self.determine_loser()
		self.players[loser].cup.lose_one()
		print('Player', self.players[loser], 'loses.')
		for h in self.__hands:
			print(h)
		return loser

	def determine_loser(self):
		count = 0
		for hand in self.__hands:
			count += len([x for x in hand if x is 1 or x is self.cur_call.v])
		if count >= self.cur_call.q:
			return self.cur_player_indx
		else:
			return self.prev_player()

	def call_is_valid(self, call):
		if call is None:
			return False
		# this is a pull
		if call.q is 0:
			return True
		if call.v is 0:
			return False
		if self.cur_call is None:
			return True

		minv = 0 if self.cur_call.v is 1 else self.cur_call.v
		minq = (self.cur_call.q * 2) + 1 if self.cur_call.v is 1 else self.cur_call.q

		# deal with aces
		if call.v is 1:
			print(1)
			if call.q < math.floor(minq / 2) + 1:
				print(2)
				return False
			else:
				return True

		if call.q < minq:
			return False
		if call.q == minq and call.v < minv:
			return False

		return True

	def next_player(self):
		return (self.cur_player_indx + 1) % len(self.players)

	def prev_player(self):
		return (self.cur_player_indx - 1) % len(self.players)


class Call:

	def __init__(self, quantity, value):
		self.q = quantity
		self.v = value

	def __str__(self):
		return '(%s, %s)' % (self.q, self.v)


class Player:

	def __init__(self, name='Player'):
		self.name = name
		self.cup = Cup()

	def __str__(self):
		return self.name

	def make_call(self, game_round):
		print(self.cup.dice)
		call_array = []
		while len(call_array) is not 2:	
			call_array = re.findall(r'\d+',input())
		return Call(int(call_array[0]), int(call_array[1]))

	# def choose_direction(self):
	# 	print('Choose direction (0 for Left, 1 for Right):')
	# 	direction_array = []
	# 	while len(call_array) is not 1:	
	# 		call_array = re.findall(r'\d',input())
	# 	return 1 if call_array[0] is 1 else -1





class Cup:
	def __init__(self, num_dice=5):
		self.num_dice = num_dice
		self.dice = [self.roll_one() for i in range(self.num_dice)]

	def __str__(self):
		return str([die for die in self.dice])

	def roll_one(self):
		return random.randint(1, 6)

	def lose_one(self):
		self.num_dice -= 1
		return self

	def shake(self):
		self.dice = [self.roll_one() for i in range(self.num_dice)]
		return self


#  ____  _                            _    ___     
# |  _ \| | __ _ _   _  ___ _ __     / \  |_ _|___ 
# | |_) | |/ _` | | | |/ _ \ '__|   / _ \  | |/ __|
# |  __/| | (_| | |_| |  __/ |     / ___ \ | |\__ \
# |_|   |_|\__,_|\__, |\___|_|    /_/   \_\___|___/
#################|___/##############################

class SafeNaivePlayerAI:

	def __init__(self, name='AI'):
		self.name = name
		self.cup = Cup()

	def __str__(self):
		return self.name

	def mode(self,l):
		if len(l) is 0:
			return 0
		return max(set(l), key=l.count)

	def prob_exactly(self,n, k, p):
		return self.choose(n,k) * (p ** k) * ((1-p) ** (n-k))

	def choose(self,n, k):
		return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

	def prob_at_least(self,n, k, v):
		if k > n or v is 0:
			return 0

		p = (1/6) if v is 1 else (1/3)
		total_sum = 0

		for i in range(k):
			total_sum += self.prob_exactly(n,i,p)
		return 1 - total_sum

	def prob_call(self,total, call):
		return self.prob_at_least(total, call.q, call.v)

	def make_call(self, game_round):
		hand = self.cup.dice
		total = game_round.total_dice - len(self.cup.dice)
		prev_call = game_round.cur_call

		# starting call
		if prev_call is None:
			start_q = max(math.floor(total/3) - 1, 1)
			start_v = random.randint(2,6)
			return Call(start_q + len([x for x in hand if x is start_v or x is 1]), start_v)

		prev_prob = self.prob_at_least(total, prev_call.q - len([x for x in hand if x is prev_call.v or x is 1]), prev_call.v)
		print('Oponent called', prev_call, 'with', prev_prob)

		prev_q = (prev_call.q * 2) + 1 if prev_call.v is 1 else prev_call.q
		prev_v = 0 if prev_call.v is 1 else prev_call.v

		# best in same quantity
		m = self.mode([x for x in hand if x > prev_v and x is not 1])
		if m is 0 and 1 in hand:
			m = random.randint(prev_v + 1,6) if prev_v < 6 else 0
		bisq = Call(prev_q, m)

		# best in next quantity
		m = self.mode([x for x in hand if x <= prev_v and x is not 1])
		if m is 0 and 1 in hand:
			m = random.randint(2,prev_v + 1)
		binq = Call(prev_q + 1, m)

		# ace call
		ace_q = math.floor(prev_q / 2) + 1
		ace_call = Call(ace_q, 1)

		# determine best call
		possible_calls = [bisq, binq, ace_call]
		probs = [self.prob_call(total, Call(call.q - len([x for x in hand if x is call.v or (x is 1)]), call.v)) for call in possible_calls]
		opt_i = np.argmax(probs)

		print(possible_calls)
		print(probs)
		print("Best call is", possible_calls[opt_i], 'with', probs[opt_i])
		if prev_prob > (1 - probs[opt_i]):
			print('You should call.')
			return possible_calls[opt_i]
		else:
			print('You should pull.')
			return Call(0,0)

