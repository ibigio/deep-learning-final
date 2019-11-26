#!/usr/bin/env python3
import math, random
import numpy as np

def mode(l):
	if len(l) is 0:
		return 0
	return max(set(l), key=l.count)

def prob_exactly(n, k, p):
	return choose(n,k) * (p ** k) * ((1-p) ** (n-k))

def choose(n, k):
	return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

def prob_at_least(n, k, v):
	if k > n or v is 0:
		return 0
	p = (1/6) if v is 1 else (1/3)
	total_sum = 0
	for i in range(k):
		total_sum += prob_exactly(n,i,p)
	return 1 - total_sum

def prob_call(total, call):
	return prob_at_least(total, call[0], call[1])

def make_call(total_dice, hand, prev_call):
	total = total_dice - len(hand)
	# starting call
	if prev_call[0] is 0:
		start_q = math.floor(total/3)
	prev_prob = prob_at_least(total, prev_call[0] - len([x for x in hand if x is prev_call[1] or x is 1]), prev_call[1])
	print('Oponent called', prev_call, 'with', prev_prob)
	prev_q = (prev_call[0] * 2) + 1 if prev_call[1] is 1 else prev_call[0]
	prev_v = 0 if prev_call[1] is 1 else prev_call[1]
	# best in same quantity
	m = mode([x for x in hand if x > prev_v and x is not 1])
	if m is 0 and 1 in hand:
		m = random.randint(prev_v + 1,6) if prev_v < 6 else 0
	bisq = (prev_q, m)
	# best in next quantity
	m = mode([x for x in hand if x <= prev_v and x is not 1])
	if m is 0 and 1 in hand:
		m = random.randint(2,prev_v + 1)
	binq = (prev_q + 1, m)
	# ace call
	ace_q = math.floor(prev_q / 2) + 1
	ace_call = (ace_q, 1)
	# determine best call
	possible_calls = [bisq, binq, ace_call]
	probs = [prob_call(total, (call[0] - len([x for x in hand if x is call[1] or (x is 1)]), call[1])) for call in possible_calls]
	opt_i = np.argmax(probs)
	print(possible_calls)
	print(probs)
	print("Best call is", possible_calls[opt_i], 'with', probs[opt_i])
	if prev_prob > (1 - probs[opt_i]):
		print('You should call.')
	else:
		print('You should pull.')
	return possible_calls[opt_i]

	